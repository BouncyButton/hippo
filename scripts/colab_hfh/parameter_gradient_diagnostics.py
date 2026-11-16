"""Read-only parameter-space probe for nnU-Net's deep-supervised loss.

This diagnostic neither selects coefficients nor performs optimizer updates.
Its model backwards must be charged separately from optimization/calibration.
"""
from contextlib import nullcontext
import json
import math
import re
import time

import torch


def probe(model, image, labels, criterion, auxiliary, *, amp=False):
    """Measure native-head and auxiliary parameter gradients on one training crop.

    `auxiliary(outputs, labels)` must return the actual scalar weighted auxiliary
    objective. RNG, module modes, buffers and existing parameter .grad values are
    preserved. The callback must be a pure loss, not a stateful online controller.
    """
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters or not hasattr(criterion, 'weight_factors') or not hasattr(criterion, 'loss'):
        raise ValueError('Expected a trainable model and explicit deep-supervision wrapper')
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    devices = [image.device.index if image.device.index is not None else torch.cuda.current_device()] if image.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            context = torch.autocast('cuda', dtype=torch.bfloat16) if amp and image.is_cuda else nullcontext()
            with torch.enable_grad(), context:
                outputs = model(image)
                if not isinstance(outputs, (list, tuple)) or len(outputs) != len(criterion.weight_factors):
                    raise ValueError('Model outputs and deep-supervision weights differ')
                targets = [torch.nn.functional.interpolate(labels.float(), size=o.shape[2:], mode='nearest').long()
                           for o in outputs]
                native = criterion(outputs, targets)
                terms = {'native_total': native}
                for index, (output, target, weight) in enumerate(zip(outputs, targets, criterion.weight_factors)):
                    if weight:
                        terms[f'native_head_{index}'] = weight * criterion.loss(output, target)
                component_sum = sum(value for name, value in terms.items() if name != 'native_total')
                if not torch.allclose(native.detach(), component_sum.detach(), rtol=1e-5, atol=1e-7):
                    raise ValueError('Native head decomposition does not match the training criterion')
                terms['weighted_auxiliary'] = auxiliary(outputs, labels)
            if any(value.numel() != 1 or not bool(torch.isfinite(value).all()) for value in terms.values()):
                raise ValueError('Expected finite scalar diagnostic losses')
            names = list(terms)
            gradients = [torch.autograd.grad(value, parameters, retain_graph=i < len(terms)-1, allow_unused=True)
                         for i, value in enumerate(terms.values())]
            gram = [[sum(float((left.float() * right.float()).sum())
                         for left, right in zip(a, b) if left is not None and right is not None)
                     for b in gradients] for a in gradients]
            if not all(math.isfinite(value) for row in gram for value in row):
                raise ValueError('Nonfinite parameter diagnostic gradient')
            native_norm = math.sqrt(max(0., gram[0][0]))
            if not math.isfinite(native_norm) or native_norm <= 0:
                raise ValueError('Degenerate native parameter gradient')
            norms = {name: math.sqrt(max(0., gram[i][i])) for i, name in enumerate(names)}
            return dict(gradient_space='all_trainable_parameters', gradient_names=names,
                gradient_gram=gram, parameter_gradient_norms=norms,
                norm_relative_to_native={name:value/native_norm for name,value in norms.items()},
                cosine_with_native={name:gram[0][i]/(native_norm*norms[name]) if norms[name] else None
                                    for i,name in enumerate(names)},
                active_output_shapes=[list(o.shape) for o,w in zip(outputs,criterion.weight_factors) if w],
                native_head_weights=list(criterion.weight_factors),
                native_gradient_includes_all_active_heads=True, optimizer_updates=0,
                model_backward_probes=len(terms),
                interpretation='Training-crop diagnostic in evaluation mode. Parameter gradients, not SGD momentum or effective optimizer updates; no validation-driven coefficient selection.')
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, mode in modes:
            module.training = mode


def conservative_flops_allowance(native_update_flops, cases, active_heads):
    """Charge every measured gradient as a full native forward/backward/update."""
    if native_update_flops <= 0 or cases < 1 or active_heads < 1:
        raise ValueError('Positive cost, crop count and active-head count required')
    return int(native_update_flops) * int(cases) * (int(active_heads) + 2)


def selected_cases(names):
    """First crop from each of the first two training subjects; no score input."""
    selected, subjects = [], set()
    for name in names:
        subject = name.rsplit('_h', 1)[0]
        if subject not in subjects:
            selected.append(name)
            subjects.add(subject)
        if len(selected) == 2:
            return selected
    raise ValueError('Parameter probe requires two distinct training subjects')


def audit_final(model, patients, arm, device, amp, criterion, protocol, output, schedule):
    """Observe two predefined pairs per new NN candidate, after final evaluation."""
    config = json.loads((output/'config.json').read_text())
    if config['model_identity'] != 'nnunet' or config['seed'] != 1 or config['fold'] not in (0, 2):
        return
    path = output/'parameter_gradient_final.json'
    if path.exists():
        return
    from scripts.cross_dataset_campaign.capped_core import model_digest
    from scripts.adni_constraints import objectives as binary
    from scripts.colab_hfh.constraint_geometry import sdf_targets, depth_weights
    calibration = json.loads((output/'calibration.json').read_text())
    weights = calibration['weights']
    names = selected_cases(config['train_cases'])
    by_name = {patient['case_name']:patient for patient in patients}
    if set(by_name) != set(config['train_cases']):
        raise ValueError('Parameter-probe inputs differ from the training split')
    count = sum(bool(w) for w in criterion.weight_factors)
    allowance = conservative_flops_allowance(schedule['baseline_flops_per_update'],len(names),count)
    before = model_digest(model)
    report = dict(status='started', phase='after_final_training_and_validation_evaluation',
        optimization_updates=0, diagnostic_flops_allowance=allowance,
        accounting='Separate post-training diagnostic allowance; no optimization/calibration update',
        selected_training_cases=names, sampling_rule='first_crop_of_first_two_distinct_training_subjects',
        model_sha256_before=before, weights=weights, cases=[])
    started = time.monotonic()
    try:
        for name in names:
            patient = by_name[name]
            image, label = patient['image'][None].to(device), patient['label'][None].to(device)
            sdf = sdf_targets(label) if 'sdf' in arm else None
            depth = depth_weights(label,protocol['native_geometry'][name]['model_affine'],protocol['depth_geometry']) if 'depth' in arm else None
            def auxiliary(outputs, labels):
                z = outputs[0]
                value = z.float().sum()*0.
                if 'bands' in arm:value = value + weights['bands']*binary.bands()(z,labels).loss
                if 'edge' in arm:value = value + weights['edge']*binary.BinaryEdge()(z,labels)
                if 'sdf' in arm:value = value + weights['sdf']*binary.sdf_loss(z,labels,sdf)
                if 'depth' in arm:value = value + weights['depth']*binary.depth_loss(z,labels,depth)
                return value
            report['cases'].append(dict(case_name=name,**probe(model,image,label,criterion,auxiliary,amp=amp)))
        report['status'] = 'complete'
    except (ValueError, RuntimeError) as error:
        # Final predictions and scores were already written. Preserve a failed
        # diagnostic explicitly; it must never silently invalidate or rerun training.
        report.update(status='unavailable',error=str(error)[:500])
    after = model_digest(model)
    if before != after:
        raise ValueError('Parameter diagnostic changed the final model')
    report.update(model_sha256_after=after,seconds=time.monotonic()-started)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_report(report, config, schedule, weights, final_model_sha256=None):
    """Validate provenance and recompute reported gradient norms from the Gram matrix."""
    import numpy as np
    expected_cases = selected_cases(config['train_cases'])
    allowance = conservative_flops_allowance(schedule['baseline_flops_per_update'],2,3)
    if (config['model_identity'] != 'nnunet' or config['seed'] != 1 or config['fold'] not in (0,2)
            or report['optimization_updates'] != 0 or report['diagnostic_flops_allowance'] != allowance
            or report['selected_training_cases'] != expected_cases or report['weights'] != weights
            or report['model_sha256_before'] != report['model_sha256_after']
            or not re.fullmatch(r'[a-f0-9]{64}',report['model_sha256_before'])
            or report['phase'] != 'after_final_training_and_validation_evaluation'
            or report['status'] not in ('complete','unavailable')):
        raise ValueError('Parameter diagnostic scope, identity or accounting differs')
    if final_model_sha256 is not None and report['model_sha256_after'] != final_model_sha256:
        raise ValueError('Parameter diagnostic model differs from the final logit probe')
    if report['status'] == 'complete':
        if [row['case_name'] for row in report['cases']] != expected_cases:
            raise ValueError('Parameter diagnostic training cases differ')
        names = ['native_total','native_head_0','native_head_1','native_head_2','weighted_auxiliary']
        for row in report['cases']:
            gram = np.asarray(row['gradient_gram'],dtype=float)
            if (row['gradient_names'] != names or row['optimizer_updates'] != 0
                    or row['gradient_space'] != 'all_trainable_parameters'
                    or row['model_backward_probes'] != 5
                    or not np.allclose(row['native_head_weights'],[4/7,2/7,1/7,0])
                    or gram.shape != (5,5) or not np.isfinite(gram).all()
                    or not np.allclose(gram,gram.T,rtol=1e-6,atol=1e-20)
                    or np.any(np.diag(gram)<0) or gram[0,0]<=0):
                raise ValueError('Invalid parameter diagnostic Gram matrix or head definition')
            if np.linalg.eigvalsh(gram).min() < -1e-5*np.diag(gram).max():
                raise ValueError('Parameter diagnostic Gram matrix is not positive semidefinite')
            for i,name in enumerate(names):
                norm = math.sqrt(gram[i,i])
                ratio = norm/math.sqrt(gram[0,0])
                cosine = gram[0,i]/math.sqrt(gram[0,0]*gram[i,i]) if norm else None
                if (not math.isclose(norm,row['parameter_gradient_norms'][name],rel_tol=1e-8,abs_tol=1e-12)
                        or not math.isclose(ratio,row['norm_relative_to_native'][name],rel_tol=1e-8,abs_tol=1e-12)
                        or (cosine is None and row['cosine_with_native'][name] is not None)
                        or (cosine is not None and not math.isclose(cosine,row['cosine_with_native'][name],rel_tol=1e-8,abs_tol=1e-12))):
                    raise ValueError('Parameter diagnostic summary differs from measured Gram matrix')
    return allowance
