"""Training-only calibration with an explicit joint auxiliary-gradient cap.

Prepared for the full constraint matrix; not used by the active BAND search.
Gradients are measured at detached FP32 full-resolution logits, not parameters.
"""
import math
import itertools
import numpy as np

ORDER = ('bands', 'edge', 'sdf', 'depth')


def components(arm):
    names = tuple(name for name in ORDER if name in arm.split('_'))
    if arm != 'dice' + ''.join('_' + name for name in names) or not names:
        raise ValueError('Expected a declared auxiliary objective')
    return names


def calibration_case(outputs, labels, arm, target_sdf=None, criterion=None, depth_weights=None):
    import torch
    from monai.losses import DiceLoss
    from scripts.adni_constraints import objectives as binary
    names = components(arm)
    if criterion is None:
        z = outputs.detach().float().requires_grad_(True)
        native = DiceLoss(to_onehot_y=True, softmax=True)(z, labels)
    else:
        from scripts.nnunet_band_objective import native_loss
        detached = [value.detach().float() for value in outputs]
        z = detached[0].requires_grad_(True)
        native = native_loss(detached, labels, criterion)
    terms = dict(bands=lambda: binary.bands()(z, labels).loss,
                 edge=lambda: binary.BinaryEdge()(z, labels),
                 sdf=lambda: binary.sdf_loss(z, labels, target_sdf),
                 depth=lambda: binary.depth_loss(z, labels, depth_weights))
    gradients = [torch.autograd.grad(native, z)[0]]
    gradients += [torch.autograd.grad(terms[name](), z)[0] for name in names]
    gram = np.empty((len(gradients), len(gradients)), dtype=np.float64)
    for i, left in enumerate(gradients):
        for j in range(i, len(gradients)):
            value = float((left * gradients[j]).mean())
            gram[i, j] = gram[j, i] = value
    if not np.isfinite(gram).all() or np.any(np.diag(gram) <= 0):
        raise ValueError('Degenerate or nonfinite calibration gradients')
    return dict(gradient_names=['native', *names], gradient_gram=gram.tolist(),
                native_rms=math.sqrt(gram[0, 0]))


def select_joint_weights(rows, arm, *, targets, component_cap, joint_target, joint_cap):
    names = components(arm)
    if not rows or min(component_cap, joint_target, joint_cap) <= 0:
        raise ValueError('Positive calibration targets and nonempty training cases required')
    if any(name not in targets or not math.isfinite(targets[name]) or targets[name] <= 0 for name in names):
        raise ValueError('Missing positive component target')
    if not all(math.isfinite(value) for value in (component_cap, joint_target, joint_cap)):
        raise ValueError('Nonfinite gradient cap')
    grams = []
    for row in rows:
        if row['gradient_names'] != ['native', *names]:
            raise ValueError('Calibration component order differs')
        gram = np.asarray(row['gradient_gram'], dtype=np.float64)
        if (gram.shape != (len(names)+1, len(names)+1) or not np.isfinite(gram).all()
                or np.any(np.diag(gram) <= 0)
                or not np.allclose(gram, gram.T, rtol=1e-6, atol=1e-30)
                or np.linalg.eigvalsh(gram).min() < -1e-5*np.diag(gram).max()):
            raise ValueError('Invalid gradient Gram matrix')
        grams.append(gram)
    grams = np.stack(grams)
    ratios = np.sqrt(np.diagonal(grams, axis1=1, axis2=2)[:, 1:] / grams[:, 0, 0, None])
    medians, p95s = np.median(ratios, axis=0), np.quantile(ratios, .95, axis=0)
    initial = np.minimum(np.asarray([targets[name] for name in names])/medians, component_cap/p95s)
    squared = np.einsum('i,nij,j->n', initial, grams[:, 1:, 1:], initial)
    if np.any(squared <= 0):
        raise ValueError('Degenerate combined auxiliary gradient')
    joint = np.sqrt(squared / grams[:, 0, 0])
    # Delayed SDF and independently ramped components may remove cancellation
    # present in the full sum. Bound every subset, including ramp endpoints.
    # At this calibration anchor, norm convexity bounds intermediate per-case
    # ramp values too. Fixed coefficients do not guarantee caps after the model
    # changes during optimization; this is not per-update gradient clipping.
    phases = {}
    shrink = 1.
    worst = np.zeros(len(rows))
    for mask in itertools.product((0., 1.), repeat=len(names)):
        if not any(mask):
            continue
        active = initial * mask
        squared_phase = np.einsum('i,nij,j->n', active, grams[:, 1:, 1:], active)
        phase = np.sqrt(np.maximum(squared_phase, 0.) / grams[:, 0, 0])
        worst = np.maximum(worst, phase)
        median, p95 = float(np.median(phase)), float(np.quantile(phase, .95))
        if median > 0:
            shrink = min(shrink, joint_target/median)
        if p95 > 0:
            shrink = min(shrink, joint_cap/p95)
        phases['+'.join(name for name, enabled in zip(names, mask) if enabled)] = dict(
            raw_median_ratio=median, raw_p95_ratio=p95)
    shrink = min(shrink, joint_target/float(np.median(worst)),
                 joint_cap/float(np.quantile(worst, .95)))
    coefficients = initial * shrink
    statistics = {name: dict(target=targets[name], raw_median_ratio=float(medians[i]),
                             weighted_median_ratio=float(medians[i]*coefficients[i]),
                             weighted_p95_ratio=float(p95s[i]*coefficients[i]))
                  for i, name in enumerate(names)}
    alignment = {}
    all_names = ['native', *names]
    for i in range(len(all_names)):
        for j in range(i+1, len(all_names)):
            cosine = grams[:, i, j] / np.sqrt(grams[:, i, i]*grams[:, j, j])
            alignment[all_names[i]+':'+all_names[j]] = dict(
                median_cosine=float(np.median(np.clip(cosine, -1, 1))),
                negative_cases=int((cosine < 0).sum()))
    statistics['joint'] = dict(target=joint_target, p95_cap=joint_cap, shrink=float(shrink),
                               weighted_median_ratio=float(np.median(joint)*shrink),
                               weighted_p95_ratio=float(np.quantile(joint, .95)*shrink),
                               gradient_space='detached_fp32_full_resolution_logits', alignment=alignment)
    statistics['joint']['active_subsets'] = {
        name: dict(weighted_median_ratio=value['raw_median_ratio']*shrink,
                   weighted_p95_ratio=value['raw_p95_ratio']*shrink)
        for name, value in phases.items()}
    statistics['joint']['worst_ramp_median_ratio'] = float(np.median(worst)*shrink)
    statistics['joint']['worst_ramp_p95_ratio'] = float(np.quantile(worst, .95)*shrink)
    return dict(zip(names, map(float, coefficients))), statistics


def extra_calibration_flops(arm, cases=10):
    """Conservative extra allowance beyond the existing measured calibration.

    Full logit-grid products/reductions and host coefficient arithmetic are
    overcharged; no extra model backward or optimization update is performed.
    """
    return cases * 32 * (2*64**3) * (len(components(arm))+1)**2
