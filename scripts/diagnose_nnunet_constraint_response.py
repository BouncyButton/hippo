"""Replay the reference trajectory and measure class errors and auxiliary gradients.

The only new optimization is the unchanged 90-update baseline replay. All probes
use training volumes and frozen checkpoints; they never update model weights.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from campaign90 import ARMS, BUDGET, digest, write_json
from optimization90 import validation_updates
import train_optimization90_core as core
from train_optimization90 import configure
from nnunet_multi_objective import native_loss
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_loss, signed_distance_maps


def gradient_vector(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().float().reshape(-1)
                      for g, p in zip(gradients, parameters)])


def gradient_relation(auxiliary, reference):
    an, rn = auxiliary.norm(), reference.norm()
    if not torch.isfinite(an) or not torch.isfinite(rn) or rn <= 0:
        raise ValueError('Invalid gradient norm')
    return dict(norm_ratio=float(an/rn), cosine=float(torch.dot(auxiliary, reference)/(an*rn).clamp_min(1e-30)))


def diagnose_checkpoint(model, criterion, patients, checkpoint, weights, scale):
    """Measure every training volume; no validation-driven weight selection."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    before = core.model_digest(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for patient in patients:
        image = patient['image'][None].cuda()
        label = patient['label'][None].cuda()
        sdf = signed_distance_maps(patient['label'][None]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(image)
            base_loss = native_loss(outputs, label, criterion)
            full_loss = criterion.weight_factors[0]*criterion.loss(outputs[0], label)
            bands = OuterBoundaryBandLoss()(outputs[0], label).loss
            edge = BoundaryEdgeConsistencyLoss()(outputs[0], label)
            distance = signed_distance_loss(outputs[0], label, sdf)
        total_grad = gradient_vector(base_loss, parameters)
        full_grad = gradient_vector(full_loss, parameters)
        row = dict(case_name=patient['case_name'], native_loss=float(base_loss.detach()),
                   full_resolution_to_total=gradient_relation(full_grad, total_grad),
                   coarse_to_total=gradient_relation(total_grad-full_grad, total_grad))
        logits = outputs[0]
        gz_base = torch.autograd.grad(base_loss, logits, retain_graph=True)[0].detach().float()
        for name, loss in [('bands', bands), ('edge', edge), ('sdf', distance)]:
            gradient = gradient_vector(loss, parameters)
            gz = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach().float()
            raw = gradient_relation(gradient, total_grad)
            raw['full_resolution_logit_norm_ratio'] = float(gz.norm()/gz_base.norm())
            if name in weights:
                raw['calibrated_weight'] = weights[name]
                raw['ramp_scale'] = scale
                raw['effective_parameter_norm_ratio'] = raw['norm_ratio']*weights[name]*scale
                raw['effective_full_resolution_logit_norm_ratio'] = raw['full_resolution_logit_norm_ratio']*weights[name]*scale
            posterior = label[:,0] == 2
            if posterior.any():
                raw['posterior_gradient_rms'] = float(gz[:,2][posterior].square().mean().sqrt())
                raw['native_posterior_gradient_rms'] = float(gz_base[:,2][posterior].square().mean().sqrt())
            row[name] = raw
            del gradient, gz
        with torch.no_grad():
            probability = logits.float().softmax(1)
            truth = label[:,0].long()
            pred = probability.argmax(1)
            confusion = torch.bincount((truth*3+pred).reshape(-1), minlength=9).reshape(3,3)
            row['confusion_gt_rows_pred_columns'] = confusion.cpu().tolist()
            post = truth == 2
            row['posterior_probability_on_gt_posterior_mean'] = float(probability[:,2][post].mean())
            row['posterior_share_of_foreground_on_gt_posterior_mean'] = float(
                (probability[:,2]/probability[:,1:].sum(1).clamp_min(1e-30))[post].mean())
        rows.append(row)
        del total_grad, full_grad, outputs, base_loss, full_loss, bands, edge, distance, logits, gz_base
        print(json.dumps({'probe_case':patient['case_name'], 'checkpoint':str(checkpoint)}), flush=True)
    if core.model_digest(model) != before:
        raise ValueError('Read-only diagnostics changed the model')
    aggregate = {}
    for name in ('bands','edge','sdf'):
        keys = rows[0][name].keys()
        aggregate[name] = {k:float(np.median([r[name][k] for r in rows])) for k in keys}
    aggregate['full_resolution_to_total_norm_ratio_median'] = float(np.median(
        [r['full_resolution_to_total']['norm_ratio'] for r in rows]))
    aggregate['coarse_to_total_norm_ratio_median'] = float(np.median(
        [r['coarse_to_total']['norm_ratio'] for r in rows]))
    aggregate['posterior_probability_on_gt_posterior_mean'] = float(np.mean(
        [r['posterior_probability_on_gt_posterior_mean'] for r in rows]))
    aggregate['posterior_share_of_foreground_on_gt_posterior_mean'] = float(np.mean(
        [r['posterior_share_of_foreground_on_gt_posterior_mean'] for r in rows]))
    return dict(checkpoint_sha256=digest(checkpoint), model_sha256=before,
                scope='all ten unaugmented training volumes; no optimizer update',
                weights=weights, scale=scale, aggregate=aggregate, cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root/'source'
    manifest = json.loads((args.root/'source_manifest.json').read_text())
    for name, sha in manifest.items():
        if digest(source/name) != sha: raise ValueError('Frozen source changed: '+name)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 CUDA allocation required')
    torch.set_num_threads(4)
    directory = source/'experiments/four_models_six_optimization_31976_20261006/nnunet'
    protocol = json.loads((directory/'protocol.json').read_text())
    if protocol['budget_flops'] != BUDGET or digest(args.pkl) != protocol['pkl_sha256']:
        raise ValueError('Reference budget or data changed')
    for name, sha in protocol['training_source_sha256'].items():
        if digest(source/name) != sha: raise ValueError('Training source changed')
    configure(protocol, 'dice')
    original_validation = core.base.validation_steps
    core.base.validation_steps = lambda steps, count=20: sorted(set(original_validation(steps,count))|set(range(80,91)))
    original_save = core.base.save_checkpoint
    snapshots = args.root/'baseline_snapshots'
    snapshots.mkdir(exist_ok=True)
    def capture(path, payload):
        original_save(path, payload)
        if path.name=='checkpoint_latest.pt' and payload['step'] in (50,85,89,90):
            original_save(snapshots/f'update{payload["step"]}.pt',
                          dict(model=payload['model'], step=payload['step'], configuration=payload['configuration']))
    core.base.save_checkpoint = capture
    replay = args.root/'baseline_replay'
    write_json(args.root/'diagnostic_recipe.json', dict(script_sha256=digest(Path(__file__)),
               baseline_replay_updates=90, budget_flops=BUDGET,
               lr_horizon_updates=174, extra_validation_updates=list(range(80,91)),
               native_loss_optimizer_augmentation_unchanged=True,
               gradient_probe_data='ten training volumes only',
               analysis_compute_separate_from_training_budget=True,
               no_validation_weight_selection=True))
    try:
        core.run(SimpleNamespace(output=replay, protocol_dir=directory, model='nnunet', objective='dice',
                  fold=0, seed=0, pkl=args.pkl, smoke_steps=0, device='cuda', threads=4, resume=True))
    finally:
        core.base.save_checkpoint = original_save
    curve = {int(r['step']):r for r in csv.DictReader((replay/'metrics.csv').open())}
    reference = {int(r['step']):r for r in csv.DictReader((args.parent_root/'nnunet/dice/fold0/seed0/run/metrics.csv').open())}
    for step in reference:
        for key in ('macro_hard_dice','anterior_hard_dice','posterior_hard_dice','learning_rate'):
            if float(curve[step][key]) != float(reference[step][key]):
                raise ValueError('Replay changed the reference trajectory')
    comparisons = []
    for arm in ARMS:
        parent = args.parent_root/'nnunet'/arm/'fold0/seed0/run'
        metrics = {int(r['step']):r for r in csv.DictReader((parent/'metrics.csv').open())}
        row = metrics[89] if arm!='dice' else curve[89]
        comparisons.append(dict(objective=arm, updates=89,
             dice=float(row['macro_hard_dice']), anterior=float(row['anterior_hard_dice']),
             posterior=float(row['posterior_hard_dice']),
             gain_pp=100*(float(row['macro_hard_dice'])-float(curve[89]['macro_hard_dice']))))
    write_json(args.root/'matched89.json', dict(status='verified', reference_replay_exact=True,
                                              comparisons=comparisons))
    architecture = json.loads((directory/'nnunet_architecture.json').read_text())
    model = core.base.build_model('nnunet',architecture).cuda()
    criterion,_ = core.base.build_loss_and_optimizer('nnunet',model,protocol)
    split = json.loads((directory/'splits_10patients.json').read_text())[0]
    patients = core.base.PreparedPatients(core.base._load_pkl_dataframe(args.pkl),split['train'])
    band_cal = json.loads((args.parent_root/'nnunet/dice_bands/fold0/seed0/run/calibration.json').read_text())
    sdf_cal = json.loads((args.parent_root/'nnunet/dice_sdf/fold0/seed0/run/calibration.json').read_text())
    weights = {**band_cal['weights'],**sdf_cal['weights']}
    probes = {}
    for name, checkpoint, scale in [
        ('baseline50',snapshots/'update50.pt',1.0),
        ('baseline89',snapshots/'update89.pt',.78),
        ('bands89',args.parent_root/'nnunet/dice_bands/fold0/seed0/run/checkpoint_latest.pt',.78),
        ('sdf89',args.parent_root/'nnunet/dice_sdf/fold0/seed0/run/checkpoint_latest.pt',.78)]:
        probes[name] = diagnose_checkpoint(model,criterion,patients,checkpoint,weights,scale)
        write_json(args.root/'gradient_probes.json',probes)
    write_json(args.root/'diagnosis.json',dict(status='complete', reference_replay_exact=True,
               matched89=comparisons, gradients={k:v['aggregate'] for k,v in probes.items()},
               training_budget_flops=BUDGET, diagnostic_compute_excluded_and_reported=True,
               no_model_updates_during_probes=True))


if __name__=='__main__':
    main()
