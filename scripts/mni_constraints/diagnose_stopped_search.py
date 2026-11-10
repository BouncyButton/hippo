"""Zero-update probes and retirement of the two user-abandoned target runs.

Execute against the search's frozen PYTHONPATH. Never imports a training entrypoint
or calls an optimizer. Checkpoints stay byte-identical until verified archival.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.unetr_subfields import core, cap_extension
from scripts.unetr_subfields.contracts import DATASETS
from scripts.mni_constraints import objectives
from scripts.unetr_subfields.mni_diagnostic import manifest
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands


def dot(a, b):
    return float((a.double() * b.double()).sum())


def gradient_stats(a, b):
    aa, bb, ab = dot(a, a), dot(b, b), dot(a, b)
    return dict(base_norm=aa**.5, constraint_norm=bb**.5,
                ratio=(bb / aa)**.5 if aa else None,
                cosine=ab / (aa * bb)**.5 if aa * bb else None)


def logit_probe(raw, label, identity, weights, scale):
    z = raw.detach().float().requires_grad_(True)
    base = objectives.base_loss(z, label, identity)
    bands = objectives.term('bands', z, label)
    edge = objectives.term('edge', z, label)
    gb = torch.autograd.grad(base, z, retain_graph=True)[0]
    gc = torch.autograd.grad(bands, z, retain_graph=True)[0]
    ge = torch.autograd.grad(edge, z, retain_graph=True)[0]
    weighted = weights['bands'] * scale * gc
    total = gb + weighted + weights['edge'] * scale * ge
    truth = label[:, 0]
    pred = z.argmax(1)
    fg = label > 0
    inner, outer = build_boundary_bands(fg, steps=2)
    support = (inner | outer)[:, 0]
    masks = dict(fn=(truth > 0) & (pred == 0), fp=(truth == 0) & (pred > 0),
                 swaps=(truth > 0) & (pred > 0) & (pred != truth))
    # A candidate for a subsequent controlled experiment, not a trained result:
    # equally balanced categorical CE on each class's two-step boundary support.
    categorical = []
    for c in range(1, 4):
        a, b = build_boundary_bands(label == c, steps=2)
        ce = F.cross_entropy(z, truth.long(), reduction='none')
        for mask in (a[:, 0], b[:, 0]):
            if mask.any():
                categorical.append(ce[mask].mean())
    alternative = torch.stack(categorical).mean()
    ga = torch.autograd.grad(alternative, z)[0]
    ga_equal = ga * (gc.norm() / ga.norm())
    # Positive margin_direction means a negative-gradient step increases
    # the correct-class logit relative to the current (wrong) argmax.
    def margin_direction(g):
        return (g.gather(1, pred[:, None]) - g.gather(1, truth[:, None].long()))[:, 0]
    directions = {k: margin_direction(g) for k, g in
                  dict(base=gb, bands=gc, weighted_bands=weighted, combined=total,
                       categorical_equal_norm=ga_equal).items()}
    errors = {}
    for name, mask in masks.items():
        errors[name] = dict(total=int(mask.sum()), in_band=int((mask & support).sum()),
                           outside_band=int((mask & ~support).sum()))
        for region, selected in [('all', mask), ('in_band', mask & support)]:
            errors[name][region + '_margin'] = {
                key: dict(improved=int((value[selected] > 0).sum()),
                          worsened=int((value[selected] < 0).sum()),
                          unchanged=int((value[selected] == 0).sum()),
                          sum=float(value[selected].double().sum()))
                for key, value in directions.items()}
        errors[name]['base_corrective_combined_reverses'] = int(
            (mask & (directions['base'] > 0) & (directions['combined'] < 0)).sum())
    probability = z.detach().softmax(1)
    return dict(losses=dict(base=float(base.detach()), bands=float(bands.detach()), edge=float(edge.detach())),
                gradient=gradient_stats(gb, weighted),
                unscaled_target_gradient=gradient_stats(gb, weights['bands'] * gc),
                joint_gradient=gradient_stats(gb, total-gb),
                categorical_gradient=gradient_stats(gb, ga_equal), errors=errors,
                union_positive_but_argmax_background=int(((probability[:, 1:].sum(1) > .5) & (pred == 0)).sum()),
                foreground_voxels=int(fg.sum()), inner_voxels=int(inner.sum()), outer_voxels=int(outer.sum()))


def parameter_probe(model, image, label, identity, weights, scale):
    # One fixed case only: all trainable parameters, BF16 forward / FP32 gradients.
    params = tuple(p for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        raw = model(image)
        base = objectives.base_loss(raw, label, identity)
        constraint = weights['bands'] * scale * objectives.term('bands', raw, label)
    gb = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(constraint, params, allow_unused=True)
    aa = bb = ab = 0.
    for a, b in zip(gb, gc):
        if a is not None: aa += dot(a, a)
        if b is not None: bb += dot(b, b)
        if a is not None and b is not None: ab += dot(a, b)
    return dict(base_norm=aa**.5, weighted_bands_norm=bb**.5,
                ratio=(bb/aa)**.5, cosine=ab/(aa*bb)**.5,
                mode='eval, first fixed training crop; not Adam-preconditioned update influence')


def run(stage, identifier):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    assert read(stage/'control.json')['pause'], 'Search must remain stopped'
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    task = next(r for r in read(stage/'plan.json')['tasks'] if r['id'] == identifier)
    output = stage/'runs'/identifier
    assert not (output/'completion.json').exists()
    assert not (output/'archive_manifest.json').exists()
    cfg = read(output/'config.json')
    folder = stage/'protocols'/task['protocol']
    p = read(folder/'protocol.json')
    assert digest(folder/'protocol.json') == cfg['protocol_sha256']
    before = {f.name: dict(sha256=digest(f), bytes=f.stat().st_size) for f in output.glob('*.pt')}
    cp = torch.load(output/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    assert cp['configuration'] == cfg
    ledger = read(output/'budget_ledger.json')
    assert cp['charged_attempts'] <= ledger['charged_attempts']
    cap_extension.bind(SimpleNamespace(protocol_dir=folder, model=p['trainer_alias'], objective=cfg['objective'], fold=cfg['fold']))
    model = core.base.build_model(p['trainer_alias'], read(folder/'nnunet_architecture.json')).to(device)
    model.load_state_dict(cp['model'])
    spec = DATASETS['MNI']
    assert digest(stage/'data/MNI.pkl') == cfg['dataset_sha256']
    frame = core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    calibration = read(output/'calibration.json')
    weights = calibration['weights']
    schedule = p['models'][p['trainer_alias']]['objectives'][cfg['objective']]
    scale = min(1., max(0., (cp['charged_attempts']-schedule['prefix_updates']) / schedule['constraint_warmup_updates']))
    result = dict(model=task['model'], step=cp['step'], charged_updates=cp['charged_attempts'],
                  scale=scale, weights=weights, calibration_target=calibration['target_ratio'],
                  source_sha256=digest(Path(__file__)), frozen_manifest_sha256=digest(stage/'manifest.json'),
                  checkpoints=before, additional_optimizer_updates=0,
                  notes=['Two interrupted bands+edge runs, different training stages; not final baseline predictions.',
                         'Logit gradients probe unaugmented evaluation mode; validation labels are diagnostic only.',
                         'Categorical CE comparison is a local direction test, not an observed Dice improvement.'], splits={})
    for split, names in [('training', cfg['train_cases']), ('validation', cfg['validation_cases'])]:
        data = core.PreparedSubfields(frame, names, spec)
        write(output/(split+'_interrupted.json'), core.evaluate(model, data, device, True, spec['classes']))
        model.eval()
        rows = []
        for item in data:
            image = item['image'][None].to(device)
            label = item['label'][None].to(device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                raw = model(image)
            rows.append(dict(case=item['case_name'], **logit_probe(raw, label, task['model'], weights, scale)))
        result['splits'][split] = rows
        if split == 'training':
            item = data[0]
            result['parameter_probe'] = dict(case=item['case_name'], **parameter_probe(
                model, item['image'][None].to(device), item['label'][None].to(device), task['model'], weights, scale))
        print(json.dumps(dict(run=identifier, evaluated=split)), flush=True)
    best = torch.load(output/'checkpoint_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    write(output/'validation_best_poststop.json', core.evaluate(model, core.PreparedSubfields(frame,cfg['validation_cases'],spec), device,True,spec['classes']))
    assert all(digest(output/name) == info['sha256'] for name, info in before.items())
    write(output/'mechanism_probe.json', result)
    write(output/'interrupted_protocol.json', p)
    write(output/'abandonment.json', dict(status='abandoned_by_user', original_run=identifier,
        completed_updates=cp['step'], charged_updates=ledger['charged_attempts'],
        consumed_training_flops=ledger['training_flops'], additional_optimizer_updates=0,
        diagnostic_source_sha256=digest(Path(__file__)), excluded_from_complete_results=True,
        ap_swaps_applicable=False, reason='User stopped gradient search for mechanism investigation.'))
    manifest(output, {**task, 'kind': 'abandoned_with_mechanism_probe'})
    print(json.dumps(dict(run=identifier, status='ready_for_verified_archive')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    run(args.stage, args.task)
