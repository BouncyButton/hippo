"""Reproduce historical arithmetic, then measure each new shape/output count."""
import argparse
import gc
import json
from pathlib import Path
import sys

from scripts.unetr_subfields.contracts import ROOT, DATASETS, NAME
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'reports/compute_budget_5pct_fold0_seed0_20261003')]
import torch
import monai
from torch.utils.flop_counter import FlopCounterMode
import profile_reference as reference
from profile_intermediate_baselines import intermediate_supplement
from scripts.compute_constraint_objectives import training_loss
from scripts.unetr_subfields.core import build_model, model_kwargs
from scripts.cross_dataset_campaign.common import write, digest


def measure(classes, shape):
    torch.manual_seed(0)
    with torch.device('meta'):
        model = build_model(classes, shape).train()
        image = torch.randn((1, 1, *shape))
        target = torch.zeros((1, 1, *shape), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, foreach=False)
    reference.supplementary_flops = intermediate_supplement
    audit = reference.OperatorAudit()
    stages = {}
    with audit, FlopCounterMode(display=False, custom_mapping={
            torch.ops.aten.convolution_backward: reference.conv_backward_flops}) as counter:
        before = counter.get_total_flops()
        logits = model(image)
        stages['forward'] = counter.get_total_flops() - before
        audit.stage = 'loss'
        before = counter.get_total_flops()
        loss = training_loss(logits, target, 'dice', {}, .2)
        stages['loss'] = counter.get_total_flops() - before
        audit.stage = 'backward'
        before = counter.get_total_flops()
        loss.backward()
        stages['backward'] = counter.get_total_flops() - before
        audit.stage = 'optimizer'
        before = counter.get_total_flops()
        optimizer.step()
        stages['optimizer'] = counter.get_total_flops() - before
    result = dict(parameters=sum(p.numel() for p in model.parameters()),
                  kwargs=model_kwargs(classes, shape), input_shape=[1, 1, *shape],
                  update_flops=sum(stages.values()) + sum(audit.supplement.values()),
                  core_flops_per_stage=stages, supplementary_per_stage=audit.supplement,
                  core_operator_flops={str(k): int(v) for k, v in counter.flop_counts['Global'].items()},
                  operator_audit=audit.ops)
    del model, optimizer, loss, logits
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'reports' / NAME / 'profiles')
    args = parser.parse_args()
    if monai.__version__ != '1.6.0' or not torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the historical profiling runtime: torch 2.2.2 / MONAI 1.6.0')
    torch.set_num_threads(4)
    path = ROOT / 'reports/unetr_candidate_20261005/unetr_vit_b16_profiles.json'
    old = json.loads(path.read_text())['profiles'][0]
    control = measure(3, (64, 64, 64))
    if control['update_flops'] != old['flops_per_update'] or control['parameters'] != old['parameters']:
        raise ValueError(f'Historical arithmetic not reproduced: {control["update_flops"]} != {old["flops_per_update"]}')
    write(args.output / 'historical_control.json', dict(**control, reference_sha256=digest(path), status='exact_match'))
    print('Historical update reproduced:', control['update_flops'], flush=True)
    for name, spec in DATASETS.items():
        result = measure(len(spec['classes']), spec['spatial_size'])
        result.update(dataset=name, runtime=dict(torch=torch.__version__, monai=monai.__version__, device='meta'),
                      historical_control_sha256=digest(args.output / 'historical_control.json'),
                      source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
                          Path(__file__), ROOT / 'scripts/unetr_subfields/core.py',
                          ROOT / 'scripts/profile_intermediate_baselines.py',
                          ROOT / 'scripts/profile_unet_constraint_budget.py', Path(reference.__file__)]},
                      accounting='Historical MAC=2 dense forward/backward plus declared scalar/loss/AdamW estimates; validation, data transforms and I/O excluded. Not hardware-counter measurements.')
        write(args.output / f'{name}.json', result)
        print(name, result['update_flops'], flush=True)


if __name__ == '__main__':
    main()
