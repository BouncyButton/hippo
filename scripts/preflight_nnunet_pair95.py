"""Verify the calibrated bands cap and unchanged native runtime before submission."""
import argparse
import ast
import importlib.metadata
import json
import math
from pathlib import Path
import sys

from run_budget_baselines import digest, write_json


def extract_function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--parent-stage', type=Path, required=True)
    args = parser.parse_args()
    source = args.stage / 'source'
    manifest_path = args.stage / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, sha in manifest.items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    prior_path = args.parent_stage / 'preflight.json'
    prior = json.loads(prior_path.read_text())
    prior_manifest = args.parent_stage / 'source_manifest.json'
    if (prior['status'] != 'passed' or prior['source_manifest_sha256'] != digest(prior_manifest)
            or not any(r['arm'] == 'bands' and r['two_updates_and_resume_passed'] for r in prior['checks'])):
        raise ValueError('Native bands smoke certificate differs')
    for name, sha in json.loads(prior_manifest.read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('Previously smoke-tested source changed: ' + name)
    directory = source / 'experiments/nnunet_pair95_20261005'
    plan = json.loads((directory / 'plan.json').read_text())
    protocol_path = directory / 'bands/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    baseline_path = directory / 'baseline/protocol.json'
    baseline = json.loads(baseline_path.read_text())
    original = json.loads((source / 'experiments/nnunet_pair_62166_20261005/bands/protocol.json').read_text())
    for key in ['splits_sha256', 'nnunet_architecture_sha256', 'pkl_sha256', 'augmentation',
                'precision', 'sgd_weight_decay', 'nnunet_sgd_momentum', 'nnunet_warmup_fraction',
                'learning_rate_policy', 'nnunet_lr_horizon_updates']:
        if protocol[key] != baseline[key]:
            raise ValueError('Baseline pairing changed: ' + key)
    for key in ['constraint_warmup_updates', 'calibration_policy', 'constraint_scale',
                'training_loss', 'sources', 'profile_evidence']:
        if protocol[key] != original[key]:
            raise ValueError('Constraint recipe changed: ' + key)
    if (plan['arm_order'] != ['baseline', 'bands'] or plan['model_order'] != ['nnunet']
            or plan['updates'] != {'baseline': 95, 'bands': 91} or (plan['fold'], plan['seed']) != (0, 0)
            or plan['budget_flops_per_run'] != 33752920148595
            or protocol['budget_flops'] != 33752920148595 or baseline['budget_flops'] != 33752920148595
            or plan['protocol_sha256']['bands'] != digest(protocol_path)
            or protocol['baseline_protocol_sha256'] != digest(baseline_path)
            or protocol['initial_lrs']['nnunet'] != baseline['initial_lrs']['nnunet']
            or protocol['schedule_adapter_sha256'] != digest(source / 'scripts/train_nnunet_pair95.py')
            or protocol['lr_prefix_adapter_sha256'] != digest(source / 'scripts/train_nnunet_fixedlr_control.py')):
        raise ValueError('Authorized plan, adapter or baseline differs')
    evidence = protocol['profile_evidence']
    profile_path = source / evidence['profile_file']
    if digest(profile_path) != evidence['profile_sha256']:
        raise ValueError('Native bands profile changed')
    profile = json.loads(profile_path.read_text())
    for name, sha in profile['source_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError('Profile source changed: ' + name)
    costs = profile['costs']
    function = extract_function(source / 'scripts/compute_constraint_objectives.py', 'scheduled_costs', {})
    expected = function(protocol['budget_flops'], costs['baseline'], costs['bands'], costs['calibration_total'], 50)
    schedule = protocol['models']['nnunet']['objectives']['dice_bands']
    if (any(schedule[k] != v for k, v in expected.items()) or schedule['max_steps'] != 91
            or schedule['flops_per_update'] != costs['bands']
            or schedule['baseline_flops_per_update'] != costs['baseline']
            or protocol['models']['nnunet']['parameters'] != baseline['models']['nnunet']['parameters']):
        raise ValueError('Compute schedule differs')
    sys.path.insert(0, str(source / 'scripts'))
    from train_nnunet_fixedlr_control import fixed_horizon_learning_rate
    native = extract_function(source / 'scripts/train_compute_matched_baselines.py', 'learning_rate', {'math': math})
    controlled = fixed_horizon_learning_rate(native, 174)
    for cap in [95, 91]:
        for step in range(cap):
            if controlled('nnunet', step, cap, .01) != native('nnunet', step, 174, .01):
                raise ValueError('LR prefix differs')
    milestones = extract_function(source / 'scripts/train_budget_bands.py', 'validation_updates', {})(protocol, 'nnunet')
    if milestones != plan['validation_updates']['bands'] or not set(range(5, 91, 5)).issubset(milestones) or min(milestones) < 1:
        raise ValueError('Validation schedule differs')
    for key, filename in [('splits_sha256', 'splits_10patients.json'), ('nnunet_architecture_sha256', 'nnunet_architecture.json')]:
        if digest(protocol_path.parent / filename) != protocol[key]:
            raise ValueError('Split or architecture changed')
    data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    if digest(data) != protocol['pkl_sha256']:
        raise ValueError('Dataset changed')
    if (plan['arm_budget_flops'] != {'baseline': 33752920148595, 'bands': 33752920148595}
            or baseline['planned_updates'] != 95 or protocol['planned_updates'] != 91
            or baseline['budget_flops'] // costs['baseline'] != 95
            or baseline['validation_target_flops'] != [i * costs['baseline'] for i in range(5, 96, 5)]
            or plan['protocol_sha256']['baseline'] != digest(baseline_path)
            or baseline['schedule_adapter_sha256'] != protocol['schedule_adapter_sha256']
            or plan['maximum_common_budget_flops'] != 33752920148595
            or max(plan['arm_budget_flops'].values()) > plan['maximum_common_budget_flops']):
        raise ValueError('Baseline update allowance or common FLOP cap differs')
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'monai', 'dynamic-network-architectures']}
    tree = ast.parse(importlib.metadata.distribution('torch').locate_file('torch/version.py').read_text())
    runtime = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    if runtime != prior['torch'] or versions['monai'] != prior['monai']:
        raise ValueError('Native runtime changed')
    write_json(args.stage / 'preflight.json', {'status': 'passed',
               'plan_sha256': digest(directory / 'plan.json'), 'source_manifest_sha256': digest(manifest_path),
               'max_updates': {'baseline': 95, 'bands': 91}, 'lr_horizon_updates': 174, 'all95_lr_values_match_baseline': True,
               'validation_updates': milestones, 'schedule': schedule, 'baseline_budget_flops': baseline['budget_flops'], 'comparison_axis': 'training_flops', 'versions': versions,
               'parent_bands_smoke_certificate_sha256': digest(prior_path),
               'basis': 'Reuse passed native bands calibration/training/resume smoke with all inherited runtime files unchanged; exact schedule adapter checked at every update; calibration and band costs verified against source-bound profile.'})


if __name__ == '__main__':
    main()
