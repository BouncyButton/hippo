"""Bind the shorter horizon to the unchanged, already smoke-tested nnU-Net runtime."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path

from run_budget_baselines import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--parent-stage', type=Path, required=True)
    args = parser.parse_args()
    source = args.stage / 'source'
    manifest_path = args.stage / 'source_manifest.json'
    for name, sha in json.loads(manifest_path.read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('New frozen source differs: ' + name)
    prior_manifest = args.parent_stage / 'source_manifest.json'
    prior = json.loads((args.parent_stage / 'preflight.json').read_text())
    if (prior['status'] != 'passed' or prior['source_manifest_sha256'] != digest(prior_manifest)
            or not any(r['arm'] == 'baseline' and r['two_updates_and_resume_passed']
                       for r in prior['checks'])):
        raise ValueError('Parent baseline smoke certificate differs')
    for name, sha in json.loads(prior_manifest.read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('Previously tested runtime changed: ' + name)
    directory = source / 'experiments/nnunet_10epochs_fixedlr_20261005'
    plan = json.loads((directory / 'plan.json').read_text())
    protocol_path = directory / 'baseline/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    original_path = source / 'experiments/nnunet_pair_62166_20261005/baseline/protocol.json'
    original = json.loads(original_path.read_text())
    for key in ['initial_lrs', 'nnunet_sgd_momentum', 'nnunet_warmup_fraction',
                'sgd_weight_decay', 'augmentation', 'precision',
                'splits_sha256', 'nnunet_architecture_sha256', 'pkl_sha256']:
        if protocol[key] != original[key]:
            raise ValueError('Baseline recipe changed: ' + key)
    if (protocol['nnunet_lr_horizon_updates'] != 174 or plan['lr_horizon_updates'] != 174
            or protocol['schedule_adapter_sha256'] != digest(source / 'scripts/train_nnunet_fixedlr_control.py')):
        raise ValueError('LR control adapter differs')
    import sys, math
    sys.path.insert(0, str(source / 'scripts'))
    from train_nnunet_fixedlr_control import fixed_horizon_learning_rate
    tree = ast.parse((source / 'scripts/train_compute_matched_baselines.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'learning_rate')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen native LR>', 'exec'), namespace)
    native = namespace['learning_rate']
    controlled = fixed_horizon_learning_rate(native, 174)
    for step in range(100):
        if controlled('nnunet', step, 100, .01, 0) != native('nnunet', step, 174, .01, 0):
            raise ValueError('LR prefix differs')
    entry = protocol['models']['nnunet']
    if (plan['model_order'] != ['nnunet'] or plan['arm_order'] != ['baseline']
            or plan['updates'] != {'baseline': 100} or (plan['fold'], plan['seed']) != (0, 0)
            or entry['flops_per_update'] != original['models']['nnunet']['flops_per_update']
            or entry['parameters'] != original['models']['nnunet']['parameters']
            or protocol['budget_flops'] != 100 * entry['flops_per_update']
            or protocol['budget_flops'] != plan['budget_flops_per_run']
            or plan['protocol_sha256']['baseline'] != digest(protocol_path)
            or protocol['parent_baseline_protocol_sha256'] != digest(original_path)):
        raise ValueError('Ten-epoch identity or arithmetic differs')
    steps = sorted({t // entry['flops_per_update'] for t in protocol['validation_target_flops']})
    if steps != list(range(5, 101, 5)):
        raise ValueError('Validation schedule differs')
    for key, filename in [('splits_sha256', 'splits_10patients.json'),
                          ('nnunet_architecture_sha256', 'nnunet_architecture.json')]:
        if digest(protocol_path.parent / filename) != protocol[key]:
            raise ValueError('Split or architecture differs')
    evidence = protocol['profile_evidence']
    if digest(source / evidence['profile_file']) != evidence['profile_sha256']:
        raise ValueError('Native update profile changed')
    data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    if digest(data) != protocol['pkl_sha256']:
        raise ValueError('Dataset changed')
    versions = {name: importlib.metadata.version(name)
                for name in ['torch', 'monai', 'dynamic-network-architectures']}
    # Wheel metadata omits the CUDA suffix; inspect the declared runtime build too.
    torch_version_path = importlib.metadata.distribution('torch').locate_file('torch/version.py')
    version_assignments = ast.parse(torch_version_path.read_text()).body
    runtime_version = next(ast.literal_eval(node.value) for node in version_assignments
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == '__version__'
                                   for target in node.targets))
    versions['torch_runtime_declared'] = runtime_version
    if (runtime_version != prior['torch'] or versions['monai'] != prior['monai']
            or versions['torch'].split('+')[0] != prior['torch'].split('+')[0]):
        raise ValueError('Previously tested runtime versions changed')
    write_json(args.stage / 'preflight.json', {
        'status': 'passed', 'plan_sha256': digest(directory / 'plan.json'),
        'source_manifest_sha256': digest(manifest_path), 'max_updates': 100, 'lr_horizon_updates': 174, 'all100_lr_values_match_parent': True,
        'validation_updates': steps, 'versions': versions,
        'parent_smoke_certificate_sha256': digest(args.parent_stage / 'preflight.json'),
        'basis': 'All inherited runtime files and optimization/data settings are unchanged; reuse the passed native baseline training/validation/resume smoke. Verify the100-update stopping cap and20 validation targets; all100 actual scheduler outputs exactly match the native174-update schedule.'})


if __name__ == '__main__':
    main()
