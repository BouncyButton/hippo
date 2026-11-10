"""Check bindings and exercise candidate training, validation and resume on synthetic data."""
import argparse
import json
from pathlib import Path
import sys

import torch

import train_simple_baseline_pilot as adapter
from run_budget_baselines import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--plan-dir', type=Path,
                        default=Path('experiments/simple_baseline_screen_20261005'))
    args = parser.parse_args()
    source = args.stage / 'source'
    for name, sha in json.loads((args.stage / 'source_manifest.json').read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    plan_path = source / args.plan_dir / 'plan.json'
    plan = json.loads(plan_path.read_text())
    data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    if digest(data) != plan['pkl_sha256']:
        raise ValueError('Dataset differs from plan')
    for run in plan['runs']:
        directory = plan_path.parent / run['name']
        protocol = json.loads((directory / 'protocol.json').read_text())
        split = json.loads((directory / 'splits_10patients.json').read_text())[0]
        if (len(set(split['train'])) != 10 or len(split['val']) != 52
                or set(split['train']) & set(split['val'])):
            raise ValueError('Split changed')
        cost = protocol['models']['unet']['flops_per_update']
        if (protocol['budget_flops'] != plan['budget_flops']
                or run['updates'] != plan['budget_flops'] // cost
                or run['updates'] * cost > plan['budget_flops']
                or (run['updates'] + 1) * cost <= plan['budget_flops']):
            raise ValueError('Schedule changed')
        evidence = protocol['profile_evidence']
        if digest(source / evidence['path']) != evidence['sha256']:
            raise ValueError('Profile changed')
        profile = json.loads((source / evidence['path']).read_text())
        if (profile['schedule']['flops_per_update'] != cost
                or profile['candidate'] != protocol['candidate']):
            raise ValueError('Preset differs from profiled architecture')

    # This smoke check uses synthetic images only, and never enters production output.
    generator = torch.Generator().manual_seed(913)
    labels = torch.zeros((1, 64, 64, 64), dtype=torch.long)
    labels[:, 20:32, 20:44, 20:44] = 1
    labels[:, 32:44, 20:44, 20:44] = 2
    image = labels.float() + .1 * torch.randn(labels.shape, generator=generator)
    adapter.base._load_pkl_dataframe = lambda path: None
    adapter.base.PreparedPatients = lambda dataframe, names: [
        {'image': image.clone(), 'label': labels.clone(), 'case_name': name}
        for name in (names if len(names) == 10 else names[:2])]
    checks = []
    for candidate in plan['candidate_order']:
        run = next(r for r in plan['runs'] if r['candidate'] == candidate)
        directory = plan_path.parent / run['name']
        output = args.stage / 'synthetic_smoke' / candidate
        sys.argv = ['train_simple_baseline_pilot.py', '--model', 'unet', '--fold', '0', '--seed', '0',
                    '--protocol-dir', str(directory), '--output', str(output), '--pkl', str(data),
                    '--device', 'cpu', '--threads', '4', '--smoke-steps', '2', '--resume']
        adapter.main()
        completion = json.loads((output / 'completion.json').read_text())
        metrics_sha = digest(output / 'metrics.csv')
        ledger_sha = digest(output / 'budget_ledger.json')
        adapter.main()
        if (completion['status'] != 'smoke_complete' or completion['updates'] != 2
                or completion['training_attempts_charged'] != 2 or completion['discarded_attempts'] != 0
                or metrics_sha != digest(output / 'metrics.csv')
                or ledger_sha != digest(output / 'budget_ledger.json')):
            raise ValueError('Smoke completion or idempotent resume failed')
        checks.append({'candidate': candidate, 'two_updates_validated': True,
                       'completed_resume_did_not_retrain': True})
    write_json(args.stage / 'preflight.json', {
        'status': 'passed', 'plan_sha256': digest(plan_path),
        'source_manifest_sha256': digest(args.stage / 'source_manifest.json'),
        'dataset_sha256': plan['pkl_sha256'], 'synthetic_only': True,
        'checks': checks, 'torch': torch.__version__, 'monai': adapter.base.monai.__version__})


if __name__ == '__main__':
    main()
