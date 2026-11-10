"""Verify the nnU-Net pair and exercise both paths with a synthetic short prefix."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import train_budget_baseline as baseline_adapter
import train_budget_bands as bands_adapter
from compute_constraint_objectives import scheduled_costs
from run_budget_baselines import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    source = args.stage / 'source'
    manifest_path = args.stage / 'source_manifest.json'
    for name, sha in json.loads(manifest_path.read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    directory = source / 'experiments/nnunet_pair_62166_20261005'
    plan = json.loads((directory / 'plan.json').read_text())
    protocols = {arm: json.loads((directory / arm / 'protocol.json').read_text())
                 for arm in ['baseline', 'bands']}
    for arm, protocol in protocols.items():
        if (digest(directory / arm / 'protocol.json') != plan['protocol_sha256'][arm]
                or protocol['budget_flops'] != 62166546655000
                or protocol['folds'] != [0] or protocol['seeds'] != [0]):
            raise ValueError('Protocol identity differs')
    for key in ['splits_sha256', 'nnunet_architecture_sha256', 'pkl_sha256', 'augmentation',
                'precision', 'sgd_weight_decay', 'nnunet_sgd_momentum',
                'nnunet_warmup_fraction', 'learning_rate_policy', 'validation_target_flops']:
        if protocols['baseline'][key] != protocols['bands'][key]:
            raise ValueError('Unpaired recipe: ' + key)
    evidence = protocols['bands']['profile_evidence']
    profile_path = source / evidence['profile_file']
    if digest(profile_path) != evidence['profile_sha256']:
        raise ValueError('Profile changed')
    profile = json.loads(profile_path.read_text())
    for name, sha in profile['source_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError('Profile source changed: ' + name)
    costs = profile['costs']
    schedule = protocols['bands']['models']['nnunet']['objectives']['dice_bands']
    expected = scheduled_costs(62166546655000, costs['baseline'], costs['bands'], costs['calibration_total'], 50)
    if (any(schedule[k] != v for k, v in expected.items()) or schedule['max_steps'] != 171
            or 62166546655000 // costs['baseline'] != 174
            or protocols['baseline']['initial_lrs']['nnunet'] != .01
            or protocols['bands']['initial_lrs']['nnunet'] != .01
            or protocols['bands']['nnunet_sgd_momentum'] != .95):
        raise ValueError('Budget or optimizer differs')
    data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    if digest(data) != plan['pkl_sha256']:
        raise ValueError('Dataset changed')
    torch = baseline_adapter.base.torch
    torch.set_num_threads(4)
    torch.manual_seed(913)
    labels = torch.zeros((1, 64, 64, 64), dtype=torch.long)
    labels[:, 20:32, 20:44, 20:44] = 1
    labels[:, 32:44, 20:44, 20:44] = 2
    image = labels.float() + .1 * torch.randn(labels.shape)
    baseline_adapter.base._load_pkl_dataframe = lambda path: None
    baseline_adapter.base.PreparedPatients = lambda frame, names: [
        {'image': image.clone(), 'label': labels.clone(), 'case_name': n}
        for n in (names if len(names) == 10 else names[:2])]
    checks = []
    for arm, adapter in [('baseline', baseline_adapter), ('bands', bands_adapter)]:
        protocol_dir = directory / arm
        output = args.stage / 'synthetic_smoke' / arm
        if arm == 'bands':
            # Exercise calibration after one synthetic update; production remains at50.
            protocol_dir = args.stage / 'synthetic_protocol'
            protocol_dir.mkdir(exist_ok=True)
            for filename in ['splits_10patients.json', 'nnunet_architecture.json', 'split_provenance.json']:
                shutil.copyfile(directory / arm / filename, protocol_dir / filename)
            smoke = copy.deepcopy(protocols['bands'])
            synthetic_schedule = scheduled_costs(62166546655000, costs['baseline'], costs['bands'], costs['calibration_total'], 1)
            synthetic_schedule.update(flops_per_update=costs['bands'], baseline_flops_per_update=costs['baseline'])
            smoke['models']['nnunet']['objectives']['dice_bands'] = synthetic_schedule
            write_json(protocol_dir / 'protocol.json', smoke)
        sys.argv = ['train.py', '--model', 'nnunet', '--fold', '0', '--seed', '0',
                    '--protocol-dir', str(protocol_dir), '--output', str(output), '--pkl', str(data),
                    '--device', 'cpu', '--threads', '4', '--smoke-steps', '2', '--resume']
        if arm == 'bands':
            sys.argv.extend(['--objective', 'dice_bands'])
        adapter.main()
        metrics, ledger = digest(output / 'metrics.csv'), digest(output / 'budget_ledger.json')
        done = json.loads((output / 'completion.json').read_text())
        adapter.main()
        if (done['status'] != 'smoke_complete' or done['updates'] != 2
                or done['discarded_attempts'] != 0 or metrics != digest(output / 'metrics.csv')
                or ledger != digest(output / 'budget_ledger.json')
                or arm == 'bands' and done['calibration_attempts'] != 1):
            raise ValueError('Synthetic training/resume failed')
        checks.append({'arm': arm, 'two_updates_and_resume_passed': True})
    write_json(args.stage / 'preflight.json', {
        'status': 'passed', 'plan_sha256': digest(directory / 'plan.json'),
        'source_manifest_sha256': digest(manifest_path), 'checks': checks, 'synthetic_only': True,
        'synthetic_calibration_prefix': 1, 'production_calibration_prefix': 50,
        'torch': torch.__version__, 'monai': baseline_adapter.base.monai.__version__})


if __name__ == '__main__':
    main()
