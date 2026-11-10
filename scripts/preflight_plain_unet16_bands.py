"""Check the paired U-Net recipe, bands gradients, calibration and checkpoint resume."""
import argparse
import json
from pathlib import Path
import sys

import train_budget_bands as adapter
from baselines.simple_segmentation import build_candidate
from compute_constraint_objectives import scheduled_costs, training_loss
from run_budget_baselines import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    source = args.stage / 'source'
    manifest = args.stage / 'source_manifest.json'
    for name, sha in json.loads(manifest.read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    directory = source / 'experiments/plain_unet16_bands_20261005/dice_bands'
    protocol = json.loads((directory / 'protocol.json').read_text())
    baseline_path = source / 'experiments/plain_unet16_lr3e4_20261005/dice/protocol.json'
    baseline = json.loads(baseline_path.read_text())
    for key in ['candidate', 'candidate_name', 'budget_flops', 'initial_lrs',
                'augmentation', 'precision', 'adamw_weight_decay', 'splits_sha256']:
        if protocol[key] != baseline[key]:
            raise ValueError('Paired baseline differs: ' + key)
    if (protocol['baseline_protocol_sha256'] != digest(baseline_path)
            or protocol['baseline_job_id'] != 684625 or protocol['folds'] != [0]
            or protocol['seeds'] != [0] or protocol['budget_flops'] != 62166546655000):
        raise ValueError('Unexpected bands experiment')
    evidence = protocol['profile_evidence']
    profile_path = source / evidence['profile_file']
    if digest(profile_path) != evidence['profile_sha256']:
        raise ValueError('Profile differs')
    profile = json.loads(profile_path.read_text())
    for name, sha in profile['source_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError('Profile source changed: ' + name)
    costs = profile['costs']
    schedule = protocol['models']['unet']['objectives']['dice_bands']
    expected = scheduled_costs(protocol['budget_flops'], costs['baseline'], costs['bands'],
                               costs['calibration_total'], 50)
    if (any(schedule[k] != v for k, v in expected.items()) or schedule['max_steps'] != 1091
            or schedule['flops_per_update'] != costs['bands']
            or schedule['baseline_flops_per_update'] != costs['baseline']):
        raise ValueError('Compute schedule differs')
    data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    if digest(data) != protocol['pkl_sha256']:
        raise ValueError('Dataset changed')
    trainer = adapter.trainer
    torch = trainer.torch
    torch.set_num_threads(4)
    torch.manual_seed(913)
    label = torch.zeros((1, 64, 64, 64), dtype=torch.long)
    label[:, 20:32, 20:44, 20:44] = 1
    label[:, 32:44, 20:44, 20:44] = 2
    image = label.float() + .1 * torch.randn(label.shape)
    split = json.loads((directory / 'splits_10patients.json').read_text())[0]
    model = build_candidate('basic_unet16')
    cases = [{'image': image.clone(), 'label': label.clone(), 'case_name': n} for n in split['train']]
    report = trainer.calibrate(model, cases, 'dice_bands', torch.device('cpu'), False)
    loss = training_loss(model(image[None]), label[None], 'dice_bands', report['weights'], 1.)
    loss.backward()
    if (not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all()
                                           for p in model.parameters())):
        raise ValueError('Nonfinite bands gradient')
    trainer.base._load_pkl_dataframe = lambda path: None
    trainer.base.PreparedPatients = lambda frame, names: [
        {'image': image.clone(), 'label': label.clone(), 'case_name': n}
        for n in (names if len(names) == 10 else names[:2])]
    output = args.stage / 'synthetic_smoke_full'
    sys.argv = ['train_budget_bands.py', '--model', 'unet', '--objective', 'dice_bands',
                '--fold', '0', '--seed', '0', '--protocol-dir', str(directory),
                '--output', str(output), '--pkl', str(data), '--device', 'cpu',
                '--threads', '4', '--smoke-steps', '52', '--resume']
    adapter.main()
    metrics, ledger = digest(output / 'metrics.csv'), digest(output / 'budget_ledger.json')
    done = json.loads((output / 'completion.json').read_text())
    adapter.main()
    if (done['status'] != 'smoke_complete' or done['updates'] != 52
            or done['calibration_attempts'] != 1 or done['discarded_attempts'] != 0
            or digest(output / 'metrics.csv') != metrics or digest(output / 'budget_ledger.json') != ledger):
        raise ValueError('Smoke/resume check failed')
    write_json(args.stage / 'preflight.json', {
        'status': 'passed', 'protocol_sha256': digest(directory / 'protocol.json'),
        'source_manifest_sha256': digest(manifest), 'synthetic_only': True,
        'bands_backward_finite': True, 'ten_case_calibration_finite': True,
        'fifty_baseline_plus_two_bands_updates': True,
        'completed_resume_idempotent': True,
        'torch': torch.__version__, 'monai': trainer.monai.__version__})


if __name__ == '__main__':
    main()
