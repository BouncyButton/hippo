"""Verify the frozen comparison recipes and actual cluster model initialization."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

import monai
import torch

import train_compute_matched_baselines as base
from baselines.residual_unet import build_unet48
from compute_constraint_objectives import scheduled_costs
from train_budget_bands import validation_updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = args.source
    directory = source / 'experiments/budget_bands_94603_20261005'
    campaign = json.loads((directory / 'campaign.json').read_text())
    if (campaign['budget_flops'] != 94603780000000 or campaign['model_order'] != ['unet', 'swinunetr', 'nnunet']
            or campaign['fold'] != 0 or campaign['seed'] != 0 or monai.__version__ != '1.6.0'):
        raise ValueError('Campaign identity, cap, or MONAI version differs')
    version = importlib.metadata.version('dynamic-network-architectures')
    if tuple(int(v) for v in version.split('.')[:3]) < (0, 4, 1) or not version.startswith('0.4.'):
        raise ValueError('Unsupported architecture package version')
    if base.digest(args.pkl) != campaign['pkl_sha256']:
        raise ValueError('Dataset hash differs')
    # These shared modules must remain exactly those used by the paired baselines.
    for name in ['scripts/train_compute_matched_baselines.py', 'scripts/compute_constraint_objectives.py',
                 'baselines/residual_unet.py', 'thesis/new_constraints/training_augmentation.py']:
        if base.digest(source / name) != base.digest(args.baseline_root / 'source' / name):
            raise ValueError('Baseline shared source changed: ' + name)
    rows = []
    for model_name in campaign['model_order']:
        protocol_dir = directory / model_name
        protocol = json.loads((protocol_dir / 'protocol.json').read_text())
        old_dir = args.baseline_root / 'source/experiments/budget_baselines_94603_20261005' / model_name
        original = json.loads((old_dir / 'protocol.json').read_text())
        if (base.digest(protocol_dir / 'protocol.json') != campaign['protocol_sha256'][model_name]
                or protocol['baseline_protocol_sha256'] != base.digest(old_dir / 'protocol.json')
                or protocol['campaign_adapter_sha256'] != base.digest(source / 'scripts/train_budget_bands.py')
                or protocol['budget_flops'] != campaign['budget_flops']
                or protocol['folds'] != [0] or protocol['seeds'] != [0]
                or list(protocol['models'][model_name]['objectives']) != ['dice_bands']):
            raise ValueError('Model protocol differs')
        for name in ['splits_10patients.json', 'nnunet_architecture.json']:
            if base.digest(protocol_dir / name) != base.digest(old_dir / name):
                raise ValueError('Split or architecture changed')
        for key in ['initial_lrs', 'adamw_weight_decay', 'sgd_weight_decay', 'augmentation',
                    'precision', 'nnunet_sgd_momentum', 'nnunet_warmup_fraction', 'learning_rate_policy']:
            if protocol.get(key) != original.get(key):
                raise ValueError('Baseline optimization recipe changed: ' + key)
        evidence = protocol['profile_evidence']
        profile_path = source / evidence['profile_file']
        if base.digest(profile_path) != evidence['profile_sha256']:
            raise ValueError('Profile evidence changed')
        profile = json.loads(profile_path.read_text())
        for name, sha in profile['source_sha256'].items():
            if base.digest(source / name) != sha:
                raise ValueError('Profile source changed: ' + name)
        costs = profile['costs']
        schedule = protocol['models'][model_name]['objectives']['dice_bands']
        expected = scheduled_costs(campaign['budget_flops'], costs['baseline'], costs['bands'], costs['calibration_total'], 50)
        if any(schedule[key] != value for key, value in expected.items()):
            raise ValueError('Schedule arithmetic differs')
        original_entry = original['models'][model_name]
        baseline_cost = original_entry['objectives']['dice']['flops_per_update'] if model_name == 'unet' else original_entry['flops_per_update']
        if costs['baseline'] != baseline_cost or schedule['flops_per_update'] != costs['bands'] or schedule['baseline_flops_per_update'] != baseline_cost:
            raise ValueError('Update-cost recipe differs')
        if protocol['constraint_warmup_updates'] != 50:
            raise ValueError('Constraint ramp differs')
        if model_name == 'unet':
            if (protocol['unet_factory_sha256'] != base.digest(source / 'baselines/residual_unet.py')
                    or protocol['unet_output_head'] != 'linear'
                    or protocol['unet_lr_policy'] != 'absolute_warmup_constant'
                    or protocol['unet_warmup_updates'] != 100):
                raise ValueError('Corrected U-Net recipe differs')
        architecture = json.loads((protocol_dir / 'nnunet_architecture.json').read_text())
        base.seed_everything(0)
        model = build_unet48() if model_name == 'unet' else base.build_model(model_name, architecture)
        criterion, optimizer = base.build_loss_and_optimizer(model_name, model, protocol)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != profile['parameters'] or parameters != protocol['models'][model_name]['parameters']:
            raise ValueError('Parameter count differs')
        if model_name == 'nnunet' and optimizer.param_groups[0]['momentum'] != .95:
            raise ValueError('nnU-Net momentum differs')
        rows.append({'model': model_name, 'parameters': parameters, 'loss': type(criterion).__name__,
                     'optimizer': type(optimizer).__name__, 'max_updates': schedule['max_steps'],
                     'calibration_flops': schedule['calibration_flops'],
                     'consumed_flops': schedule['consumed_flops'], 'unused_flops': schedule['unused_flops'],
                     'validation_updates': validation_updates(protocol, model_name)})
        del model, criterion, optimizer
        gc.collect()
    result = {'status': 'passed', 'budget_gflops_per_run': campaign['budget_gflops'], 'baseline_job_id': 684354,
              'torch': torch.__version__, 'monai': monai.__version__, 'dynamic_network_architectures': version,
              'campaign_sha256': base.digest(directory / 'campaign.json'), 'models': rows}
    base.atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
