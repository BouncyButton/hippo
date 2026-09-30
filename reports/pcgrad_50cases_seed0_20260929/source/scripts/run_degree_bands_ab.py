#!/usr/bin/env python3
"""One allocation: calibrate and train degree-band A/B; reuse historical controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GIB = 1024 ** 3


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_payload():
    payload = json.loads((REPO / 'PAYLOAD.json').read_text())
    for name, expected in payload['files'].items():
        if digest(REPO / name) != expected:
            raise RuntimeError(f'Frozen source changed: {name}')


def quota_free_bytes(output):
    for line in output.splitlines():
        if re.search(r'\buser\b', line):
            match = re.search(r'([\d.]+)([KMGT]?i?B)/([\d.]+)([KMGT]?i?B)', line)
            if match:
                units = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': GIB, 'TiB': 1024**4}
                used, unit, limit, limit_unit = match.groups()
                return int(float(limit) * units[limit_unit] - float(used) * units[unit])
    raise RuntimeError('Could not parse finite BeeGFS user quota; refusing to guess free space.')


def check_quota(root, tag, minimum_gib):
    process = subprocess.run(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--gids', 'current'],
                             text=True, capture_output=True, check=True)
    free = quota_free_bytes(process.stdout)
    record = {'output': process.stdout, 'free_bytes': free, 'minimum_free_gib': minimum_gib,
              'passed': free >= minimum_gib * GIB}
    save(root / f'quota_{tag}.json', record)
    if not record['passed']:
        raise RuntimeError(f'BeeGFS quota reserve failed: {free / GIB:.2f} GiB free, {minimum_gib:.2f} required.')
    return record


def load_references(args):
    references = {}
    identity = None
    for name, path in [('dice', args.reference_dice), ('bands', args.reference_bands)]:
        config = json.loads((path / 'config.json').read_text())
        run = config['run']
        completion = json.loads((path / 'completion_manifest.json').read_text())
        if completion.get('status') != 'complete' or completion['run'] != run:
            raise ValueError(f'Historical control is incomplete: {path}')
        expected_set = 'none' if name == 'dice' else 'bands'
        if run['constraint_set'] != expected_set or run['dataset'] != 'MSD' or run['fold'] != 0 or run['seed'] != 0:
            raise ValueError('Expected matched full-data fold-0/seed-0 Dice and bands controls.')
        keys = ('dataset', 'fold', 'seed', 'batch_size', 'spatial_size', 'resize', 'optimizer_mode',
                'learning_rate', 'weight_decay', 'step_size', 'adamw_gamma', 'epochs',
                'training_augmentation', 'early_stopping_patience', 'early_stopping_min_epochs',
                'early_stopping_min_delta', 'amp', 'drop_rate', 'activation_checkpointing', 'plain_tensors')
        settings = {key: run[key] for key in keys}
        if identity is not None and identity != settings:
            raise ValueError('Historical control settings differ.')
        identity = settings
        for field, hash_field in [('pkl', 'pkl_sha256'), ('splits_json', 'splits_json_sha256')]:
            if digest(config[field]) != config[hash_field]:
                raise ValueError(f'Historical input changed: {field}')
        if run['early_stopping_patience'] <= 0:
            raise ValueError('This experiment requires early stopping.')
        checkpoints = [path / 'checkpoint_best.pt', path / 'MSD_fold0/model.pt']
        if not all(p.is_file() for p in checkpoints):
            raise ValueError('Retained historical best checkpoint/export is missing.')
        exported = 'MSD_fold0/model.pt'
        if digest(path / exported) != completion['artifacts'][exported]:
            raise ValueError('Historical selected-model hash changed.')
        references[name] = config
    source = json.loads((args.calibration_checkpoint.parent / 'config.json').read_text())
    if source['run']['epochs'] != 5 or source['run']['constraint_set'] != 'none':
        raise ValueError('An existing dedicated epoch-five calibration checkpoint is required.')
    if digest(REPO / 'baselines/swin_unetr/swin_unetr.py') != source['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py']:
        raise ValueError('The frozen model/data source differs from the calibration source.')
    bindings = {}
    for name, directory in [('dice', args.reference_dice), ('bands', args.reference_bands)]:
        bindings[name] = {'directory': str(directory),
                          'config_sha256': digest(directory / 'config.json'),
                          'best_checkpoint_sha256': digest(directory / 'checkpoint_best.pt'),
                          'selected_model_sha256': digest(directory / 'MSD_fold0/model.pt')}
    bindings['calibration'] = {'checkpoint': str(args.calibration_checkpoint),
                               'sha256': digest(args.calibration_checkpoint)}
    save(args.root / 'reference_bindings.json', bindings)
    return references


def training_arguments(reference, output, report, normalization, alpha):
    run = reference['run']
    flags = {'--dataset': 'dataset', '--fold': 'fold', '--seed': 'seed', '--epochs': 'epochs',
             '--batch-size': 'batch_size', '--optim-mode': 'optimizer_mode', '--learning-rate': 'learning_rate',
             '--weight-decay': 'weight_decay', '--step-size': 'step_size', '--adamw-gamma': 'adamw_gamma',
             '--training-augmentation': 'training_augmentation', '--drop-rate': 'drop_rate',
             '--constraint-warmup-epochs': 'constraint_warmup_epochs',
             '--early-stopping-patience': 'early_stopping_patience',
             '--early-stopping-min-epochs': 'early_stopping_min_epochs',
             '--early-stopping-min-delta': 'early_stopping_min_delta'}
    result = ['--pkl', reference['pkl'], '--splits-json', reference['splits_json'],
              '--output-dir', str(output), '--num-workers', '0', '--device', 'cuda',
              '--supervised-loss', 'dice', '--constraint-set', 'bands', '--constraint-eval-every', '0',
              '--band-steps', '2', '--bands-focal-gamma', '0', '--bands-loss-type', 'focal_bce',
              '--bands-degree-alpha', str(alpha), '--bands-degree-normalization', normalization,
              '--bands-calibration-json', str(report),
              '--bands-weight', str(json.loads(report.read_text())['recommended_bands_weight']),
              '--calibration-diagnostics', '--spatial-size', *map(str, run['spatial_size'])]
    for flag, key in flags.items():
        result.extend((flag, str(run[key])))
    result.append('--amp' if run['amp'] else '--no-amp')
    result.append('--activation-checkpointing' if run['activation_checkpointing'] else '--no-activation-checkpointing')
    if run['plain_tensors']:
        result.append('--plain-tensors')
    if run['resize']:
        result.append('--resize')
    return result


def execute(root, tag, relative, arguments):
    command = [sys.executable, str(REPO / relative), *arguments]
    save(root / f'{tag}_command.json', command)
    print(json.dumps({'phase': tag, 'command': command}), flush=True)
    with (root / f'{tag}.log').open('x') as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)


def audit_selected(root, runs):
    """Re-score existing and new best checkpoints; never save image arrays."""
    import numpy as np
    import torch
    from scipy import ndimage
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.audit_followup import segmentation_metrics

    device = torch.device('cuda')
    summaries = {}
    records = {}
    identity = None
    for name, directory in runs.items():
        config = json.loads((directory / 'config.json').read_text())
        run = config['run']
        current = (config['pkl_sha256'], config['splits_json_sha256'], run['fold'], run['spatial_size'], run['resize'])
        if identity is not None and current != identity:
            raise ValueError('Audit cohorts differ.')
        identity = current
        data_args = SimpleNamespace(pkl=Path(config['pkl']), splits_json=Path(config['splits_json']),
                                    dataset=run['dataset'], fold=run['fold'], spatial_size=run['spatial_size'],
                                    resize=run['resize'], num_classes=3, num_workers=0, batch_size=1, plain_tensors=True)
        _, loader, _, train_n, val_n = trainer.build_data(data_args, torch.Generator().manual_seed(0))
        if (train_n, val_n) != (208, 52):
            raise ValueError('Expected the fixed full-data 208/52 split.')
        model = trainer.build_swinunetr(tuple(run['spatial_size']), 3, device,
                                       drop_rate=run['drop_rate'], activation_checkpointing=False)
        path = directory / 'checkpoint_best.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if json.loads(json.dumps(checkpoint['run'])) != run:
            raise ValueError('Audit checkpoint/config mismatch.')
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        model.eval()
        rows = []
        cross = ndimage.generate_binary_structure(3, 1)
        kernel = cross.astype(np.int16)
        kernel[1, 1, 1] = 0
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                with torch.autocast('cuda', enabled=run['amp']):
                    logits = model(images)
                row = segmentation_metrics(logits[0].float(), labels[0, 0])
                row['case_name'] = str(batch['case_name'][0])
                truth = labels[0, 0].cpu().numpy()
                pred = logits[0].argmax(0).cpu().numpy()
                fg, decoded = truth > 0, pred > 0
                degree = ndimage.convolve(fg.astype(np.int16), kernel, mode='constant', cval=0)
                row['degree'] = {}
                for region, mask in [('whole', fg), ('anterior', truth == 1), ('posterior', truth == 2)]:
                    row['degree'][region] = {}
                    for label, lo, hi in [('0_1', 0, 1), ('2_3', 2, 3), ('4_5', 4, 5), ('6', 6, 6)]:
                        subset = mask & (degree >= lo) & (degree <= hi)
                        row['degree'][region][label] = {'count': int(subset.sum()), 'missed': int((subset & ~decoded).sum())}
                true_surface = fg & ~ndimage.binary_erosion(fg, structure=cross)
                pred_surface = decoded & ~ndimage.binary_erosion(decoded, structure=cross)
                if true_surface.any() and pred_surface.any():
                    distances = np.concatenate((ndimage.distance_transform_edt(~true_surface)[pred_surface],
                                                ndimage.distance_transform_edt(~pred_surface)[true_surface]))
                    row['assd_mm'] = float(distances.mean())
                    row['hd95_mm'] = float(np.quantile(distances, .95))
                else:
                    row['assd_mm'] = row['hd95_mm'] = None
                rows.append(row)
        summary = {'selected_epoch': epoch, 'case_count': len(rows), 'checkpoint_sha256': digest(path),
                   'degree_definition': 'GT whole-foreground six-face degree; regional rows subset this same degree',
                   'surface_definition': 'pooled bidirectional six-face surface-voxel distances, 1-mm isotropic',
                   'macro_dice': float(np.mean([r['macro_dice'] for r in rows])),
                   'union_dice': float(np.mean([r['union_dice'] for r in rows])),
                   'anterior_dice': float(np.mean([r['classes']['1']['dice'] for r in rows])),
                   'posterior_dice': float(np.mean([r['classes']['2']['dice'] for r in rows])),
                   'total_fp': sum(r['fp'] for r in rows), 'total_fn': sum(r['fn'] for r in rows),
                   'total_ap_swaps': sum(r['swaps'] for r in rows), 'degree': {}}
        for metric in ['assd_mm', 'hd95_mm']:
            valid = [r[metric] for r in rows if r[metric] is not None]
            summary[metric] = float(np.mean(valid)) if valid else None
            summary[metric + '_valid_cases'] = len(valid)
        for region in ['whole', 'anterior', 'posterior']:
            summary['degree'][region] = {}
            for label in ['0_1', '2_3', '4_5', '6']:
                count = sum(r['degree'][region][label]['count'] for r in rows)
                missed = sum(r['degree'][region][label]['missed'] for r in rows)
                summary['degree'][region][label] = {'count': count, 'missed': missed,
                                                   'miss_rate': missed / count if count else None}
        records[name] = rows
        summaries[name] = summary
        save(root / 'audit' / f'{name}_cases.json', rows)
        save(root / 'audit' / 'summary.json', summaries)
        print(json.dumps({'audit_complete': name, 'macro_dice': summary['macro_dice']}), flush=True)
        del model, loader
        torch.cuda.empty_cache()
    paired = {}
    for first, second in [('A', 'B'), ('A', 'bands'), ('B', 'bands'), ('A', 'dice'), ('B', 'dice')]:
        a = {r['case_name']: r for r in records[first]}
        b = {r['case_name']: r for r in records[second]}
        if a.keys() != b.keys():
            raise ValueError('Paired audit case identities differ.')
        deltas = np.array([a[k]['macro_dice'] - b[k]['macro_dice'] for k in sorted(a)])
        bootstrap = np.random.default_rng(20260924).choice(deltas, size=(10000, len(deltas))).mean(1)
        paired[f'{first}_minus_{second}'] = {'mean_macro_dice_delta': float(deltas.mean()),
                                            'paired_case_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
                                            'better_cases': int((deltas > 0).sum()), 'worse_cases': int((deltas < 0).sum())}
    save(root / 'audit' / 'comparison.json', {'paired': paired, 'higher_validation_dice_arm': max(('A', 'B'), key=lambda k: summaries[k]['macro_dice']),
                                             'scope': 'One seed, one repeatedly used development fold; exploratory comparison.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-dice', type=Path, required=True)
    parser.add_argument('--reference-bands', type=Path, required=True)
    parser.add_argument('--calibration-checkpoint', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--allow-calibration-mig-profile-change', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    verify_payload()
    reference = load_references(args)['bands']
    # A conservative allowance for two best/latest checkpoint pairs, final model
    # exports, atomic replacement/snapshot copies, reports and storage overhead.
    check_quota(args.root, 'preflight' if args.preflight_only else 'job_start', 3.5)
    if args.preflight_only:
        print(json.dumps({'status': 'preflight_passed', 'training_runs': 2}), flush=True)
        return
    try:
        import torch
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        trainer.seed_everything(0)
        if not torch.cuda.is_available():
            raise RuntimeError('A/B training requires the requested CUDA allocation.')
        calibration_config = json.loads((args.calibration_checkpoint.parent / 'config.json').read_text())
        if trainer.collect_runtime_provenance() != calibration_config['runtime_provenance']:
            raise ValueError('Calibration checkpoint runtime changed.')
        current_execution = trainer.collect_execution_provenance(torch.device('cuda'))
        trainer.validate_calibration_execution(
            calibration_config['execution_provenance'], current_execution,
            allow_mig_profile_change=args.allow_calibration_mig_profile_change)
        save(args.root / 'calibration_device_transfer.json', {
            'checkpoint_execution': calibration_config['execution_provenance'],
            'current_execution': current_execution,
            'allow_mig_profile_change': args.allow_calibration_mig_profile_change})
        for arm, normalization in [('A', 'inner'), ('B', 'surface')]:
            report = args.root / f'calibration_{arm}.json'
            calibration = ['--pkl', reference['pkl'], '--splits-json', reference['splits_json'],
                           '--checkpoint', str(args.calibration_checkpoint), '--output', str(report),
                           '--dataset', 'MSD', '--fold', '0', '--seed', '0', '--max-cases', '32',
                           '--spatial-size', '64', '64', '64', '--training-augmentation', 'mild_v1',
                           '--bands-degree-alpha', str(args.alpha), '--bands-degree-normalization', normalization,
                           '--device', 'cuda', '--amp']
            if args.allow_calibration_mig_profile_change:
                calibration.append('--allow-calibration-mig-profile-change')
            execute(args.root, f'calibrate_{arm}', 'thesis/new_constraints/bands/calibrate_weight.py', calibration)
        runs = {'dice': args.reference_dice, 'bands': args.reference_bands}
        for index, (arm, normalization) in enumerate([('A', 'inner'), ('B', 'surface')]):
            check_quota(args.root, f'before_{arm}', 3.5 if index == 0 else 2.25)
            output = args.root / 'runs' / arm
            arguments = training_arguments(reference, output, args.root / f'calibration_{arm}.json', normalization, args.alpha)
            if args.allow_calibration_mig_profile_change:
                arguments.append('--allow-calibration-mig-profile-change')
            execute(args.root, f'train_{arm}', 'thesis/new_constraints/train_swinunetr_constraints.py', arguments)
            runs[arm] = output
        audit_selected(args.root, runs)
        verify_payload()
        save(args.root / 'completion.json', {'status': 'complete', 'job_id': os.getenv('SLURM_JOB_ID'),
                                             'trained_arms': ['A', 'B'], 'reused_controls': ['dice', 'bands'],
                                             'comparison': str(args.root / 'audit/comparison.json')})
    except BaseException:
        import traceback
        save(args.root / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
