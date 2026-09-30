"""CUDA-only, training-only epoch-5 calibration of common-support teacher KL.

No optimizer or validation evaluation. Frozen FP32 logit gradients are a scale
calibration, not proof of a beneficial network-parameter update.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..bands.calibrate_weight import _build_training_loader
from ..equivariance import translate_3d
from ..source_bootstrap import source_digest
from ..supervised import build_supervised_loss
from ..train_swinunetr_constraints import build_swinunetr, file_sha256, collect_runtime_provenance
from .translation_teacher import DEFAULT_TEACHER_SHIFTS, TranslationTeacherKLLoss


def choose_weight(ratios):
    """One prespecified scale rule, with no quality selection or validation input."""
    values = np.asarray(ratios, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Calibration needs finite positive gradient ratios.')
    return float(min(0.10 / np.median(values), 0.50 / np.quantile(values, 0.95)))


def gradient_comparison(auxiliary, supervised):
    a, s = auxiliary.float().flatten(), supervised.float().flatten()
    an, sn = a.norm(), s.norm()
    if sn <= 0 or not torch.isfinite(a).all() or not torch.isfinite(s).all():
        raise ValueError('Invalid calibration gradients.')
    return {'ratio': float(an / sn),
            'cosine': float(torch.dot(a, s) / (an * sn).clamp_min(1e-30))}


def regional_pressure(logits, labels, auxiliary, supervised):
    """Partition gradient energy into FP, FN, A/P swaps and correct voxels."""
    prediction, truth = logits.detach().argmax(1), labels[:, 0]
    masks = {'false_positive': (truth == 0) & (prediction != 0),
             'false_negative': (truth != 0) & (prediction == 0),
             'ap_swap': (truth != 0) & (prediction != 0) & (prediction != truth),
             'correct': prediction == truth}
    total = auxiliary.float().square().sum().clamp_min(1e-30)
    result = {}
    for name, mask in masks.items():
        a = auxiliary.float().movedim(1, -1)[mask].flatten()
        s = supervised.float().movedim(1, -1)[mask].flatten()
        denominator = a.norm() * s.norm()
        result[name] = {'voxels': int(mask.sum()),
                        'gradient_energy_fraction': float(a.square().sum() / total),
                        'supervised_cosine': float(torch.dot(a, s) / denominator)
                        if denominator > 0 else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback.')
    if not 1 <= args.max_cases <= 32 or args.output.exists():
        raise ValueError('Use 1..32 training cases and a new output path.')
    torch.set_num_threads(4)
    source = source_digest()
    root = args.run_dir.resolve()
    checkpoint_path = root / 'checkpoint_latest.pt'
    checkpoint_hash = file_sha256(checkpoint_path)
    config = json.loads((root / 'config.json').read_text())
    completion = json.loads((root / 'completion_manifest.json').read_text())
    run = config['run']
    required = {'constraint_set': 'none', 'dataset': 'MSD', 'fold': 0, 'seed': 0,
                'epochs': 5, 'batch_size': 1, 'amp': True, 'resize': False,
                'learning_rate': 0.0001, 'weight_decay': 0.00001,
                'optimizer_mode': 'adamw_0.01', 'step_size': 20, 'adamw_gamma': 0.5}
    if any(run.get(key) != value for key, value in required.items()):
        raise ValueError('Require the Family-B five-epoch Dice-only seed-0 control.')
    if (run.get('supervised_loss', 'dice') != 'dice' or run.get('ce_weight', 0) != 0
            or run.get('translation_augmentation', False)
            or run.get('initial_checkpoint') is not None or run['spatial_size'] != [64, 64, 64]):
        raise ValueError('Calibration source must be an unaugmented Dice-only run from scratch.')
    if completion['status'] != 'complete' or completion['run'] != run:
        raise ValueError('Source completion mismatch.')
    final_path = root / 'MSD_fold0/model.pt'
    if file_sha256(final_path) != completion['artifacts']['MSD_fold0/model.pt']:
        raise ValueError('Final model hash differs from completion manifest.')
    for key in ('pkl', 'splits_json'):
        if file_sha256(Path(config[key])) != config[key + '_sha256']:
            raise ValueError(f'{key} digest changed.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != 5 or json.loads(json.dumps(checkpoint['run'])) != run:
        raise ValueError('Checkpoint run/epoch mismatch.')
    # The legacy completion manifest hashes the final exported model, not the
    # latest optimizer checkpoint. Bind the latter's weights to that artifact.
    final_weights = torch.load(final_path, map_location='cpu', weights_only=True)
    if final_weights.keys() != checkpoint['model'].keys() or any(
        not torch.equal(value, checkpoint['model'][name]) for name, value in final_weights.items()
    ):
        raise ValueError('Latest checkpoint weights differ from the verified final model.')
    del final_weights
    if collect_runtime_provenance() != config['runtime_provenance']:
        raise ValueError('Calibration runtime differs from the source control.')
    loader_args = argparse.Namespace(pkl=Path(config['pkl']), splits_json=Path(config['splits_json']),
        dataset='MSD', spatial_size=(64, 64, 64), resize=False, fold=0,
        seed=20260905, max_cases=args.max_cases)
    loader, selected, _ = _build_training_loader(loader_args)
    model = build_swinunetr((64, 64, 64), 3, torch.device('cuda'))
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.requires_grad_(False).eval()
    teacher_loss = TranslationTeacherKLLoss(num_views=2, support='common')
    supervised_loss = build_supervised_loss('dice')
    generator = torch.Generator().manual_seed(20260905)
    rows, ratios = [], []
    for index, (selected_case, batch) in enumerate(zip(selected, loader)):
        images, labels = batch['image'].cuda(), batch['label'].cuda()
        with torch.no_grad(), torch.autocast('cuda', enabled=True):
            original = model(images)
            views = [model(translate_3d(images, shift)) for shift in DEFAULT_TEACHER_SHIFTS]
        logits = original.detach().float().requires_grad_()
        supervised_gradient, = torch.autograd.grad(supervised_loss(logits, labels), logits)
        teacher = teacher_loss.build_from_cached(views, DEFAULT_TEACHER_SHIFTS)
        full_gradient, = torch.autograd.grad(teacher_loss.loss_from_teacher(logits, teacher).loss, logits)
        draws = []
        for _ in range(8):
            indices = torch.randperm(12, generator=generator)[:2].tolist()
            target = teacher_loss.build_from_cached([views[i] for i in indices],
                                                   [DEFAULT_TEACHER_SHIFTS[i] for i in indices])
            gradient, = torch.autograd.grad(teacher_loss.loss_from_teacher(logits, target).loss, logits)
            comparison = gradient_comparison(gradient, supervised_gradient)
            comparison['relative_noise_norm'] = float((gradient - full_gradient).norm()
                                                       / full_gradient.norm().clamp_min(1e-30))
            comparison['view_indices'] = indices
            draws.append(comparison)
            ratios.append(comparison['ratio'])
        rows.append({'case_name': selected_case['case_name'], 'draws': draws,
                     'full_teacher': gradient_comparison(full_gradient, supervised_gradient),
                     'full_teacher_pressure': regional_pressure(logits, labels, full_gradient, supervised_gradient),
                     'valid_fraction': float(teacher.valid_mask.float().mean())})
        print(json.dumps({'phase': 'train_only_calibration', 'case': index + 1,
                          'of': len(selected)}), flush=True)
    if source_digest() != source or file_sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError('Source or checkpoint changed during calibration.')
    weight = choose_weight(ratios)
    result = {'status': 'complete', 'purpose': 'training_only_logit_scale_calibration',
              'source_sha256': source, 'runtime': collect_runtime_provenance(),
              'gpu': torch.cuda.get_device_name(), 'source_run': str(root),
              'source_run_spec': run, 'checkpoint_sha256': checkpoint_hash,
              'cohort': 'fold0_train_only', 'selection_seed': 20260905,
              'teacher': {'views': 2, 'temperature': 1.0, 'support': 'common',
                          'shifts': DEFAULT_TEACHER_SHIFTS},
              'supervised_loss': 'dice', 'translation_augmentation': False,
              'recommended_weight': weight, 'median_weighted_ratio': float(weight * np.median(ratios)),
              'p95_weighted_ratio': float(weight * np.quantile(ratios, .95)),
              'cases': rows, 'limitation': 'Scale only; no training benefit or parameter-gradient claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'recommended_weight': weight, 'report': str(args.output)}))


if __name__ == '__main__':
    main()
