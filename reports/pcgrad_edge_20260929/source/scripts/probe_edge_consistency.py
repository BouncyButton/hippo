#!/usr/bin/env python3
"""Training-only calibration and paired logit-gradient test of edge consistency."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def counts(logits, labels, inner, outer):
    pred = logits.argmax(1)
    truth = labels[:, 0].long()
    fg = truth > 0
    return {'inner_fn': int((inner[:, 0] & (pred == 0)).sum()),
            'outer_fp': int((outer[:, 0] & (pred > 0)).sum()),
            'all_fn': int((fg & (pred == 0)).sum()),
            'all_fp': int((~fg & (pred > 0)).sum()),
            'ap_swaps': int((fg & (pred > 0) & (pred != truth)).sum()),
            'class_errors': int((pred != truth).sum())}


def decide_gate(aggregate, finite, cosine):
    deltas = {}
    for step in ('0.25', '1.0'):
        base, combined = aggregate[step]['base'], aggregate[step]['combined']
        d = {k: combined[k] - base[k] for k in base}
        d['boundary_errors'] = d['inner_fn'] + d['outer_fp']
        deltas[step] = d
    checks = {'finite_nonzero_gradients': bool(finite), 'mean_gradient_cosine_nonnegative': cosine >= 0,
              'no_inner_or_outer_regression_at_both_small_steps': all(d['inner_fn'] <= 0 and d['outer_fp'] <= 0 for d in deltas.values()),
              'strict_boundary_improvement_at_a_small_step': any(d['boundary_errors'] < 0 for d in deltas.values()),
              'no_total_class_error_regression_at_both_small_steps': all(d['class_errors'] <= 0 for d in deltas.values())}
    return {'passed': all(checks.values()), 'checks': checks, 'candidate_minus_base': deltas,
            'scope': 'Training-only logit intervention; neither parameter-update evidence nor validation improvement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-bands', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case-report', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'probe.lock').open('x') as f: f.write(str(os.getenv('SLURM_JOB_ID')))
    try:
        import numpy as np
        import torch
        from monai.losses import DiceLoss
        from torch.utils.data import DataLoader, Subset
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        from thesis.new_constraints.bands import OuterBoundaryBandLoss, build_boundary_bands
        from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
        from thesis.new_constraints.training_augmentation import augment_mild

        for name, sha in json.loads((REPO / 'PAYLOAD.json').read_text())['files'].items():
            assert digest(REPO / name) == sha, name
        trainer.seed_everything(0)
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        config = json.loads((args.reference_bands / 'config.json').read_text())
        run = config['run']
        completed = json.loads((args.reference_bands / 'completion_manifest.json').read_text())
        assert completed['status'] == 'complete' and completed['run'] == run
        assert run['constraint_set'] == 'bands' and run['fold'] == run['seed'] == 0
        assert run['training_augmentation'] == 'mild_v1'
        for name, sha in [('pkl', 'pkl_sha256'), ('splits_json', 'splits_json_sha256')]:
            assert digest(config[name]) == config[sha]
        case_report = json.loads(args.case_report.read_text())
        assert case_report['status'] == 'complete'
        assert case_report['checkpoint_sha256'] == digest(args.checkpoint)
        assert case_report['pkl_sha256'] == config['pkl_sha256']
        assert case_report['splits_json_sha256'] == config['splits_json_sha256']
        bindings = json.loads((args.case_report.parent / 'reference_bindings.json').read_text())
        assert bindings['bands']['best_checkpoint_sha256'] == digest(args.reference_bands / 'checkpoint_best.pt')
        selected = case_report['training_case_ids']
        assert len(selected) == len(set(selected)) == 32
        splits = json.loads(Path(config['splits_json']).read_text())[0]
        assert set(selected) <= set(splits['train']) and not set(selected) & set(splits['val'])
        data_args = SimpleNamespace(pkl=Path(config['pkl']), splits_json=Path(config['splits_json']), dataset='MSD',
            fold=0, spatial_size=run['spatial_size'], resize=run['resize'], batch_size=1, num_workers=0,
            num_classes=3, plain_tensors=True)
        train_loader, _, _, train_n, val_n = trainer.build_data(data_args, torch.Generator().manual_seed(0))
        assert (train_n, val_n) == (208, 52)
        index = {item['case_name']: i for i, item in enumerate(train_loader.dataset.data)}
        loader = DataLoader(Subset(train_loader.dataset, [index[name] for name in selected]), batch_size=1, shuffle=False)
        model = trainer.build_swinunetr(tuple(run['spatial_size']), 3, device, drop_rate=0, activation_checkpointing=False)
        dice = DiceLoss(to_onehot_y=True, softmax=True)
        band = OuterBoundaryBandLoss()
        edge = BoundaryEdgeConsistencyLoss()
        band_weight = run['constraint_config']['bands_weight']
        assert band_weight > 0 and run['constraint_config'].get('bands_degree_alpha', 0) == 0
        source_config = json.loads((args.checkpoint.parent / 'config.json').read_text())
        assert source_config['run']['epochs'] == 5 and source_config['run']['constraint_set'] == 'none'
        assert source_config['pkl_sha256'] == config['pkl_sha256'] and source_config['splits_json_sha256'] == config['splits_json_sha256']
        assert source_config['runtime_provenance'] == trainer.collect_runtime_provenance()
        trainer.validate_calibration_execution(source_config['execution_provenance'],
            trainer.collect_execution_provenance(device), allow_mig_profile_change=True)
        assert source_config['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py'] == digest(REPO / 'baselines/swin_unetr/swin_unetr.py')
        records, aggregate = {}, {}
        finite, cosines, ratios = True, [], []
        edge_weight = None
        for stage, path, expected_config in [('calibration', args.checkpoint, source_config),
                ('converged', args.reference_bands / 'checkpoint_best.pt', config)]:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert json.loads(json.dumps(checkpoint['run'])) == expected_config['run']
            if stage == 'converged': assert checkpoint['epoch'] == completed['selected_epoch']
            else: assert checkpoint['epoch'] == 5
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            model.eval()
            generator = torch.Generator().manual_seed(1)
            records[stage] = []
            for case_name, batch in zip(selected, loader, strict=True):
                assert str(batch['case_name'][0]) == case_name
                images, labels, augmentation = augment_mild(batch['image'].to(device), batch['label'].to(device), generator=generator)
                with torch.no_grad(), torch.autocast('cuda', enabled=True):
                    raw = model(images)
                logits = raw.detach().float().requires_grad_(True)
                base_loss = dice(logits, labels) + band_weight * band(logits, labels).loss
                edge_loss = edge(logits, labels)
                g_base = torch.autograd.grad(base_loss, logits, retain_graph=True)[0]
                g_edge = torch.autograd.grad(edge_loss, logits)[0]
                base_rms = float(g_base.square().mean().sqrt())
                edge_rms = float(g_edge.square().mean().sqrt())
                finite &= bool(torch.isfinite(g_base).all() and torch.isfinite(g_edge).all()) and base_rms > 0 and edge_rms > 0
                cosine = float((g_base.flatten() @ g_edge.flatten()) / (g_base.norm() * g_edge.norm()).clamp_min(1e-30))
                row = {'case_name': case_name, 'base_loss': float(base_loss.detach()), 'edge_loss': float(edge_loss.detach()),
                       'base_gradient_rms': base_rms, 'edge_gradient_rms': edge_rms,
                       'cosine': cosine, 'augmentation': augmentation}
                if stage == 'calibration':
                    ratios.append(edge_rms / base_rms)
                else:
                    cosines.append(cosine)
                    inner, outer = build_boundary_bands(labels > 0)
                    row['original'] = counts(logits, labels, inner, outer)
                    combined = g_base + edge_weight * g_edge
                    scale = float(g_base.abs().max())
                    row['steps'] = {}
                    for step in (.25, 1.0, 4.0):
                        key = str(step)
                        row['steps'][key] = {}
                        aggregate.setdefault(key, {})
                        for method, gradient in [('base', g_base), ('combined', combined), ('edge_only', edge_weight * g_edge)]:
                            updated = logits.detach() - step * gradient / scale
                            values = counts(updated, labels, inner, outer)
                            row['steps'][key][method] = values
                            target = aggregate[key].setdefault(method, {k: 0 for k in values})
                            for k, v in values.items(): target[k] += v
                    foreground_odds_derivative = g_edge[:, 0] - (logits[:, 1:].softmax(1) * g_edge[:, 1:]).sum(1)
                    pred_fg, true_fg = logits.argmax(1) > 0, labels[:, 0] > 0
                    row['error_direction'] = {}
                    for name, mask, correct_direction in [
                        ('fn', true_fg & ~pred_fg, foreground_odds_derivative > 0),
                        ('fp', ~true_fg & pred_fg, foreground_odds_derivative < 0)]:
                        row['error_direction'][name] = {'count': int(mask.sum()), 'helpful': int((mask & correct_direction).sum()),
                            'zero_gradient': int((mask & (foreground_odds_derivative == 0)).sum())}
                records[stage].append(row)
            if stage == 'calibration':
                edge_weight = min(.1 / float(np.median(ratios)), .5 / float(np.quantile(ratios, .95)))
                print(json.dumps({'stage': stage, 'edge_weight': edge_weight, 'band_weight': band_weight}), flush=True)
        gate = decide_gate(aggregate, finite, float(np.mean(cosines)))
        directions = {name: {key: sum(row['error_direction'][name][key] for row in records['converged'])
                            for key in ('count', 'helpful', 'zero_gradient')} for name in ('fn', 'fp')}
        original = {key: sum(row['original'][key] for row in records['converged']) for key in records['converged'][0]['original']}
        save(args.root / 'cases.json', records)
        save(args.root / 'summary.json', {'status': 'complete', 'job_id': os.getenv('SLURM_JOB_ID'),
            'cases': len(selected), 'calibration': {'edge_weight': edge_weight, 'band_weight': band_weight,
                'median_edge_over_base_rms': float(np.median(ratios)), 'p95_edge_over_base_rms': float(np.quantile(ratios, .95)),
                'target_ratio': .1, 'cap_ratio': .5},
            'mean_gradient_cosine': float(np.mean(cosines)), 'error_direction': directions,
            'original': original, 'logit_interventions': aggregate, 'gate': gate,
            'case_report_sha256': digest(args.case_report), 'calibration_checkpoint_sha256': digest(args.checkpoint),
            'converged_checkpoint_sha256': digest(args.reference_bands / 'checkpoint_best.pt'),
            'runtime': trainer.collect_runtime_provenance(), 'execution': trainer.collect_execution_provenance(device),
            'scope': 'Training-only augmented cases. Paired logit steps share base max-gradient normalization; step 4 is stress-only, excluded from gate.'})
        print(json.dumps({'stage': 'converged', 'gate': gate}), flush=True)
    except BaseException:
        import traceback
        save(args.root / 'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
