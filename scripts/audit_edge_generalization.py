#!/usr/bin/env python3
"""Inference-only train/validation comparison of the three selected checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from probe_edge_consistency import digest, save

METRICS = ('macro_dice', 'whole/dice', 'anterior/dice', 'posterior/dice',
           'balanced_boundary_error', 'inner_fn_rate', 'outer_fp_rate',
           'boundary_union_errors', 'boundary_class_errors')


def compare_splits(train, validation, *, candidate='edge', references=('dice', 'bands')):
    """Difference of paired model effects, with independent resampling of splits.

    Cases are paired across models, never across the distinct train/val splits.
    Intervals describe this fitted model on this development split, not seed or
    checkpoint-selection uncertainty.
    """
    import numpy as np
    assert set(train) == set(validation) == {candidate, *references}
    train_ids = [r['case_name'] for r in train[candidate]]
    val_ids = [r['case_name'] for r in validation[candidate]]
    assert not set(train_ids) & set(val_ids)
    assert len(set(train_ids)) == len(train_ids) and len(set(val_ids)) == len(val_ids)
    for records, names in ((train, train_ids), (validation, val_ids)):
        for rows in records.values():
            assert [r['case_name'] for r in rows] == names
    rng = np.random.default_rng(20260924)
    ti = rng.integers(len(train_ids), size=(10000, len(train_ids)))
    vi = rng.integers(len(val_ids), size=(10000, len(val_ids)))
    gaps, effects = {}, {}
    for model in train:
        gaps[model] = {metric: float(np.mean([r['metrics'][metric] for r in validation[model]])
                                    - np.mean([r['metrics'][metric] for r in train[model]]))
                       for metric in METRICS}
    for reference in references:
        effects[reference] = {}
        for metric in METRICS:
            td = np.array([a['metrics'][metric] - b['metrics'][metric]
                           for a, b in zip(train[candidate], train[reference], strict=True)])
            vd = np.array([a['metrics'][metric] - b['metrics'][metric]
                           for a, b in zip(validation[candidate], validation[reference], strict=True)])
            interval = np.quantile(vd[vi].mean(1) - td[ti].mean(1), [.025, .975])
            higher_better = metric.endswith('dice')
            train_improved = td.mean() > 0 if higher_better else td.mean() < 0
            val_improved = vd.mean() > 0 if higher_better else vd.mean() < 0
            effects[reference][metric] = {
                f'train_{candidate}_minus_reference': float(td.mean()),
                f'validation_{candidate}_minus_reference': float(vd.mean()),
                'effect_gap_validation_minus_train': float(vd.mean() - td.mean()),
                'effect_gap_bootstrap_95_interval': interval.tolist(),
                'train_improved_point_estimate': bool(train_improved),
                'validation_improved_point_estimate': bool(val_improved),
                'training_only_improvement_point_estimate': bool(train_improved and not val_improved)}
    return {'case_counts': {'train': len(train_ids), 'validation': len(val_ids)},
            'model_gaps_validation_minus_train': gaps, f'{candidate}_effect_relative_to': effects,
            'interpretation': 'Lower error/rate is better; higher Dice is better. Gaps use case means, never pooled split totals. Training-only improvement is descriptive, not proof of overfitting. Validation selected checkpoints and has been used for development; bootstrap excludes selection and seed uncertainty.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--edge-experiment', type=Path, required=True)
    parser.add_argument('--reference-dice', type=Path, required=True)
    parser.add_argument('--reference-bands', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'audit.lock').open('x') as handle:
        handle.write(str(os.getenv('SLURM_JOB_ID')))
    try:
        import torch
        from torch.utils.data import DataLoader
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        from train_edge_consistency_experiment import audit
        from run_degree_bands_ab import check_quota
        for name, sha in json.loads((REPO / 'PAYLOAD.json').read_text())['files'].items():
            assert digest(REPO / name) == sha, name
        assert json.loads((args.edge_experiment / 'training/completion.json').read_text())['status'] == 'complete'
        check_quota(args.output, 'start', .1)
        trainer.seed_everything(0)
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        runtime = trainer.collect_runtime_provenance()
        execution = trainer.collect_execution_provenance(device)
        configs, bindings = {}, {}
        for name, directory in [('dice', args.reference_dice), ('bands', args.reference_bands),
                                ('edge', args.edge_experiment / 'training/run')]:
            config = json.loads((directory / 'config.json').read_text())
            completion = json.loads((directory / 'completion_manifest.json').read_text())
            assert completion['status'] == 'complete' and completion['run'] == config['run']
            assert config['runtime_provenance'] == runtime
            trainer.validate_calibration_execution(config['execution_provenance'], execution, allow_mig_profile_change=True)
            assert config['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py'] == digest(REPO / 'baselines/swin_unetr/swin_unetr.py')
            checkpoint_path = directory / 'checkpoint_best.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['epoch'] == completion['selected_epoch']
            exported = torch.load(directory / 'MSD_fold0/model.pt', map_location='cpu', weights_only=True)
            assert digest(directory / 'MSD_fold0/model.pt') == completion['artifacts']['MSD_fold0/model.pt']
            assert checkpoint['model'].keys() == exported.keys()
            assert all(torch.equal(v, exported[k]) for k, v in checkpoint['model'].items())
            del checkpoint, exported
            configs[name] = (directory, config)
            bindings[name] = {'directory': str(directory), 'checkpoint_sha256': digest(checkpoint_path),
                              'config_sha256': digest(directory / 'config.json'), 'selected_epoch': completion['selected_epoch']}
        config = configs['edge'][1]
        for _, reference in configs.values():
            for key in ('pkl', 'splits_json'):
                assert digest(reference[key]) == reference[key + '_sha256'] == config[key + '_sha256']
            for key in ('fold', 'spatial_size', 'resize', 'plain_tensors', 'amp'):
                assert reference['run'][key] == config['run'][key]
        data_args = SimpleNamespace(pkl=Path(config['pkl']), splits_json=Path(config['splits_json']), dataset='MSD',
            fold=0, spatial_size=config['run']['spatial_size'], resize=False, batch_size=1, num_workers=0,
            num_classes=3, plain_tensors=True)
        training, validation, _, train_n, val_n = trainer.build_data(data_args, torch.Generator().manual_seed(0))
        assert (train_n, val_n) == (208, 52)
        training = DataLoader(training.dataset, batch_size=1, shuffle=False, num_workers=0)
        save(args.output / 'bindings.json', {'models': bindings, 'runtime': runtime, 'execution': execution,
             'pkl_sha256': config['pkl_sha256'], 'splits_json_sha256': config['splits_json_sha256'],
             'preprocessing': 'Identical unaugmented padded 64^3 cases; model.eval(); CUDA AMP; no training or checkpoint reselection.'})
        for name, loader in [('train', training), ('validation', validation)]:
            print(json.dumps({'starting_split': name, 'cases': len(loader.dataset)}), flush=True)
            audit(args.output / name, configs, device, loader)
        records = {name: json.loads((args.output / name / 'audit/cases.json').read_text()) for name in ('train', 'validation')}
        save(args.output / 'generalization.json', compare_splits(records['train'], records['validation']))
        save(args.output / 'completion.json', {'status': 'complete', 'job_id': os.getenv('SLURM_JOB_ID'),
                                             'train_cases': train_n, 'validation_cases': val_n, 'models': 3})
    except BaseException:
        import traceback
        save(args.output / 'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
