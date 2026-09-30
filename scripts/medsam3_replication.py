"""Freeze a matched three-fold study and summarize paired replication results."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import nibabel as nib
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def required_quota_bytes(study, pending):
    """BeeGFS charges both copies of buddy-mirrored artifacts against quota."""
    factor = study['storage_replication_factor']
    if factor != 2:
        raise ValueError('This study requires the verified two-copy BeeGFS budget.')
    return math.ceil(study['reserve_gib']*1024**3) + factor*sum(
        study['artifact_budget_bytes'][name] for name in pending)


def freeze_study(data, pilot, output):
    folds = json.loads((data/'splits_final.json').read_text())
    template = json.loads(pilot.read_text())
    if digest(data/'splits_final.json') != template['split_sha256']:
        raise ValueError('Source split differs from the pilot.')
    excluded = set(template['train_cases'] + template['evaluation_cases'])
    chosen, used = {}, set()
    for fold in (0, 1, 2):
        pool = sorted(set(folds[fold]['train']) - excluded - used)
        chosen[fold] = sorted(random.Random(20260929+fold).sample(pool, 5))
        used.update(chosen[fold])
    protocols, shapes, eval_seen = {}, {}, set()
    for fold in (0, 1, 2):
        cases = sorted(set(folds[fold]['val']) - excluded - used)
        if not cases or eval_seen & set(cases):
            raise ValueError('Empty or overlapping evaluation folds.')
        eval_seen.update(cases)
        for case in cases:
            shapes[case] = list(nib.load(data/'imagesTr'/f'{case}_0000.nii.gz').shape)
        for seed in (17, 83, 191):
            name = f'fold{fold}_seed{seed}'
            protocol = dict(template, train_cases=chosen[fold], evaluation_cases=cases,
                            seed=seed, fold_index=fold, calibration_seed=20260929,
                            plot_cases=cases[:1], study_run=name)
            protocols[name] = protocol
    if used & eval_seen or len(used) != 15:
        raise ValueError('Global support/evaluation overlap.')
    # No compression benefit assumed: float32 probability + uint8 hard mask.
    # 80 MiB per adapter exceeds the measured 74.3 MB; 12 MiB/run covers metadata.
    per_run = {}
    for name, protocol in protocols.items():
        voxels = sum(math.prod(shapes[c]) for c in protocol['evaluation_cases'])
        per_run[name] = 3*80*1024**2 + math.ceil(voxels*3*5*1.05) + 12*1024**2
    manifest = {}
    for case in sorted(used | eval_seen):
        for name in (f'imagesTr/{case}_0000.nii.gz', f'labelsTr/{case}.nii.gz'):
            manifest[name] = digest(data/name)
    study = dict(study_id='medsam3_replication_20260928', folds=[0, 1, 2], seeds=[17, 83, 191],
                 protocols=protocols, excluded_pilot_cases=sorted(excluded),
                 training_union=sorted(used), evaluation_union=sorted(eval_seen),
                 data_hashes=manifest, artifact_budget_bytes=per_run, reserve_gib=1.0,
                 storage_replication_factor=2,
                 primary_contrast=['bands_edge', 'baseline'],
                 secondary_contrasts=[['bands', 'baseline'], ['bands_edge', 'bands']],
                 contrast_roles={'bands_edge_minus_baseline': 'primary',
                                 'bands_minus_baseline': 'secondary exploratory',
                                 'bands_edge_minus_bands': 'descriptive'},
                 bootstrap_draws=20000, bootstrap_seed=20260930,
                 inference_unit='volume; patient mapping unavailable',
                 analysis='paired crossed fold/seed bootstrap with shared within-fold case draws; exact fold sign-flip sensitivity',
                 inference_caveat='Only three fold blocks; bootstrap tails are approximate and no patient-independent claim is possible.')
    validate_study(study, folds)
    output.mkdir(parents=True, exist_ok=False)
    (output/'study.json').write_text(json.dumps(study, indent=2)+'\n')
    for name, protocol in protocols.items():
        (output/f'{name}.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return study


def validate_study(study, folds):
    expected = {f'fold{f}_seed{s}' for f in study['folds'] for s in study['seeds']}
    if set(study['protocols']) != expected:
        raise ValueError('Incomplete or extra run matrix.')
    train_union, evaluation_union = set(), set()
    blocked = set(study['excluded_pilot_cases'])
    template = next(iter(study['protocols'].values()))
    fixed_keys = ('updates', 'accumulation', 'warmup_updates', 'learning_rate',
                  'final_learning_rate', 'arms', 'calibration_slabs_per_volume',
                  'constraint_warmup_updates', 'gradient_ratio_target', 'gradient_ratio_p95_cap',
                  'calibration_seed', 'base_sha256', 'split_sha256')
    for fold in study['folds']:
        runs = [study['protocols'][f'fold{fold}_seed{s}'] for s in study['seeds']]
        train, evaluation = set(runs[0]['train_cases']), set(runs[0]['evaluation_cases'])
        if len(train) != 5 or not evaluation or train & evaluation or (train | evaluation) & blocked:
            raise ValueError('Invalid five-shot split or pilot contamination.')
        if train & train_union or evaluation & evaluation_union:
            raise ValueError('Fold support sets or evaluation sets overlap.')
        if not train <= set(folds[fold]['train']) or not evaluation <= set(folds[fold]['val']):
            raise ValueError('Source fold membership mismatch.')
        for seed, run in zip(study['seeds'], runs):
            if run['seed'] != seed or run['fold_index'] != fold:
                raise ValueError('Run identity mismatch.')
            if run['train_cases'] != runs[0]['train_cases'] or run['evaluation_cases'] != runs[0]['evaluation_cases']:
                raise ValueError('Cases vary within a fold across seeds.')
            if any(run[k] != template[k] for k in fixed_keys):
                raise ValueError('Training protocol differs across runs.')
        sources = {run.get('calibration_source_run') for run in runs}
        if len(sources) != 1:
            raise ValueError('Calibration reuse policy varies within a fold.')
        if sources != {None} and sources != {f'fold{fold}_seed{study["seeds"][0]}'}:
            raise ValueError('Shared calibration must come from the first seed of its own fold.')
        train_union.update(train)
        evaluation_union.update(evaluation)
    if train_union & evaluation_union:
        raise ValueError('A training volume appears in another evaluation fold.')
    if sorted(train_union) != study['training_union'] or sorted(evaluation_union) != study['evaluation_union']:
        raise ValueError('Declared union mismatch.')


def extend_study(data, prior_path, output):
    """Add the next unused fold without recycling prior evaluation cases as support."""
    prior = json.loads(prior_path.read_text())
    folds = json.loads((data/'splits_final.json').read_text())
    validate_study(prior, folds)
    unused = sorted(set(range(len(folds)))-set(prior['folds']))
    if not unused:
        raise ValueError('No unused dataset fold remains.')
    fold = unused[0]
    blocked = set(prior['training_union']+prior['evaluation_union']+prior['excluded_pilot_cases'])
    pool = sorted(set(folds[fold]['train'])-blocked)
    train = sorted(random.Random(20260929+fold).sample(pool, 5))
    evaluation = sorted(set(folds[fold]['val'])-set(prior['training_union']+prior['excluded_pilot_cases']))
    template = next(iter(prior['protocols'].values()))
    protocols = dict(prior['protocols'])
    budgets = dict(prior['artifact_budget_bytes'])
    voxels = sum(math.prod(nib.load(data/'imagesTr'/f'{case}_0000.nii.gz').shape) for case in evaluation)
    new_runs = []
    for seed in prior['seeds']:
        name = f'fold{fold}_seed{seed}'
        protocols[name] = dict(template, train_cases=train, evaluation_cases=evaluation,
                               seed=seed, fold_index=fold, plot_cases=evaluation[:1], study_run=name,
                               calibration_source_run=f'fold{fold}_seed{prior["seeds"][0]}')
        budgets[name] = 3*80*1024**2 + math.ceil(voxels*3*5*1.05) + 12*1024**2
        new_runs.append(name)
    hashes = dict(prior['data_hashes'])
    for case in train+evaluation:
        for name in (f'imagesTr/{case}_0000.nii.gz', f'labelsTr/{case}.nii.gz'):
            hashes[name] = digest(data/name)
    study = dict(prior, study_id='medsam3_fold3_extension_20260929',
                 prior_study_sha256=digest(prior_path), prior_runs=list(prior['protocols']),
                 folds=prior['folds']+[fold], protocols=protocols, new_runs=new_runs,
                 training_union=sorted(prior['training_union']+train),
                 evaluation_union=sorted(prior['evaluation_union']+evaluation),
                 data_hashes=hashes, artifact_budget_bytes=budgets,
                 retains_prior_calibration_deviation=True,
                 inference_caveat='Exploratory extension requested after observing three-fold outcomes. All prior folds, including the negative fold, are retained. Four fold blocks and no verified patient mapping cannot establish patient-independent significance.',
                 extension_policy='Next unused fold; support excludes every prior training, evaluation and pilot case; new fold reuses one training-only calibration exactly across seeds.')
    validate_study(study, folds)
    output.mkdir(parents=True, exist_ok=False)
    (output/'study.json').write_text(json.dumps(study, indent=2)+'\n')
    for name, protocol in protocols.items():
        (output/f'{name}.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return study


def paired_statistics(blocks, draws=20000, seed=20260930):
    """Crossed fold/seed resampling; within-fold cases stay paired across seeds."""
    if (len(blocks) < 2 or draws < 1 or any(b.ndim != 2 or 0 in b.shape or not np.isfinite(b).all() for b in blocks)
            or len({b.shape[0] for b in blocks}) != 1):
        raise ValueError('Require finite seed-by-case arrays for multiple fold blocks.')
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        means = []
        seeds = rng.integers(blocks[0].shape[0], size=blocks[0].shape[0])
        for f in rng.integers(len(blocks), size=len(blocks)):
            block = blocks[f]
            cases = rng.integers(block.shape[1], size=block.shape[1])
            means.append(block[np.ix_(seeds, cases)].mean())
        estimates[i] = np.mean(means)
    fold_means = np.asarray([b.mean() for b in blocks])
    observed = float(fold_means.mean())
    permuted = [abs(np.mean(fold_means*np.asarray(signs)))
                for signs in itertools.product((-1, 1), repeat=len(blocks))]
    p = float(np.mean(np.asarray(permuted) >= abs(observed)-1e-14))
    run_means = np.concatenate([b.mean(1) for b in blocks])
    return dict(mean_delta=observed,
                exploratory_hierarchical_ci95=np.quantile(estimates, [.025, .975]).tolist(),
                fold_sign_flip_p_two_sided=p, fold_deltas=fold_means.tolist(),
                run_deltas=run_means.tolist(), positive_runs=int((run_means > 0).sum()),
                runs=len(run_means), positive_folds=int((fold_means > 0).sum()),
                fold_sd=float(fold_means.std(ddof=1)),
                within_fold_seed_sd=[float(b.mean(1).std(ddof=1)) for b in blocks],
                mean_seed_averaged_case_win_rate=float(np.mean([(b.mean(0) > 0).mean() for b in blocks])))


def aggregate(study_path, results, output, *, report_calibration_deviation=False):
    study = json.loads(study_path.read_text())
    records, run_table = {}, []
    for name, protocol in study['protocols'].items():
        path = results/name
        result = json.loads((path/'results.json').read_text())
        audit = json.loads((path/'audit.json').read_text())
        if (result['protocol'] != protocol or result['status'] != 'complete' or audit['status'] != 'passed'
                or audit.get('results_sha256') != digest(path/'results.json')):
            raise ValueError(f'Unverified or mismatched run: {name}')
        records[name] = result
        run_table.append(dict(run=name, fold=protocol['fold_index'], seed=protocol['seed'],
                              evaluation_volumes=len(protocol['evaluation_cases']), **result['mean_dice']))
    # Calibration RNG is fixed within a fold; weights must match across seeds.
    calibration_checks = []
    for fold in study['folds']:
        fold_records = [records[f'fold{fold}_seed{s}'] for s in study['seeds']]
        if fold_records[0]['protocol'].get('calibration_source_run'):
            if any(r['calibration'] != fold_records[0]['calibration'] for r in fold_records[1:]):
                raise ValueError('Shared calibration is not exactly identical across seeds.')
        for arm in ('bands', 'edge'):
            values = [records[f'fold{fold}_seed{s}']['calibration'][arm]['weight'] for s in study['seeds']]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError('Invalid calibration coefficient.')
            calibration_checks.append(dict(fold=fold, constraint=arm, values=values,
                passed=bool(np.allclose(values, values[0], rtol=1e-5, atol=1e-8)),
                max_relative_deviation=float(np.max(np.abs(np.asarray(values)/values[0]-1)))))
    calibration_passed = all(row['passed'] for row in calibration_checks)
    if not calibration_passed and not report_calibration_deviation:
        raise ValueError('Unexpected calibration variation within fold; strict study gate failed.')
    contrasts = {}
    for positive, negative in [study['primary_contrast']] + study['secondary_contrasts']:
        blocks = []
        for fold in study['folds']:
            rows = []
            for seed in study['seeds']:
                name = f'fold{fold}_seed{seed}'
                rows.append([records[name]['per_case'][c][positive]['dice'] - records[name]['per_case'][c][negative]['dice']
                             for c in study['protocols'][name]['evaluation_cases']])
            blocks.append(np.asarray(rows))
        contrasts[f'{positive}_minus_{negative}'] = paired_statistics(
            blocks, study['bootstrap_draws'], study['bootstrap_seed'])
    report = dict(status='complete' if calibration_passed else 'reported_with_calibration_deviation',
                  strict_study_gate_passed=calibration_passed,
                  calibration_consistency=dict(passed=calibration_passed, rtol=1e-5, atol=1e-8,
                                               checks=calibration_checks),
                  analysis_code_sha256=digest(__file__),
                  runs=run_table, contrasts=contrasts,
                  mean_dice={arm: float(np.mean([r[arm] for r in run_table]))
                             for arm in ('baseline', 'bands', 'bands_edge')},
                  averaging='equal fold, equal seed; each run uses mean volume Dice',
                  primary='bands_edge_minus_baseline',
                  contrast_roles=study.get('contrast_roles', {}),
                  limitation=study['inference_caveat'],
                  statistical_note=f'Repeated case/seed predictions are not independent. With {len(study["folds"])} fold blocks, the minimum two-sided exact sign-flip p-value is {2/2**len(study["folds"]):g}. Bootstrap intervals are exploratory, not a significance declaration.',
                  study_sha256=digest(study_path))
    output.mkdir(parents=True, exist_ok=True)
    (output/'aggregate.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = [f'# MedSAM3 {len(study["folds"])}-fold, {len(study["seeds"])}-seed results', '',
             f'All {3*len(run_table)} fine-tuning arms completed and each run passed the independent saved-mask/training audit.', '', study['inference_caveat'], '']
    if not calibration_passed:
        lines += ['**The strict cross-seed calibration gate did not pass.** This is an explicitly requested descriptive report of all observed runs, with the original threshold retained. No coefficients, predictions, protocols or runs were changed or excluded. The cause and effect of the calibration differences remain unresolved; this is not a fully compliant replication claim.', '',
                  '| Fold | Constraint | Maximum relative coefficient difference | Original check |',
                  '|---|---|---:|---|']
        for row in calibration_checks:
            lines.append(f"| {row['fold']} | {row['constraint']} | {100*row['max_relative_deviation']:.6f}% | {'passed' if row['passed'] else 'failed'} |")
        lines.append('')
    lines += [
             '| Fold | Seed | Validation volumes | Baseline Dice | Bands Dice | Bands + edge Dice |',
             '|---|---:|---:|---:|---:|---:|']
    for row in run_table:
        lines.append(f"| {row['fold']} | {row['seed']} | {row['evaluation_volumes']} | {row['baseline']:.6f} | {row['bands']:.6f} | {row['bands_edge']:.6f} |")
    lines += ['', '| Contrast | Mean change (Dice points) | Exploratory 95% interval | Positive runs |', '|---|---:|---|---:|']
    for name, row in contrasts.items():
        lo, hi = [100*x for x in row['exploratory_hierarchical_ci95']]
        lines.append(f"| {name} | {100*row['mean_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['positive_runs']}/{row['runs']} |")
    lines += ['', report['statistical_note'], '', study['inference_caveat'], '',
              'The original pilot is excluded. All training volumes are excluded from every evaluation set. Training support sets differ across folds; each is held fixed across three new optimization seeds. Coefficients are calibrated from training data only with a fixed calibration seed per fold. No validation-based tuning or early stopping is used.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    for key in ('data', 'pilot', 'output'):
        prep.add_argument('--'+key, type=Path, required=True)
    extension = sub.add_parser('extend')
    for key in ('data', 'prior-study', 'output'):
        extension.add_argument('--'+key, type=Path, required=True)
    summary = sub.add_parser('aggregate')
    summary.add_argument('--report-calibration-deviation', action='store_true',
                         help='Report all observed runs while explicitly retaining a failed calibration gate; never relax its threshold.')
    for key in ('study', 'results', 'output'):
        summary.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        study = freeze_study(args.data, args.pilot, args.output)
        print(json.dumps(dict(runs=len(study['protocols']), training_volumes=len(study['training_union']),
                              evaluation_volumes=len(study['evaluation_union']),
                              artifact_budget_gib=sum(study['artifact_budget_bytes'].values())/1024**3)))
    elif args.mode == 'extend':
        study = extend_study(args.data, args.prior_study, args.output)
        print(json.dumps(dict(new_runs=study['new_runs'], minimum_free_gib=required_quota_bytes(study, study['new_runs'])/1024**3)))
    else:
        report = aggregate(args.study, args.results, args.output,
                           report_calibration_deviation=args.report_calibration_deviation)
        print(json.dumps(dict(status=report['status'], mean_dice=report['mean_dice'], contrasts=report['contrasts'])))


if __name__ == '__main__':
    main()
