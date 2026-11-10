"""Matched Dice controls and paired comparisons for all views and profile losses."""
import json
import csv
from pathlib import Path
from scripts.depth_supervision.utils import atomic_json


METRICS = ('macro_dice', 'anterior_dice', 'posterior_dice', 'foreground_dice', 'missed', 'extra', 'swaps',
           'surface_distance_mean_mm', 'hd95_mm')


def execution_metadata(folder, record):
    config_path = folder/'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    parameters = record.get('parameter_count', config.get('parameters'))
    runtime = record.get('training_seconds')
    basis = 'sum of epoch train/calibration/validation seconds; excludes checkpoint writes and final detailed evaluation'
    metrics_path = folder/'metrics.csv'
    if runtime is None and metrics_path.exists():
        rows = list(csv.DictReader(metrics_path.open()))
        if rows and all('epoch_seconds' in r for r in rows):
            runtime = sum(float(r['epoch_seconds']) for r in rows)
        elif rows and all('elapsed_seconds' in r for r in rows):
            values = [float(r['elapsed_seconds']) for r in rows]
            # Legacy elapsed counters restart on process resume.
            runtime = sum(b-a if b >= a else b for a, b in zip([0., *values[:-1]], values))
            basis = 'legacy elapsed counters; approximate across resumed processes'
    return dict(parameter_count=parameters, training_seconds=runtime, runtime_basis=basis if runtime is not None else 'unavailable')


def baseline_folder(p, fraction, arm, root=None):
    if arm == 'informed' or '_mri_' in arm:
        if root is None:
            raise ValueError('Informed Dice baseline belongs to the current campaign')
        return Path(root)/'runs'/f'{fraction}_dice_s0_{arm}'
    if arm == 'native':
        return Path(p['prior_root'])/'runs'/fraction/'dice'
    return Path(p['branch_baseline_root'])/'runs'/f'{fraction}_dice_ablation_s0_{arm}_none'


def paired_difference(base, new):
    b, n = [json.loads((folder/'validation_best.json').read_text()) for folder in (base, new)]
    by_name = {row['case_name']: row for row in b['cases']}
    if set(by_name) != {row['case_name'] for row in n['cases']}:
        raise ValueError('Comparison requires identical validation cases')
    return [dict(case_name=row['case_name'],
        **{k: (row[k]-by_name[row['case_name']][k] if row.get(k) is not None and by_name[row['case_name']].get(k) is not None else None)
           for k in METRICS}) for row in n['cases']]


def report(p, root):
    root = Path(root)
    rows, paired, seen = [], {}, set()
    completed, folders = 0, {}
    for entry in p['runs']:
        fraction, arm = entry['fraction'], entry.get('arm', 'native')
        variant = entry['profile_variant']
        base, new = baseline_folder(p, fraction, arm, root), root/'runs'/entry['id']
        if arm != 'informed' and '_mri_' not in arm and (fraction, arm) not in seen and (base/'completion.json').exists():
            record = json.loads((base/'completion.json').read_text())
            rows.append(dict(fraction=fraction, arm=arm, objective='dice_baseline',
                             best_epoch=record['best_epoch'], epochs=record['completed_epochs'],
                             **execution_metadata(base, record), **record['validation']))
            seen.add((fraction, arm))
        if not (new/'completion.json').exists():
            continue
        record = json.loads((new/'completion.json').read_text())
        label = {'none': 'dice_baseline', 'pooled': 'dice_profile', 'multiscale': 'dice_multiscale'}[variant]
        rows.append(dict(fraction=fraction, arm=arm, objective=label,
                         best_epoch=record['best_epoch'], epochs=record['completed_epochs'],
                         **execution_metadata(new, record), **record['validation']))
        completed += 1
        if new != base and (base/'completion.json').exists():
            paired[entry['id']+'_minus_dice'] = paired_difference(base, new)
        seen.add((fraction, arm))
        folders[(fraction, arm, variant)] = new
    for fraction, arm in seen:
        original = folders.get((fraction, arm, 'pooled'))
        finer = folders.get((fraction, arm, 'multiscale'))
        if original and finer:
            suffix = '' if arm == 'native' else '_'+arm
            paired[fraction+suffix+'_multiscale_minus_pooled'] = paired_difference(original, finer)
    for fraction, arms in p.get('view_pairs', {}).items():
        for variant in ('pooled', 'multiscale'):
            plane, unfolded = [folders.get((fraction, arm, variant)) for arm in arms]
            if plane and unfolded:
                paired[fraction+'_'+variant+'_unfold_minus_plane'] = paired_difference(plane, unfolded)
    for fraction in ('5pct', 'full'):
        for variant in ('none', 'pooled', 'multiscale'):
            informed = folders.get((fraction, 'informed', variant))
            if not informed:
                continue
            additional = p.get('mri_view_pairs', {}).get(fraction, [])
            native_aux = p.get('native_mri_arms', {}).get(fraction)
            for arm in ['native', *p.get('view_pairs', {}).get(fraction, []), *additional, *([native_aux] if native_aux else [])]:
                control = (baseline_folder(p, fraction, arm, root) if variant == 'none' else
                           folders.get((fraction, arm, variant)))
                if control and (control/'completion.json').exists():
                    paired[f'{fraction}_{variant}_informed_minus_{arm}'] = paired_difference(control, informed)
    for fraction, arms in p.get('mri_view_pairs', {}).items():
        for variant in ('none', 'pooled', 'multiscale'):
            suffix = '' if variant == 'none' else '_'+variant
            for arm, original in zip(arms, p['view_pairs'][fraction]):
                new = folders.get((fraction, arm, variant))
                if not new:
                    continue
                position = baseline_folder(p, fraction, original) if variant == 'none' else folders.get((fraction, original, variant))
                native = baseline_folder(p, fraction, 'native') if variant == 'none' else folders.get((fraction, 'native', variant))
                for label, base in [('position_branch', position), ('native', native),
                                    ('informed', folders.get((fraction, 'informed', variant)))]:
                    if base and (base/'completion.json').exists():
                        paired[f'{fraction}_{arm}{suffix}_minus_{label}'] = paired_difference(base, new)
                if position and (position/'completion.json').exists():
                    paired[f'{fraction}_{original}_{variant}_position_minus_mri'] = paired_difference(new, position)
            plane, unfolded = [folders.get((fraction, arm, variant)) for arm in arms]
            if plane and unfolded:
                paired[fraction+'_mri'+suffix+'_unfold_minus_plane'] = paired_difference(plane, unfolded)
    for fraction, arm in p.get('native_mri_arms', {}).items():
        for variant in ('none', 'pooled', 'multiscale'):
            auxiliary = folders.get((fraction, arm, variant))
            if not auxiliary:
                continue
            ordinary = baseline_folder(p, fraction, 'native') if variant == 'none' else folders.get((fraction, 'native', variant))
            if ordinary and (ordinary/'completion.json').exists():
                paired[f'{fraction}_{variant}_native_aux_minus_ordinary'] = paired_difference(ordinary, auxiliary)
            for alternative in p['mri_view_pairs'][fraction]:
                view = folders.get((fraction, alternative, variant))
                if view:
                    paired[f'{fraction}_{variant}_{alternative}_minus_native_aux'] = paired_difference(auxiliary, view)
    result = dict(protocol_sha256=p['identity_sha'], new_completed=completed, rows=rows, paired_deltas=paired)
    atomic_json(root/'comparison.json', result)
    return result
