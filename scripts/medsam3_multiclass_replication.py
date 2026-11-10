"""Independent saved-mask audit and equal-seed/equal-fold multiclass reporting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_medsam3_supervised import check_initial_losses

ARMS = ['baseline', 'bands', 'bands_edge']
CLASSES = ['background', 'anterior', 'posterior']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def confusion_metrics(cm):
    """Disjoint foreground errors and swaps; per-class FP/FN also include swaps."""
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (3, 3) or (cm < 0).any():
        raise ValueError('Invalid 3x3 confusion matrix')
    row = {}
    for c, name in enumerate(CLASSES):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum()-tp); fn = int(cm[c].sum()-tp)
        row[name+'_dice'] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        row[name+'_fp'] = fp
        row[name+'_fn'] = fn
    row['ap_mean_dice'] = (row['anterior_dice']+row['posterior_dice'])/2
    row['foreground_fp'] = int(cm[0, 1]+cm[0, 2])
    row['foreground_fn'] = int(cm[1, 0]+cm[2, 0])
    row['a_to_p'] = int(cm[1, 2])
    row['p_to_a'] = int(cm[2, 1])
    row['ap_swaps'] = row['a_to_p']+row['p_to_a']
    row['total_error_voxels'] = row['foreground_fp']+row['foreground_fn']+row['ap_swaps']
    return row


def average_rows(rows):
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('Missing or inconsistent metric rows')
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def audit(result_dir, data, protocol_path=None):
    if not __debug__:
        raise RuntimeError('Audit requires assertions enabled')
    result_dir, data = Path(result_dir), Path(data)
    report = json.loads((result_dir/'results.json').read_text())
    p = report['protocol']
    assert report['status'] == 'complete' and p['arms'] == ARMS and p['classes'] == CLASSES
    if protocol_path:
        assert p == json.loads(Path(protocol_path).read_text()), 'Protocol changed'
    assert set(report['results']) == {'train', 'validation'}
    assert not set(p['train_cases']) & set(p['evaluation_cases'])
    assert report['probability_class_order'] == CLASSES
    expected_hashes = {f'{case}/{arm}_{suffix}.nii.gz' for case in p['train_cases']+p['evaluation_cases']
                       for arm in ARMS for suffix in ('mask', 'probabilities')}
    assert set(report['output_hashes']) == expected_hashes
    for name, expected in report['output_hashes'].items():
        assert digest(result_dir/name) == expected, name
    for name, expected in p['data_hashes'].items():
        assert digest(data/name) == expected, name
    assert digest(data/'splits_final.json') == p['split_sha256']
    summaries, checked = {}, 0
    for split, cases in [('train', p['train_cases']), ('validation', p['evaluation_cases'])]:
        recorded = report['results'][split]
        assert set(recorded['per_case']) == set(cases)
        independent = {arm: [] for arm in ARMS}
        for case in cases:
            truth_img = nib.as_closest_canonical(nib.load(data/'labelsTr'/f'{case}.nii.gz'))
            truth = truth_img.get_fdata()
            assert np.isin(truth, [0, 1, 2]).all()
            for arm in ARMS:
                mask_img = nib.load(result_dir/case/f'{arm}_mask.nii.gz')
                prob_img = nib.load(result_dir/case/f'{arm}_probabilities.nii.gz')
                mask, prob = mask_img.get_fdata(), prob_img.get_fdata()
                assert mask.shape == truth.shape and prob.shape == (*truth.shape, 3)
                assert np.allclose(mask_img.affine, truth_img.affine) and np.allclose(prob_img.affine, truth_img.affine)
                assert np.isin(mask, [0, 1, 2]).all() and np.isfinite(prob).all()
                assert (prob >= 0).all() and (prob <= 1).all() and np.allclose(prob.sum(-1), 1, atol=1e-5)
                assert np.array_equal(mask, prob.argmax(-1))
                cm = np.bincount((3*truth.astype(np.int64)+mask.astype(np.int64)).ravel(), minlength=9).reshape(3, 3)
                saved = recorded['per_case'][case][arm]
                assert cm.tolist() == saved['confusion_true_rows_predicted_columns']
                values = confusion_metrics(cm)
                for c, name in enumerate(CLASSES):
                    assert saved['per_class'][name]['tp'] == int(cm[c, c])
                    for k in ('fp', 'fn', 'dice'):
                        assert abs(saved['per_class'][name][k]-values[name+'_'+k]) < 1e-12
                for recorded_key, independent_key in [('ap_mean_dice', 'ap_mean_dice'),
                        ('anterior_to_posterior', 'a_to_p'), ('posterior_to_anterior', 'p_to_a'),
                        ('total_error_voxels', 'total_error_voxels')]:
                    assert abs(saved[recorded_key]-values[independent_key]) < 1e-12
                independent[arm].append(values)
                checked += 1
        summaries[split] = {arm: average_rows(rows) for arm, rows in independent.items()}
        for arm in ARMS:
            assert abs(summaries[split][arm]['ap_mean_dice']-recorded['mean_ap_dice'][arm]) < 1e-12
    schedule = json.loads((result_dir/'slab_schedule.json').read_text())
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    assert len(schedule) == p['updates']*p['accumulation']
    inventory = json.loads((result_dir/'training_inventory.json').read_text())
    assert [r['case'] for r in inventory] == p['train_cases']
    shapes = {r['case']: r['shape'] for r in inventory}
    exposures = {case: 0 for case in p['train_cases']}
    for case, anchor in schedule:
        assert case in exposures and 0 <= anchor < shapes[case][2]
        exposures[case] += 2
    traces = {}
    for arm in ARMS:
        t = report['training'][arm]
        assert t == json.loads((result_dir/arm/'training_summary.json').read_text())
        assert t['updates'] == p['updates'] and t['schedule_sha256'] == schedule_hash
        assert t['slice_exposures'] == exposures and t['frozen_before'] == t['frozen_after']
        assert t['adapter_before'] != t['adapter_after']
        assert digest(result_dir/arm/'final_lora_weights.pt') == t['final_adapter_sha256']
        rows = [json.loads(line) for line in (result_dir/arm/'training.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows] == list(range(1, p['updates']+1))
        for row in rows:
            assert np.isfinite([row[k] for k in ['native', 'bands', 'edge', 'total', 'gradient_norm']]).all()
            assert np.isclose(row['total'], row['native']+row['bands_weight']*row['bands']+row['edge_weight']*row['edge'], rtol=2e-5, atol=2e-4)
            ramp = min(row['update']/p['constraint_warmup_updates'], 1.)
            for constraint in ['bands', 'edge']:
                assert np.isclose(row[constraint+'_weight'], ramp*t[constraint+'_weight'])
        traces[arm] = rows
    for key in ['adapter_before', 'frozen_before', 'schedule_sha256']:
        assert len({t[key] for t in report['training'].values()}) == 1
    assert len({t['adapter_after'] for t in report['training'].values()}) == 3
    for arm in ARMS[1:]:
        assert [r['lr'] for r in traces[arm]] == [r['lr'] for r in traces['baseline']]
    initial_losses = check_initial_losses(traces, ARMS)
    t, cal = report['training'], report['calibration']
    assert t['baseline']['bands_weight'] == t['baseline']['edge_weight'] == t['bands']['edge_weight'] == 0
    assert t['bands']['bands_weight'] == t['bands_edge']['bands_weight'] == cal['bands']['weight'] > 0
    assert t['bands_edge']['edge_weight'] == cal['edge']['weight'] > 0
    assert json.loads((result_dir/'calibration.json').read_text()) == cal
    assert cal['evaluation_labels_used'] is False
    assert set(r['case'] for r in cal['slabs']) == set(p['train_cases'])
    for constraint in ['bands', 'edge']:
        assert sum(r[constraint+'_sq'] > 0 for r in cal['slabs']) == cal[constraint]['valid_slabs'] >= 5
    if p.get('calibration_source_run'):
        assert cal['source_run'] == p['calibration_source_run']
    result = dict(status='passed', run=p['study_run'], fold=p['fold_index'], seed=p['seed'],
                  results_sha256=digest(result_dir/'results.json'), calibration_sha256=digest(result_dir/'calibration.json'),
                  checked_masks=checked, checked_output_hashes=len(expected_hashes), checked_adapters=3,
                  initial_loss_check=initial_losses, summary=summaries,
                  averaging='Mean over volumes within this run; error counts are mean voxels per volume.')
    write(result_dir/'audit.json', result)
    return result


def aggregate(root):
    root = Path(root)
    study = json.loads((root/'study.json').read_text())
    records = []
    for name, p in study['protocols'].items():
        path = root/'results'/name
        record = json.loads((path/'audit.json').read_text())
        assert record['status'] == 'passed' and record['results_sha256'] == digest(path/'results.json')
        result = json.loads((path/'results.json').read_text())
        assert result['protocol'] == p
        assert record['run'] == name and record['fold'] == p['fold_index'] and record['seed'] == p['seed']
        records.append(record)
    rows = []
    for fold in study['folds']:
        subset = [r for r in records if r['fold'] == fold]
        assert sorted(r['seed'] for r in subset) == sorted(study['seeds'])
        assert len({r['calibration_sha256'] for r in subset}) == 1, 'Cross-seed calibration differs'
        for arm in ARMS:
            values = [r['summary']['validation'][arm] for r in subset]
            row = dict(fold=fold, arm=arm, **average_rows(values))
            row['ap_dice_seed_sd'] = float(np.std([v['ap_mean_dice'] for v in values], ddof=1))
            deltas = [r['summary']['validation'][arm]['ap_mean_dice']-r['summary']['validation']['baseline']['ap_mean_dice'] for r in subset]
            row['delta_vs_baseline'] = float(np.mean(deltas))
            row['positive_seeds'] = sum(d > 0 for d in deltas)
            rows.append(row)
    keys = records[0]['summary']['validation']['baseline'].keys()
    overall = {arm: {key: float(np.mean([r[key] for r in rows if r['arm'] == arm])) for key in keys} for arm in ARMS}
    result = dict(status='complete', folds=study['folds'], seeds=study['seeds'], runs=records,
        fold_validation=rows, overall_equal_fold=overall,
        averaging='Equal volume within seed; equal seed within fold; equal fold overall. FP/FN/swaps are mean voxels per volume, not pooled totals.',
        selection_note=study['selection_note'], pilot_seed46_included=False)
    write(root/'analysis/aggregate.json', result)
    lines = ['# Three-class MedSAM3 replication: audited validation results', '', study['selection_note'], '',
        'Primary Dice is the mean of anterior and posterior Dice, excluding background. Values first average volumes, then the three seeds within each fold. FP/FN below are disjoint foreground/background errors; A/P swaps are reported separately. Error units: mean voxels per volume.', '',
        '| Fold | Arm | A/P Dice (%) | Gain (pp) | Seed SD (pp) | Foreground FP | Foreground FN | A→P | P→A | Total swaps | Positive seeds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['fold']} | {r['arm']} | {100*r['ap_mean_dice']:.3f} | {100*r['delta_vs_baseline']:+.3f} | {100*r['ap_dice_seed_sd']:.3f} | {r['foreground_fp']:.2f} | {r['foreground_fn']:.2f} | {r['a_to_p']:.2f} | {r['p_to_a']:.2f} | {r['ap_swaps']:.2f} | {r['positive_seeds']}/3 |")
    lines += ['', '## Class-specific validation errors', '', 'Per-class FP and FN include class swaps; do not add these to the disjoint foreground-error table.', '',
        '| Fold | Arm | Class | Dice (%) | FP/volume | FN/volume |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        for c in CLASSES:
            lines.append(f"| {r['fold']} | {r['arm']} | {c} | {100*r[c+'_dice']:.3f} | {r[c+'_fp']:.2f} | {r[c+'_fn']:.2f} |")
    lines += ['', '## Individual seeds', '', '| Fold | Seed | Arm | A/P Dice (%) | FP/volume | FN/volume | Swaps/volume |', '|---|---|---|---:|---:|---:|---:|']
    for record in records:
        for arm in ARMS:
            v = record['summary']['validation'][arm]
            lines.append(f"| {record['fold']} | {record['seed']} | {arm} | {100*v['ap_mean_dice']:.3f} | {v['foreground_fp']:.2f} | {v['foreground_fn']:.2f} | {v['ap_swaps']:.2f} |")
    lines += ['', 'All saved predictions, raw confusion counts, input/output hashes, adapters, matched training traces and exact cross-seed calibration were checked before aggregation. The exploratory seed-46 pilot is excluded. Repeated seeds reuse each fold’s volumes and do not create independent patients.', '']
    (root/'analysis/RESULTS.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--results', type=Path, required=True)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--protocol', type=Path)
    a = sub.add_parser('aggregate')
    a.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results, args.data, args.protocol) if args.command == 'audit' else aggregate(args.root)
    print(json.dumps({k: result[k] for k in ('status', 'run', 'fold', 'seed', 'overall_equal_fold') if k in result}), flush=True)


if __name__ == '__main__':
    main()
