"""Independently verify saved Dice, prediction hashes and matched training traces."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

if __package__:
    from .medsam3_calibration_cache import calibration_context
else:
    from medsam3_calibration_cache import calibration_context


def check_initial_losses(traces, arms):
    """Allow a few float32 rounding units; parameter identity is checked separately."""
    tolerance = 4*float(np.finfo(np.float32).eps)
    reference = traces[arms[0]][0]
    differences = {}
    for arm in arms[1:]:
        differences[arm] = {}
        for name in ('native', 'bands', 'edge'):
            actual, expected = traces[arm][0][name], reference[name]
            if not (np.isfinite(actual) and np.isfinite(expected)) or not np.isclose(
                    actual, expected, rtol=tolerance, atol=tolerance):
                raise AssertionError(f'Initial {name} loss mismatch for {arm}: {actual} vs {expected}')
            differences[arm][name] = float(actual-expected)
    return dict(rtol=tolerance, atol=tolerance, differences_from_baseline=differences)


def main():
    if not __debug__:
        raise RuntimeError('Audit requires Python assertions enabled.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.results/'results.json').read_text())
    protocol = report['protocol']
    arms, cases = protocol['arms'], protocol['evaluation_cases']
    assert not set(protocol['train_cases']) & set(cases)
    assert report['status'] == 'complete'
    inventory = json.loads((args.results/'training_inventory.json').read_text())
    assert [r['case'] for r in inventory] == protocol['train_cases']
    for row in inventory:
        case = row['case']
        for directory, filename, key in (
                ('imagesTr', f'{case}_0000.nii.gz', 'image_sha256'),
                ('labelsTr', f'{case}.nii.gz', 'label_sha256')):
            assert hashlib.sha256((args.data/directory/filename).read_bytes()).hexdigest() == row[key]
    checked = 0
    independent = {arm: [] for arm in arms}
    for case in cases:
        label_path = args.data/'labelsTr'/f'{case}.nii.gz'
        assert hashlib.sha256(label_path.read_bytes()).hexdigest() == report['evaluation_label_hashes'][case]
        label = nib.as_closest_canonical(nib.load(label_path))
        truth = label.get_fdata() > 0
        for arm in arms:
            fields = {}
            for suffix in ('mask', 'probability'):
                relative = f'{case}/{arm}_{suffix}.nii.gz'
                path = args.results/relative
                assert hashlib.sha256(path.read_bytes()).hexdigest() == report['output_hashes'][relative]
                image = nib.load(path)
                assert image.shape == label.shape and np.allclose(image.affine, label.affine)
                fields[suffix] = image.get_fdata()
                checked += 1
            assert np.isfinite(fields['probability']).all()
            assert fields['probability'].min() >= 0 and fields['probability'].max() <= 1
            assert np.isin(fields['mask'], [0, 1]).all()
            hard = fields['mask'].astype(bool)
            assert np.array_equal(hard, fields['probability'] > .5)
            tp, fp, fn = int((hard & truth).sum()), int((hard & ~truth).sum()), int((~hard & truth).sum())
            score = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
            recorded = report['per_case'][case][arm]
            assert (tp, fp, fn) == (recorded['tp'], recorded['fp'], recorded['fn'])
            assert abs(score-recorded['dice']) < 1e-12
            independent[arm].append(score)
    traces = {}
    schedule = json.loads((args.results/'slab_schedule.json').read_text())
    digest = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    expected_exposures = {c: 0 for c in protocol['train_cases']}
    for case, _ in schedule:
        expected_exposures[case] += 2
    assert len(schedule) == protocol['updates']*protocol['accumulation']
    for arm in arms:
        summary = report['training'][arm]
        assert summary['schedule_sha256'] == digest
        assert summary['slice_exposures'] == expected_exposures
        assert summary['frozen_before'] == summary['frozen_after']
        assert summary['adapter_before'] != summary['adapter_after']
        trace = [json.loads(line) for line in (args.results/arm/'training.jsonl').read_text().splitlines()]
        assert [r['update'] for r in trace] == list(range(1, protocol['updates']+1))
        for row in trace:
            assert np.isfinite([row[k] for k in ('native','bands','edge','total','gradient_norm')]).all()
            expected = row['native']+row['bands_weight']*row['bands']+row['edge_weight']*row['edge']
            assert np.isclose(expected, row['total'], rtol=2e-5, atol=2e-4)
            ramp = min(row['update']/protocol['constraint_warmup_updates'], 1.)
            assert np.isclose(row['bands_weight'], ramp*summary['bands_weight'])
            assert np.isclose(row['edge_weight'], ramp*summary['edge_weight'])
        traces[arm] = trace
        assert abs(np.mean(independent[arm])-report['mean_dice'][arm]) < 1e-12
    assert report['training']['baseline']['bands_weight'] == 0
    assert report['training']['baseline']['edge_weight'] == 0
    assert report['training']['bands']['edge_weight'] == 0
    assert report['training']['bands']['bands_weight'] == report['training']['bands_edge']['bands_weight'] > 0
    assert report['training']['bands_edge']['edge_weight'] > 0
    assert len({r['adapter_before'] for r in report['training'].values()}) == 1
    assert len({r['frozen_before'] for r in report['training'].values()}) == 1
    assert len({r['adapter_after'] for r in report['training'].values()}) == 3
    for arm in arms[1:]:
        assert [r['lr'] for r in traces[arm]] == [r['lr'] for r in traces[arms[0]]]
    initial_loss_check = check_initial_losses(traces, arms)
    calibration = report['calibration']
    if protocol.get('calibration_source_run'):
        summary = report['training'][arms[0]]
        assert calibration['source_run'] == protocol['calibration_source_run']
        assert calibration['context'] == calibration_context(
            protocol, inventory, summary['adapter_before'], summary['frozen_before'])
        assert json.loads((args.results/'calibration.json').read_text()) == calibration
    assert calibration['evaluation_labels_used'] is False
    assert set(r['case'] for r in calibration['slabs']) == set(protocol['train_cases'])
    for row in calibration['slabs']:
        assert np.isfinite(row['native_sq']) and row['native_sq'] > 0
        assert all(np.isfinite(row[k]) and row[k] >= 0 for k in ('bands_sq','edge_sq'))
    for kind in ('bands', 'edge'):
        assert sum(r[kind+'_sq'] > 0 for r in calibration['slabs']) == calibration[kind]['valid_slabs'] >= 5
    audit = dict(status='passed', checked_nifti_files=checked,
                 mean_dice={a: float(np.mean(v)) for a, v in independent.items()},
                 matched_initialization_schedule_learning_rates=True,
                 initial_loss_check=initial_loss_check,
                 frozen_backbone_unchanged=True, training_constraints_active=True,
                 results_sha256=hashlib.sha256((args.results/'results.json').read_bytes()).hexdigest(),
                 updates_per_arm=protocol['updates'], slice_exposures_per_arm=expected_exposures)
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
