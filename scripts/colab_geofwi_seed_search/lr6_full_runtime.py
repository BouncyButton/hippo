"""One LR6 baseline, followed by corrected BANDS only if the baseline improves."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time

BASELINE = 'lr6'
BANDS = 'lr6_frame_bands'
REFERENCE_BASELINE = .5061980031645187
REFERENCE_BANDS = .5203030753393142


def qualifies(result):
    """Compare final-budget Dice against the exact archived baseline score."""
    if not result:
        return False
    score = result['final_dice']
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid final Dice')
    return score > REFERENCE_BASELINE


def references(root, campaign):
    io = campaign.io
    rows = {}
    for name, expected in [('baseline', REFERENCE_BASELINE), ('corrected_bands', REFERENCE_BANDS)]:
        folder = root / ('reference_lr5_' + name)
        manifest = io.read(folder / 'archive_manifest.json')
        for filename, info in manifest['metadata'].items():
            f = folder / filename
            if io.digest(f) != info['sha256'] or f.stat().st_size != info['bytes']:
                raise ValueError('Archived reference changed: ' + filename)
        row = campaign.engine.validate_output(root, folder, manifest['task'])
        if abs(row['final_dice'] - expected) > 1e-12:
            raise ValueError('Reference score differs')
        rows[name] = row
    return rows


def compare(io, reference_folder, result_folder):
    previous = io.read(reference_folder / 'validation_latest.json')['cases']
    current = {r['case_name']: r for r in io.read(result_folder / 'validation_latest.json')['cases']}
    if {r['case_name'] for r in previous} != set(current):
        raise ValueError('Comparison cases differ')
    rows = []
    for old in previous:
        new = current[old['case_name']]
        a, b = old['confusion'], new['confusion']
        if sum(a[1]) != sum(b[1]):
            raise ValueError('Ground truth differs')
        rows.append(dict(case_name=old['case_name'], reference_dice=old['macro_hard_dice'],
            result_dice=new['macro_hard_dice'], delta=new['macro_hard_dice']-old['macro_hard_dice'],
            reference_fp=a[0][1], result_fp=b[0][1], reference_fn=a[1][0], result_fn=b[1][0]))
    return dict(delta_percentage_points=100*sum(r['delta'] for r in rows)/len(rows), cases=rows)


def run(root, durable, hours):
    sys.path.insert(0, str(root))
    import lr3_runtime as campaign
    io = campaign.io
    root = io.activate(root)
    campaign.verify(root)
    refs = references(root, campaign)
    durable.mkdir(parents=True, exist_ok=True)
    # Same lock used by preceding experiments on this Colab runtime.
    lock_path = root.parent / 'hippo_geofwi_unetr_lr3_bands_v1/work/augmentation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.time() + hours * 3600
        authorization = dict(instruction='Test LR 6e-4 baseline; conditional matched corrected BANDS',
            only_training_change='Peak learning rate 5e-4 to 6e-4; all other training choices fixed',
            peak_lr=6e-4, halfway_gate=False, frame_width=2, cap_flops_per_arm=campaign.CAP,
            maximum_new_optimization_flops=campaign.CAP*2, maximum_new_runs=2,
            bands_trigger=dict(metric='final_budget_validation_dice', comparison='strictly_greater', threshold=REFERENCE_BASELINE),
            references=refs, driver_sha256=io.digest(__file__),
            diagnostic_preflight='Three separately reported updates per launched arm, established convention')
        io.write(durable/'lr6_authorization.json', authorization)
        runs = {}
        for arm, expected_updates in [(BASELINE, 414), (BANDS, 407)]:
            if arm == BANDS and not qualifies(runs.get(BASELINE)):
                break
            total = campaign.settings(root, arm)[2]
            if total != expected_updates:
                raise ValueError('Capped schedule changed')
            result = campaign.final_result(root, durable, arm)
            if result is None:
                campaign.preflight(root, durable, arm)
                out = root/'work'/campaign.task(arm)['id']
                out.mkdir(parents=True, exist_ok=True)
                io.write(out/'experiment_authorization.json', authorization)
                io.write(durable/'lr6_status.json', dict(status='running', active_arm=arm,
                    target_updates=total, completed_runs=runs, **authorization))
                print('RUN', arm, 'through', total, 'total updates; LR 6e-4; no halfway gate', flush=True)
                if not campaign.execute(root, durable, arm, total, deadline):
                    io.write(durable/'lr6_status.json', dict(status='paused', active_arm=arm,
                        target_updates=total, completed_runs=runs, **authorization))
                    return
                result = campaign.final_result(root, durable, arm)
            if not result:
                raise RuntimeError('Full-budget archive not verified')
            runs[arm] = result
        baseline, bands = runs[BASELINE], runs.get(BANDS)
        baseline_folder = durable/'runs'/campaign.task(BASELINE)['id']
        comparisons = dict(baseline_vs_lr5=compare(io, root/'reference_lr5_baseline', baseline_folder))
        if bands:
            bands_folder = durable/'runs'/campaign.task(BANDS)['id']
            comparisons['corrected_bands_vs_lr5'] = compare(io, root/'reference_lr5_corrected_bands', bands_folder)
            comparisons['paired_bands_vs_baseline'] = compare(io, baseline_folder, bands_folder)
        io.write(durable/'lr6_comparisons.json', comparisons)
        status = dict(status='complete', baseline=baseline, corrected_bands=bands,
            bands_status='complete' if bands else 'not_triggered', baseline_improved=qualifies(baseline),
            corrected_bands_improved=bool(bands and bands['final_dice'] > REFERENCE_BANDS),
            deltas_percentage_points={name: report['delta_percentage_points'] for name, report in comparisons.items()},
            references=refs, cap_flops_per_arm=campaign.CAP, halfway_gate=False,
            scope='One fold/seed development experiment; no further runs are launched')
        io.write(durable/'lr6_status.json', status)
        io.write(durable/'LR6_FINISHED.json', status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=2)
    args = parser.parse_args()
    if not 0 < args.hours <= 2:
        parser.error('Session limit must be in (0,2] hours')
    run(args.bundle.resolve(), args.durable.resolve(), args.hours)
