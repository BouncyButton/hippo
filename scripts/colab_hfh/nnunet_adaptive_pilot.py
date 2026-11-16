"""Three-seed adaptive BAND pilot, gated expansion, then resume the other models."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scripts.colab_hfh.bands_campaign import read, write
from scripts.colab_hfh.constraint_drive import Drive
from scripts.colab_hfh.constraint_mean_search import complete, qualifies, summary, POLICY
from scripts.colab_hfh.constraint_report import write_report
from scripts.colab_hfh.constraint_schedule import make_payload
from scripts.colab_hfh.constraint_search import recipes

NAME = 'nnunet__dice_bands__r05'
SPEC = dict(prefix=22, ramp=9, target=.1, cap=.5, adaptive_sides=True)


def pilot_summary(rows):
    if len(rows) != 3 or {(r['fold'], r['seed']) for r in rows} != {(0, s) for s in range(3)}:
        raise ValueError('Pilot requires exactly fold 0, seeds 0, 1, 2')
    gain = statistics.mean(r['delta_dice_pp'] for r in rows)
    return dict(mean_delta_pp=gain, worst_delta_pp=min(r['delta_dice_pp'] for r in rows),
        expand=gain >= 1. and all(r['constraint_dice'] >= .5 for r in rows),
        fp_change=sum(r['constraint_foreground_fp']-r['baseline_foreground_fp'] for r in rows),
        fn_change=sum(r['constraint_foreground_fn']-r['baseline_foreground_fn'] for r in rows))


def run_pilot(root, deadline):
    plan, base = read(root/'constraint_matrix_plan.json'), read(root/'bands_v1_payload.json')
    for name, expected in plan['calibration_implementation']['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Installed source differs: '+name)
    assert recipes('nnunet', plan, 'dice_bands')[5] == SPEC
    work = root/'constraint_search'
    path = work/'search_status.json'
    state = read(path)
    record = state['attempts'][NAME]
    if record['specification'] != SPEC:
        raise ValueError('Existing adaptive candidate changed')
    write(work/('before_adaptive_pilot_'+str(time.time_ns())+'.json'), state)
    drive = Drive('results_constraint_search_v1')
    pilot_path = work/'nnunet_adaptive_band_pilot.json'
    status = dict(status='running_fold0', recipe=NAME, specification=SPEC,
        expansion_gate='Fold0 three-seed mean >=1 Dice point and every Dice >=50%',
        budget_flops=plan['budget_flops_per_run'], results=[],
        interpretation='Exploratory validation-selected pilot; all three results retained')

    def publish():
        write(pilot_path, status); drive.upload(pilot_path)
        write(path, state); drive.upload(path)

    trial = work/NAME
    (trial/'STOP').unlink(missing_ok=True)
    for folds in ([0], [0, 2, 1]):
        if (work/'STOP').exists() or time.time() >= deadline:
            status['status'] = 'paused'; publish(); return False
        payload = make_payload(base, 'nnunet', 'dice_bands',
            plan['cost_references']['nnunet']['canonical_costs'], **SPEC, screening=False)
        payload['folds'] = folds
        payload_path = work/(NAME+'_payload.json')
        write(payload_path, payload)
        state.update(status='running', phase='adaptive_band_staged_pilot', current=NAME,
                     model_scope=['nnunet'])
        record['status'] = 'running'
        publish()
        print('ADAPTIVE BAND STAGE', folds, 'seeds', payload['seeds'], flush=True)
        subprocess.run([sys.executable, '-m', 'scripts.colab_hfh.constraint_campaign',
            '--root', str(root), '--payload', str(payload_path), '--work', str(trial),
            '--folder', 'results_constraints_'+NAME,
            '--hours', str(max(.01, (deadline-time.time())/3600))], check=True)
        result = read(trial/'campaign_status.json')
        rows = result.get('results', [])
        record['results'] = rows
        status['results'] = rows
        if result['status'] != 'complete_requested_matrix':
            status['status'] = 'paused'; publish(); return False
        if len(folds) == 1:
            status['pilot'] = pilot_summary(rows)
            print('ADAPTIVE FOLD0 RESULT', status['pilot'], flush=True)
            record['fold0_pilot'] = status['pilot']
            if not status['pilot']['expand']:
                record['status'] = 'pilot_below_expansion_target'
                status['status'] = 'pilot_no_expansion'
                publish(); break
            status['status'] = 'expanding_remaining_folds'; publish()
        else:
            if not complete(rows):
                raise ValueError('Expanded campaign lacks nine fixed pairs')
            record.update(full_matrix_complete=True, mean_evaluation=summary(rows),
                status='qualified_mean' if qualifies(rows) else 'below_mean_target')
            status.update(status='complete_nine', summary=summary(rows))
            if qualifies(rows):
                state['selected']['nnunet__dice_bands'] = dict(candidate=5,
                    specification=SPEC, recipe=NAME, qualification='mean_target_met_all_nine',
                    frozen_under_policy=POLICY['version'], **summary(rows))
            publish()
    state.update(status='adaptive_pilot_complete'); state.pop('current', None); publish()
    _, matrix, report = write_report(root)
    for item in (matrix, report):
        drive.upload_report(item)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    deadline = time.time()+args.hours*3600
    with (args.root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finished = run_pilot(args.root, deadline)
    if finished and time.time() < deadline and not (args.root/'constraint_search/STOP').exists():
        print('RESUMING THREE-MODEL MATRIX', flush=True)
        subprocess.run([sys.executable, '-m', 'scripts.colab_hfh.complete_selected_matrix',
            '--root', str(args.root), '--hours', str((deadline-time.time())/3600)], check=True)


if __name__ == '__main__':
    main()
