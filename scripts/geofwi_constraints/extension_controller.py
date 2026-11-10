"""Frozen confirmation and exploratory strength queues on two isolated GPUs."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import parent_controller as parent

runtime = parent.runtime
CAP, read, write, sha = parent.CAP, parent.read, parent.write, parent.sha
key = parent.key
validate_run = parent.validate_run
ClusterArchive = parent.ClusterArchive
submit, run = parent.submit, parent.run


def verify(stage, data=True):
    for name, digest in read(stage / 'manifest.json').items():
        assert sha(stage / name) == digest, name
    plan, design = read(stage / 'plan.json'), read(stage / 'design.json')
    expected = set()
    for model, d in design.items():
        expected |= {(model, f, s, d['arm'], d['confirmation']) for f in range(3) for s in range(3)}
        expected |= {(model, 0, s, d['arm'], v) for v in d['exploration'] for s in range(3)}
    assert len(plan['tasks']) == 27 and len(plan['reused_rows']) == 6
    assert len(expected) == 33 and {key(t) for t in plan['tasks'] + plan['reused_rows']} == expected
    assert len({t['id'] for t in plan['tasks']}) == 27
    assert plan['cap'] == CAP and plan['workers'] == 2 and plan['baseline_runs'] == 0
    assert len(read(stage / 'baseline_results.json')['rows']) == 36
    splits = read(stage / 'splits.json')
    assert len(splits) == 3 and len(set().union(*(set(s['val']) for s in splits))) == 288
    for split in splits:
        assert len(set(split['train'])) == 19 and len(set(split['val'])) == 96
        assert not set(split['train']) & set(split['val'])
    by_id = {t['id']: t for t in plan['tasks']}
    scheduled_ids = [identity for lane in plan['lane_task_ids'] for identity in lane]
    assert len(scheduled_ids) == len(set(scheduled_ids)) == 27 and set(scheduled_ids) == set(by_id)
    assert len(plan['lane_task_ids']) == 2
    for lane, model in enumerate(design):
        queue = [by_id[identity] for identity in plan['lane_task_ids'][lane]]
        assert {t['model'] for t in queue} == {model}
        assert [t['role'] for t in queue] == ['confirmation'] * 6 + ['exploration'] * (3 * len(design[model]['exploration']))
    for t in plan['tasks']:
        folder = stage / 'protocols' / t['id']
        p = read(folder / 'protocol.json')
        d = design[t['model']]
        assert p['objective'] == t['arm'] == d['arm'] != 'dice'
        assert [p['calibration_target_ratio'], p['calibration_p95_cap']] == d['variants'][t['strength']]
        assert p['strength'] == t['strength'] and p['study_role'] == t['role']
        assert (t['role'] == 'confirmation') == (t['fold'] in (1, 2))
        for filename, field in [('dataset.json', 'pkl_sha256'), ('splits_10patients.json', 'splits_sha256'),
                                ('nnunet_architecture.json', 'nnunet_architecture_sha256'), ('profile.json', 'profile_sha256')]:
            assert sha(folder / filename) == p[field]
        assert p['splits_sha256'] == sha(stage / 'splits.json')
        assert p['model_identity'] == t['model'] and p['seeds'] == [t['seed']] and p['folds'] == [t['fold']]
        assert p['budget_flops'] == CAP and p['batch_size'] == 1 and not p['early_stopping']
        sched = p['models'][p['trainer_alias']]['objectives'][t['arm']]
        assert sched['consumed_total_flops'] <= CAP < sched['consumed_total_flops'] + sched['flops_per_update']
        assert sched['consumed_total_flops'] == sched['consumed_training_flops'] + sched['calibration_flops']
        assert sched['max_steps'] == t['updates'] and sched['calibration_flops'] > 0
        assert sched['prefix_updates'] + sched['constraint_warmup_updates'] < sched['max_steps']
        baseline = p['baseline_reference']
        assert (baseline['model'], baseline['fold'], baseline['seed']) == (t['model'], t['fold'], t['seed'])
    for row in plan['reused_rows']:
        assert row['fold'] == 0 and row['role'] == 'confirmation' and row['total_flops'] <= CAP
        proof = read(stage / 'reuse_verification.json')
        assert proof['status'] == 'passed' and row['id'] in {r['id'] for r in proof['rows']}
    if data:
        for name, digest in read(stage / 'data_manifest.json').items():
            assert sha(stage / name) == digest, name
    return plan


def summarize(stage, status):
    plan = read(stage / 'plan.json')
    rows = list(plan['reused_rows'])
    for t in plan['tasks']:
        out = stage / 'runs' / t['id']
        if (out / 'cleanup.json').exists() and read(out / 'cleanup.json')['status'] == 'complete':
            rows.append(validate_run(stage, t))
    assert len({key(r) for r in rows}) == len(rows)
    baselines = {(r['model'], r['fold'], r['seed']): r for r in read(stage / 'baseline_results.json')['rows']}
    controls = {r['id']: r for r in read(stage / 'controls.json')['rows']}
    for r in rows:
        assert math.isfinite(r['final_dice'])
        r['baseline_dice'] = baselines[key(r)[:3]]['final_dice']
        r['delta_dice_percentage_points'] = 100 * (r['final_dice'] - r['baseline_dice'])
        if r['role'] == 'exploration':
            r['control_final_dice'] = controls[r['control_id']]['final_dice']
            r['delta_vs_selected_constraints_pp'] = 100 * (r['final_dice'] - r['control_final_dice'])
    confirmation = [r for r in rows if r['role'] == 'confirmation']
    exploration = [r for r in rows if r['role'] == 'exploration']
    complete = len(rows) == 33
    doc = dict(status='complete' if complete else status, rows=rows, required_cells=33,
               complete_matrix=complete, confirmation_complete=len(confirmation) == 18,
               confirmation_rows=confirmation, exploration_rows=exploration,
               all_above_50=complete and all(r['final_dice'] > .5 for r in rows),
               new_runs=27, baseline_runs=0, baselines_reused=18, laptop_required=False, updated_at=time.time())
    write(stage / 'status.json', doc)
    if complete:
        write(stage / 'FINISHED.json', doc)
    return doc


parent.verify = runtime.verify = verify
parent.summarize = runtime.summarize = summarize


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('verify', 'submit', 'run', 'lane', 'status'))
    ap.add_argument('--stage', type=Path, required=True)
    args = ap.parse_args()
    stage = args.stage.resolve()
    try:
        if args.command == 'verify':
            print(json.dumps(dict(status='passed', tasks=len(verify(stage)['tasks']))))
        elif args.command == 'submit':
            print(json.dumps(submit(stage)))
        elif args.command == 'status':
            print(json.dumps(summarize(stage, 'inspected')))
        elif args.command == 'lane':
            sys.exit(runtime.lane(stage))
        else:
            sys.exit(run(stage))
    except Exception:
        write(stage / f"failure_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_PROCID', 'main')}.json",
              dict(status='failed', traceback=traceback.format_exc(), time=time.time()))
        raise
