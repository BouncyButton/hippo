"""Two-worker milestone scheduler, with per-milestone durable archive gates."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.cross_dataset_campaign import controller, adni_foundation_full_pool as pool
from scripts.cross_dataset_campaign.common import read, write, digest, stopping
from scripts.cross_dataset_campaign.storage import prepare, quota, verify_archive
from scripts.mni_medsam3_cluster import campaign as base
from scripts.mni_medsam3_cluster.adaptive_policy import choose_seeds, next_trials, needs_checkpoint


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    plan = read(stage/'plan.json')
    if (plan['threshold'], plan['maximum_updates'], plan['seeds'], plan['parallelism']['workers']) != (.60, 1000, [0, 1, 2], 2):
        raise ValueError('Adaptive policy differs from user authorization')
    expected = {(f,s,u) for f in (0,1,2) for s in (0,1,2) for u in range(400,1001,100)}
    if {(t['fold'], t['seed'], t['updates']) for t in plan['tasks']} != expected or len(plan['tasks']) != len(expected):
        raise ValueError('Incomplete milestone inventory')
    for t in plan['tasks']:
        p = read(stage/'protocols'/t['protocol']/'protocol.json')
        if (p['fold'],p['seed'],p['updates'],p['train_subjects'],p['train_crops']) != (t['fold'],t['seed'],t['updates'],5,10):
            raise ValueError('Protocol differs from task')
        if p['augmentation'] != 'none' or p['requested_arms'] != ['dice']:
            raise ValueError('Loss/augmentation changed')
        if {c.rsplit('_',1)[0] for c in p['train_cases']} & {c.rsplit('_',1)[0] for c in p['evaluation_cases']}:
            raise ValueError('Subject leakage')
    return plan


def records(stage, plan):
    rows = []
    for t in plan['tasks']:
        d = stage/'runs'/t['id']
        if not (d/'archive_acknowledgment.json').exists():
            continue
        done = read(d/'completion.json')
        rows.append(dict(**t, validation_dice=done['validation_dice'],
            training_dice=done['training_dice'], observed_at=read(d/'milestone_observed.json')['time']))
    return rows


def selected_seeds(stage, rows, plan):
    path = stage/'selected_seeds.json'
    if path.exists():
        return read(path)['seeds']
    selected = choose_seeds(rows, plan['threshold'])
    if selected:
        write(path, dict(seeds=selected, threshold=plan['threshold'], selected_at=time.time(),
            method='First two seeds whose archived fold0 milestone reaches60%; no score-based ranking.',
            evidence=[r for r in rows if r['fold']==0 and r['seed'] in selected and r['validation_dice']>=plan['threshold']]))
    return selected


def summarize(stage, plan, status, active=None):
    rows = records(stage, plan)
    selected = selected_seeds(stage, rows, plan)
    outcome = None
    if status == 'complete':
        if not selected:
            outcome = 'fewer_than_two_seeds_reached60_at_or_below1000'
        else:
            passed = {(r['fold'],r['seed']) for r in rows if r['validation_dice']>=plan['threshold']}
            outcome = ('two_seeds_qualified_on_all_three_folds' if all((f,s) in passed for f in (1,2) for s in selected)
                       else 'some_validation_folds_below60_at1000')
    write(stage/'status.json', dict(status=status, active=active or [], completed_milestones=len(rows),
        selected_seeds=selected, outcome=outcome, threshold=plan['threshold'], maximum_updates=1000,
        runs=rows, job_id=os.environ.get('SLURM_JOB_ID'), updated_at=time.time(), quota=quota()))


def parent_task(plan, task):
    if task['updates'] == 400:
        return None
    return next(t for t in plan['tasks'] if (t['fold'],t['seed'],t['updates']) == (task['fold'],task['seed'],task['updates']-100))


def command(stage, plan, task, base_path, lane):
    result = base.worker_command(stage, plan, task, base_path, lane)
    result[result.index('scripts.mni_medsam3_cluster.worker')] = 'scripts.mni_medsam3_cluster.adaptive_worker'
    parent = parent_task(plan, task)
    if parent:
        result += ['--resume-parent', str(stage/'runs'/parent['id'])]
    return result


def retire_ready(stage, plan, rows, selected, active_ids):
    transferred = set()
    for t in plan['tasks']:
        p = parent_task(plan, t)
        proof = stage/'runs'/t['id']/'resume_provenance.json'
        if p and proof.exists():
            rec = read(proof); parent = stage/'runs'/p['id']
            if rec['status'] != 'verified' or rec['parent_path'] != str(parent) or rec['parent_checkpoint_sha256'] != read(parent/'completion.json')['checkpoint_sha256']:
                raise ValueError('Invalid continuation dependency receipt')
            transferred.add(p['id'])
    for row in rows:
        output = stage/'runs'/row['id']
        if (output/'cleanup.json').exists():
            continue
        if row['id'] not in transferred and any(t['id'] in active_ids and
                (t['fold'],t['seed'],t['updates']) == (row['fold'],row['seed'],row['updates']+100)
                for t in plan['tasks']):
            continue
        if needs_checkpoint(row, plan, selected, transferred):
            continue
        base.retire(output)


def dispatch(stage, plan, base_path, publish):
    workers = {}
    reason = None
    failure = None
    try:
        while True:
            changed = False
            if stopping() and reason is None:
                reason = 'paused' if read(stage/'control.json').get('pause') else 'continuing_at_time_limit'
            if reason:
                pool.signal_workers(workers)
            for lane, w in list(workers.items()):
                out = w['output']
                if w['phase'] == 'running':
                    code = w['process'].poll()
                    if code is None:
                        continue
                    w['stream'].close()
                    if code:
                        marker = out/'requested_stop.json'
                        if reason and marker.exists() and read(marker)['checkpoint_sha256'] == digest(out/'checkpoint_latest.pt'):
                            del workers[lane]; changed=True; continue
                        failure = f"{w['task']['id']}: worker exited{code}"
                        reason = 'failed'; del workers[lane]; changed=True; continue
                    if not (out/'completion.json').exists():
                        raise RuntimeError('Worker returned without a completed evaluation')
                    if not (out/'milestone_observed.json').exists():
                        write(out/'milestone_observed.json', dict(time=time.time()))
                    prepare(out, w['task'])
                    w.update(phase='waiting_for_verified_archive', archive_started=time.monotonic())
                    changed=True
                if w['phase'] == 'waiting_for_verified_archive':
                    if (out/'archive_acknowledgment.json').exists():
                        ack=read(out/'archive_acknowledgment.json')
                        if ack['status']!='verified' or ack['manifest_sha256']!=digest(out/'archive_manifest.json') or not ack['source_verified']:
                            raise ValueError('Archive acknowledgment mismatch')
                        verify_archive(out, read(out/'archive_manifest.json'))
                        del workers[lane]; changed=True
                    elif reason or time.monotonic()-w['archive_started'] > plan['archive_wait_timeout_seconds']:
                        reason=reason or 'waiting_for_archive_offline'
                        del workers[lane]; changed=True
            rows=records(stage,plan); selected=selected_seeds(stage,rows,plan)
            retire_ready(stage,plan,rows,selected,{w['task']['id'] for w in workers.values()})
            if reason:
                pool.signal_workers(workers)
                if not workers:
                    publish(reason,{})
                    if failure:raise RuntimeError(failure)
                    return reason
            else:
                for lane in range(2):
                    if lane in workers:continue
                    active={w['task']['id'] for w in workers.values()}
                    ready=next_trials(plan,rows,selected,active)
                    if not ready:continue
                    task=ready[0]; out=stage/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
                    if quota()['free_gib'] < plan['storage']['minimum_free_gib']:
                        raise RuntimeError('Insufficient personal quota for next milestone')
                    w=dict(task=task,output=out)
                    if (out/'completion.json').exists():
                        if not (out/'milestone_observed.json').exists():write(out/'milestone_observed.json',dict(time=time.time()))
                        prepare(out,task)
                        w.update(phase='waiting_for_verified_archive',archive_started=time.monotonic())
                    else:
                        # A previous allocation's stop marker cannot validate a new failure.
                        (out/'requested_stop.json').unlink(missing_ok=True)
                        stream=(out/'train.log').open('a')
                        try:process=subprocess.Popen(command(stage,plan,task,base_path,lane),cwd=stage/'source',stdout=stream,stderr=subprocess.STDOUT)
                        except Exception:stream.close();raise
                        w.update(phase='running',process=process,stream=stream)
                    workers[lane]=w;changed=True
                if not workers:
                    publish('complete',{})
                    return 'complete'
            if changed:publish(reason or 'running',workers)
            time.sleep(2)
    finally:
        pool.signal_workers(workers)
        for w in workers.values():
            process=w.get('process')
            if process and process.poll() is None:
                try:process.wait(timeout=300)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
            if w.get('stream'):w['stream'].close()


def run(stage):
    plan=verify(stage)
    controller.summarize=summarize
    pool.dispatch=dispatch
    pool.gpu_preflight=lambda stage,plan,base,publish:True
    pool.run(stage,plan,lambda:None,lambda:None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['verify','preflight','submit','admit','run'])
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args();stage=args.stage.resolve()
    base.verify=verify
    if args.command=='run':run(stage)
    elif args.command=='verify':verify(stage)
    else:getattr(base,args.command)(stage)


if __name__=='__main__':main()
