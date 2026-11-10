"""Two-worker cluster supervisor with an absolute deadline and no laptop dependency."""
import fcntl
import os
import signal
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign.common import read, write, install_signals, stopping
from scripts.cross_dataset_campaign.storage import prepare, retire, quota
from scripts.mni_constraints import campaign
from scripts.mni_constraints import nnunet_swin_refine3h as study


def admit(plan, now):
    return now+plan['admission_reserve_seconds'] < plan['deadline']


def run(stage):
    plan = read(stage/'plan.json')
    if time.time() >= plan['deadline']-plan['stop_buffer_seconds']:
        write(stage/'status.json', dict(status='deadline_reached', active=[], training_completed=0,
            deadline=plan['deadline'], updated_at=time.time(), job_id=os.environ['SLURM_JOB_ID']))
        return
    study.verify(stage)
    install_signals()
    os.environ['HIPPO_CONTROL'] = str(stage/'control.json')
    os.environ['HIPPO_DEADLINE'] = str(plan['deadline']-plan['stop_buffer_seconds'])
    campaign.MODULE = study.MODULE
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices) == 2
    active, prepared, signaled = {}, set(), set()
    reason = None
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                for device, (task, process, log) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close();del active[device]
                    if code != 0 and not stopping():
                        raise RuntimeError(dict(task=task['id'], exit_code=code))
                    if not (stage/'runs'/task['id']/'completion.json').exists() and not stopping():
                        raise RuntimeError('Missing completion: '+task['id'])
                active_ids = {t['id'] for t,_,_ in active.values()}
                for task in plan['tasks']:
                    d = stage/'runs'/task['id']
                    if task['id'] in active_ids or not (d/'completion.json').exists():
                        continue
                    if task['id'] not in prepared:
                        done = read(d/'completion.json')
                        assert done['status'] == 'complete' and done['discarded_attempts'] == 0
                        prepare(d, task);prepared.add(task['id'])
                    if (d/'archive_acknowledgment.json').exists() and (not (d/'cleanup.json').exists() or read(d/'cleanup.json')['status'] != 'complete'):
                        retire(d)
                rows = study.collect_rows(stage);state = study.select(rows)
                write(stage/'refinement_results.json', dict(rows=rows, selection=state['models'],
                    complete=state['finished'], updated_at=time.time(), deadline=plan['deadline']))
                if not admit(plan, time.time()):
                    reason = 'deadline_admission_cutoff'
                if stopping():
                    reason = 'paused' if read(stage/'control.json').get('pause') else 'deadline_reached'
                    for _, process, _ in active.values():
                        if process.pid not in signaled:
                            process.send_signal(signal.SIGUSR1);signaled.add(process.pid)
                pending = [t for t in state['pending'] if t['id'] not in active_ids]
                for device in devices:
                    if device in active or not pending or reason:
                        continue
                    usage = quota()
                    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
                        reason = 'storage_headroom_limit';break
                    task = pending.pop(0);d = stage/'runs'/task['id'];d.mkdir(parents=True, exist_ok=True)
                    log = (d/'train.log').open('a')
                    process = subprocess.Popen(campaign.command_for(stage, plan, task), cwd=stage/'source',
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':device}, stdout=log, stderr=subprocess.STDOUT)
                    active[device] = (task, process, log)
                complete_ids = [r['id'] for r in rows]
                cleaned = [n for n in complete_ids if (stage/'runs'/n/'cleanup.json').exists()
                           and read(stage/'runs'/n/'cleanup.json')['status'] == 'complete']
                pending_archive = [n for n in complete_ids if not (stage/'runs'/n/'archive_acknowledgment.json').exists()]
                finished = not active and (state['finished'] or reason is not None)
                status = ('complete' if len(cleaned) == len(rows) else 'complete_pending_archive') if finished and state['finished'] else reason if finished else 'running_offline'
                record = dict(status=status, active=[t['id'] for t,_,_ in active.values()],
                    completed=len(cleaned), training_completed=len(rows), total=plan['num_runs'],
                    maximum_possible_runs=plan['num_runs'], screen_runs=plan['screen_runs'],
                    selection=state['models'], pending_durable_archive=pending_archive,
                    pending_cleanup=[n for n in complete_ids if n not in cleaned],
                    deadline=plan['deadline'], updated_at=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                    reason=reason, broad_sweep='paused; never automatically modified or resumed')
                write(stage/'status.json', record)
                if finished:
                    write(stage/'bounded_completion.json', record)
                    return
                time.sleep(5)
        except Exception:
            for _, process, _ in active.values():
                if process.poll() is None and process.pid not in signaled:
                    process.send_signal(signal.SIGUSR1)
            for _, process, log in active.values():
                process.wait();log.close()
            write(stage/'failure.json', dict(traceback=traceback.format_exc(), time=time.time()))
            raise
