"""Laptop-independent continuation; retain artifacts until durable archive returns.

Operational supervisor only. Training commands use the frozen study module.
No synthetic archive acknowledgment or cleanup receipt is ever created.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.cross_dataset_campaign.common import read, write, install_signals, stopping
from scripts.cross_dataset_campaign.storage import prepare, retire, quota
from scripts.mni_constraints import unet_delayed10_confirm as study
from scripts.mni_constraints import campaign


def run(stage):
    plan = study.verify(stage)
    install_signals()
    os.environ['HIPPO_CONTROL'] = str(stage/'control.json')
    os.environ['HIPPO_DEADLINE'] = str(time.time()+plan['graceful_stop_seconds'])
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices) == 2
    assert quota()['free_gib'] >= plan['storage']['minimum_free_gib']
    campaign.MODULE = study.MODULE
    active = {}
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Never override a later user pause or change the fixed training recipe.
        if stopping():
            return
        pending = [t for t in plan['tasks'] if not (stage/'runs'/t['id']/'completion.json').exists()]
        try:
            while True:
                completed, archived, cleaned = [], [], []
                for task in plan['tasks']:
                    folder = stage/'runs'/task['id']
                    if any(t['id'] == task['id'] and process.poll() is None for t, process, _ in active.values()):
                        continue
                    if not (folder/'completion.json').exists():
                        continue
                    done = read(folder/'completion.json')
                    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
                    if not (folder/'archive_manifest.json').exists():
                        prepare(folder, task)
                    completed.append(task['id'])
                    if (folder/'archive_acknowledgment.json').exists():
                        archived.append(task['id'])
                        if not (folder/'cleanup.json').exists():
                            retire(folder)
                    if (folder/'cleanup.json').exists() and read(folder/'cleanup.json')['status'] == 'complete':
                        cleaned.append(task['id'])
                for device, (task, process, log) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close();del active[device]
                    if code != 0:
                        raise RuntimeError(dict(task=task['id'], exit_code=code))
                    if not (stage/'runs'/task['id']/'completion.json').exists() and not stopping():
                        raise RuntimeError('Trainer exited without completion: '+task['id'])
                for device in devices:
                    if device in active or not pending or stopping():
                        continue
                    assert quota()['free_gib'] >= plan['storage']['minimum_free_gib']
                    task = pending.pop(0);folder = stage/'runs'/task['id'];folder.mkdir(parents=True, exist_ok=True)
                    log = (folder/'train.log').open('a')
                    process = subprocess.Popen(campaign.command_for(stage, plan, task), cwd=stage/'source',
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': device}, stdout=log, stderr=subprocess.STDOUT)
                    active[device] = (task, process, log)
                finished = len(completed) == len(plan['tasks']) and not active
                status = ('complete' if len(cleaned) == len(plan['tasks']) else 'complete_pending_archive') if finished else 'running_offline'
                record = dict(status=status, active=[t['id'] for t, _, _ in active.values()],
                    completed=len(cleaned), training_completed=len(completed), total=len(plan['tasks']),
                    pending_durable_archive=[n for n in completed if n not in archived],
                    pending_cleanup=[n for n in completed if n not in cleaned],
                    job_id=os.environ['SLURM_JOB_ID'], updated_at=time.time(),
                    offline_policy='Retain completed metadata/checkpoints until laptop reconnects and durable archive is verified; no fake acknowledgments.')
                write(stage/'status.json', record)
                write(stage/'offline_status.json', record)
                if finished:
                    return
                if stopping():
                    for _, process, _ in active.values():
                        if process.poll() is None:
                            process.send_signal(signal.SIGUSR1)
                    if not active:
                        write(stage/'offline_stopped.json', record)
                        return
                time.sleep(3)
        except Exception:
            for _, process, _ in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGUSR1)
            for _, process, log in active.values():
                process.wait();log.close()
            write(stage/'offline_failure.json', dict(traceback=traceback.format_exc(), time=time.time()))
            raise


if __name__ == '__main__':
    run(Path(sys.argv[1]))
