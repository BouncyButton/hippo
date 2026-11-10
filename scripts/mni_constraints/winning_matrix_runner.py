"""Two-GPU matrix supervisor with exact recovery and verified-archive cleanup."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign.common import read,write,digest,install_signals,stopping,REMOTE
from scripts.cross_dataset_campaign.storage import prepare,retire,quota
from scripts.cross_dataset_campaign import controller
from scripts.mni_constraints import campaign,winning_matrix2030 as study


WORKER_BOOTSTRAP = """import runpy,signal,sys
from scripts.cross_dataset_campaign.common import install_signals,stopping
install_signals()
signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGUSR1})
if stopping():sys.exit(75)
module=sys.argv[1]
sys.argv=[module,*sys.argv[2:]]
runpy.run_module(module,run_name='__main__',alter_sys=True)
"""


def start_worker(command, **kwargs):
    """Install shutdown handling before imports; preserve even an immediate signal."""
    index=command.index('-m')
    wrapped=command[:index]+['-c',WORKER_BOOTSTRAP,*command[index+1:]]
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1})
    try:
        return subprocess.Popen(wrapped,**kwargs)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def retire_recovery(stage):
    rec=read(stage/'recovery.json');d=stage/'runs'/rec['id']
    if not (d/'cleanup.json').exists() or read(d/'cleanup.json')['status']!='complete':return
    if (stage/'recovery_resolved.json').exists():return
    assert read(d/'archive_acknowledgment.json')['status']=='verified'
    origin=REMOTE/study.PARENT/'runs'/rec['id'];removed=[]
    assert read(REMOTE/study.PARENT/'control.json')['pause']
    for name,sha in rec['files'].items():
        f=origin/name
        if f.exists():assert digest(f)==sha,name
    for name,sha in rec['files'].items():
        if not name.endswith('.pt'):continue
        f=origin/name
        if f.exists():
            removed.append(dict(path=str(f),sha256=sha,bytes=f.stat().st_size));f.unlink()
    record=dict(status='resolved_after_verified_resume',removed=removed,
        archive=read(d/'archive_acknowledgment.json'),resolved_at=time.time())
    write(stage/'recovery_resolved.json',record);write(origin/'winning_matrix_recovery_resolution.json',record)


def run(stage):
    plan=study.verify(stage);install_signals();job=os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_CONTROL']=str(stage/'control.json')
    os.environ['HIPPO_DEADLINE']=str(time.time()+plan['graceful_stop_seconds'])
    campaign.MODULE=study.MODULE;controller.verify=study.verify
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
    active={};prepared=set();signaled=set()
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopping():return
        controller.submit(stage,continuation=job)
        try:
            while True:
                for device,(task,process,log) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    log.close();del active[device]
                    requested_signal_stop=(code==-signal.SIGUSR1 and stopping() and process.pid in signaled)
                    if (code not in (0,75) and not requested_signal_stop) or (code==75 and not stopping()):
                        raise RuntimeError(dict(task=task['id'],exit_code=code))
                    if not (stage/'runs'/task['id']/'completion.json').exists() and not stopping():raise RuntimeError('Missing completion: '+task['id'])
                active_ids={t['id'] for t,_,_ in active.values()};completed=[];cleaned=[];pending=[]
                for task in plan['tasks']:
                    d=stage/'runs'/task['id']
                    if task['id'] in active_ids:continue
                    if not (d/'completion.json').exists():pending.append(task);continue
                    done=read(d/'completion.json');assert done['status']=='complete' and done['discarded_attempts']==0
                    if task['id'] not in prepared:prepare(d,task);prepared.add(task['id'])
                    completed.append(task['id'])
                    if (d/'archive_acknowledgment.json').exists():
                        if not (d/'cleanup.json').exists() or read(d/'cleanup.json')['status']!='complete':retire(d)
                    if (d/'cleanup.json').exists() and read(d/'cleanup.json')['status']=='complete':cleaned.append(task['id'])
                retire_recovery(stage)
                usage=quota();low_space=usage['free_gib']<plan['storage']['minimum_free_gib']
                if stopping():
                    for _,process,_ in active.values():
                        if process.pid not in signaled:process.send_signal(signal.SIGUSR1);signaled.add(process.pid)
                for device in devices:
                    if device in active or not pending or stopping() or low_space:continue
                    task=pending.pop(0);d=stage/'runs'/task['id'];d.mkdir(parents=True,exist_ok=True)
                    log=(d/'train.log').open('a')
                    process=start_worker(campaign.command_for(stage,plan,task),cwd=stage/'source',
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=log,stderr=subprocess.STDOUT)
                    active[device]=(task,process,log)
                finished=len(completed)==len(plan['tasks']) and not active
                paused=read(stage/'control.json').get('pause',False)
                status=('complete' if len(cleaned)==len(completed) else 'complete_pending_archive') if finished else (
                    'paused' if paused and not active else 'continuing_at_time_limit' if stopping() and not active
                    else 'waiting_for_archive_offline' if low_space and not active else 'running_offline')
                write(stage/'status.json',dict(status=status,active=[t['id'] for t,_,_ in active.values()],
                    training_completed=len(completed),completed=len(cleaned),total=plan['num_runs'],
                    logical_completed=len(completed)+plan['reused_runs'],logical_total=504,reused=plan['reused_runs'],
                    pending_durable_archive=[n for n in completed if not (stage/'runs'/n/'archive_acknowledgment.json').exists()],
                    quota=usage,job_id=job,updated_at=time.time()))
                if finished or (not active and (stopping() or low_space)):
                    if finished or paused or low_space:controller.cancel_successor(stage,job)
                    return
                time.sleep(5)
        except Exception:
            for _,process,_ in active.values():
                if process.poll() is None and process.pid not in signaled:process.send_signal(signal.SIGUSR1)
            for _,process,log in active.values():process.wait();log.close()
            controller.cancel_successor(stage,job)
            failure=dict(time=time.time(),traceback=traceback.format_exc())
            write(stage/'failure.json',failure)
            previous=read(stage/'status.json') if (stage/'status.json').exists() else {}
            write(stage/'status.json',dict(previous,status='failed',active=[],job_id=job,
                updated_at=time.time(),failure=failure))
            raise
