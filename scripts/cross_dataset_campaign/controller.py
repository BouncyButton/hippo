"""One running allocation, one dependent continuation, bounded by 24 hours."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.cross_dataset_campaign.common import read, write, digest, stopping, install_signals
from scripts.cross_dataset_campaign.storage import quota, prepare, retire


def queue():
    raw=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True)
    return [r.split('|') for r in raw.splitlines() if r.strip()]


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:
            raise ValueError('Frozen experiment changed: '+name)
    plan=read(stage/'plan.json')
    if plan['augmentation']!='none' or len(plan['tasks'])!=50:
        raise ValueError('Wrong exploration plan')
    return plan


def submit(stage, continuation=None):
    plan=verify(stage)
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=queue()
        if len(current)>1 or any(row[1] not in ('RUNNING','COMPLETING','CONFIGURING') for row in current):
            raise RuntimeError('No slot under one-running/one-queued policy: '+repr(current))
        if continuation and (len(current)!=1 or current[0][0]!=str(continuation)):
            raise ValueError('Continuation must depend on this running job')
        usage=quota()
        if usage['free_gib']<plan['storage']['minimum_free_gib']:
            raise RuntimeError('Insufficient personal quota headroom')
        if read(stage/'control.json')['pause']:
            raise RuntimeError('Campaign is paused')
        if not (stage/'preflight.json').exists() or read(stage/'preflight.json')['status']!='passed':
            raise RuntimeError('A verified preflight is required')
        if read(stage/'preflight.json')['manifest_sha256']!=digest(stage/'manifest.json'):
            raise ValueError('Preflight applies to another frozen source')
        dependency=str(continuation) if continuation else current[0][0] if current else None
        command=['sbatch','--parsable','--chdir='+str(stage),'--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err')]
        if dependency:command+=['--dependency=afterany:'+dependency]
        command += [str(stage/'run.sbatch'),str(stage)]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Unexpected sbatch response')
        receipt=dict(job_id=job,dependency=dependency,quota=usage,submitted_at=time.time(),command=command,
                     manifest_sha256=digest(stage/'manifest.json'))
        write(stage/'submissions'/f'{job}.json',receipt)
        if continuation:write(stage/'successor.json',receipt)
        else:write(stage/'submission.json',receipt)
        return receipt


def cancel_successor(stage,job):
    path=stage/'successor.json'
    if path.exists():
        record=read(path)
        if record['dependency']==str(job):
            for row in queue():
                if row[0]==record['job_id'] and row[1]=='PENDING':
                    subprocess.run(['scancel',record['job_id']],check=True)


def command_for(stage,plan,task,base_path):
    folder=stage/'protocols'/task['protocol'];output=stage/'runs'/task['id']
    if task['kind']=='foundation':
        return [plan['foundation_python'],'-m','scripts.cross_dataset_campaign.foundation','--protocol',str(folder/'protocol.json'),
                '--output',str(output),'--arm',task['arm'],'--base-weights',str(base_path)]
    p=read(folder/'protocol.json')
    return [plan['conventional_python'],'-m','scripts.cross_dataset_campaign.conventional',
        '--model',p['trainer_alias'],'--objective',task['arm'],'--fold',str(task['fold']),'--seed',str(task['seed']),
        '--pkl',p['data'],'--protocol-dir',str(folder),'--output',str(output),'--resume']


def summarize(stage,plan,status,active=None):
    rows=[]
    for task in plan['tasks']:
        root=stage/'runs'/task['id']
        if not (root/'cleanup.json').exists() or read(root/'cleanup.json')['status']!='complete':continue
        done=read(root/'completion.json')
        if task['kind']=='conventional':
            value=read(root/'validation_latest.json')['summary']['macro_hard_dice']
            best=done['best_validation_hard_dice']
        else:value=best=done['validation_dice']
        rows.append(dict(**task,final_dice=value,best_dice=best,
            released_bytes=read(root/'cleanup.json')['released_file_bytes']))
    write(stage/'status.json',dict(status=status,active=active,completed=len(rows),total=len(plan['tasks']),runs=rows,
        job_id=os.environ.get('SLURM_JOB_ID'),updated_at=time.time(),quota=quota()))


def run(stage):
    plan=verify(stage);install_signals();job=os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_CONTROL']=str(stage/'control.json')
    os.environ['HIPPO_DEADLINE']=str(min(time.time()+plan['graceful_stop_seconds'],
        float(os.environ.get('HIPPO_ALLOCATION_DEADLINE','inf'))))
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(stage/'control.json')['pause']:
            summarize(stage,plan,'paused');return
        # One successor is queued before useful work so a hard time limit also
        # has a continuation. Failures and manual pause cancel this successor.
        submit(stage,continuation=job)
        base_path=Path('/tmp')/f'hippo-sam3-{os.getuid()}-{job}.pt'
        active=None
        try:
            for task in plan['tasks']:
                active=task['id'];output=stage/'runs'/active
                control=read(stage/'control.json')
                if active in control.get('disabled_tasks',[]):continue
                if (output/'cleanup.json').exists() and read(output/'cleanup.json')['status']=='complete':continue
                if stopping():
                    paused=control.get('pause',False)
                    summarize(stage,plan,'paused' if paused else 'continuing_at_time_limit',active)
                    if paused:cancel_successor(stage,job)
                    return
                if quota()['free_gib']<plan['storage']['minimum_free_gib']:
                    raise RuntimeError('Insufficient quota before next experiment')
                output.mkdir(parents=True,exist_ok=True)
                if not (output/'completion.json').exists():
                    if task['kind']=='foundation' and not base_path.exists():
                        write(stage/'base_request.json',dict(job_id=job,destination=str(base_path),expected_sha256=plan['base_sha256']))
                        summarize(stage,plan,'waiting_for_base_checkpoint',active)
                        base_wait_started=time.monotonic()
                        while not base_path.exists() or not (stage/f'base_ready_{job}.json').exists():
                            if stopping():break
                            if (plan.get('archive_wait_timeout_seconds') and
                                    time.monotonic()-base_wait_started>=plan['archive_wait_timeout_seconds']):
                                cancel_successor(stage,job)
                                summarize(stage,plan,'waiting_for_base_offline',active)
                                return
                            time.sleep(5)
                        if stopping():return
                        if digest(base_path)!=plan['base_sha256']:raise ValueError('Base delivery hash mismatch')
                    summarize(stage,plan,'running',active)
                    with (output/'train.log').open('a') as stream:
                        # Python searches its working directory before PYTHONPATH.
                        # A resumed predecessor must import its own frozen source.
                        process=subprocess.Popen(command_for(stage,plan,task,base_path),cwd=stage/'source',
                            stdout=stream,stderr=subprocess.STDOUT)
                        sent=False
                        while process.poll() is None:
                            if stopping() and not sent:
                                process.send_signal(signal.SIGUSR1);sent=True
                            time.sleep(2)
                        if process.returncode==75:
                            paused=read(stage/'control.json')['pause']
                            summarize(stage,plan,'paused' if paused else 'continuing_at_time_limit',active)
                            if paused:cancel_successor(stage,job)
                            return
                        if process.returncode:
                            raise RuntimeError(f'{active}: trainer exited {process.returncode}')
                prepare(output,task)
                summarize(stage,plan,'waiting_for_verified_archive',active)
                archive_wait_started=time.monotonic()
                while not (output/'archive_acknowledgment.json').exists():
                    if stopping():
                        paused=read(stage/'control.json')['pause']
                        summarize(stage,plan,'paused' if paused else 'continuing_at_time_limit',active)
                        if paused:cancel_successor(stage,job)
                        return
                    archive_timeout=plan.get('archive_wait_timeout_seconds')
                    if archive_timeout and time.monotonic()-archive_wait_started>=archive_timeout:
                        # Keep the completed recovery/evaluation files, release
                        # the GPU, and let the durable collector resume later.
                        cancel_successor(stage,job)
                        summarize(stage,plan,'waiting_for_archive_offline',active)
                        return
                    time.sleep(5)
                retire(output)
                summarize(stage,plan,'running',active)
            summarize(stage,plan,'complete')
            cancel_successor(stage,job)
        except Exception:
            write(stage/'failure.json',dict(active=active,traceback=traceback.format_exc(),time=time.time()))
            summarize(stage,plan,'failed',active)
            cancel_successor(stage,job)
            raise
        finally:
            # This is a generated private scratch copy, never the shared base.
            if base_path.exists() and digest(base_path)==plan['base_sha256']:base_path.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['submit','run','status','pause','resume'])
    parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage.resolve()
    if args.command=='run':run(stage)
    elif args.command=='submit':print(submit(stage))
    elif args.command=='status':print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        control=read(stage/'control.json');control['pause']=args.command=='pause';write(stage/'control.json',control)
        if args.command=='pause':
            # Cancel only pending jobs whose receipts bind them to this campaign.
            own={p.stem for p in (stage/'submissions').glob('*.json')}
            for row in queue():
                if row[0] in own and row[1]=='PENDING':subprocess.run(['scancel',row[0]],check=True)
        else:
            own={p.stem for p in (stage/'submissions').glob('*.json')}
            if not any(row[0] in own for row in queue()):print(submit(stage))


if __name__=='__main__':main()
