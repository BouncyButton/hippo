"""One queued allocation finishes exact MSD work, then runs the 126-cell follow-on."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from scripts.cross_dataset_campaign import controller, common
from scripts.cross_dataset_campaign import msd_ce_constraints_job as msd
from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, SEISMIC_CAP, read, write, digest
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.prepare_followon import NAME, MODULE

BASE_SUBMIT=controller.submit
BASE_COMMAND=controller.command_for
BASE_CANCEL=controller.cancel_successor


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen follow-on changed: '+name)
    plan=read(stage/'plan.json')
    expected=[]
    for arm in ARMS:
        expected.extend((d,m,arm) for d in ('ADNI','GeoFWI3D') for m in MODELS)
        expected.append(('ADNI','MedSAM3',arm))
    if (plan['name']!=NAME or plan['augmentation']!='none' or plan['num_runs']!=126
            or [(t['dataset'],t['model'],t['arm']) for t in plan['tasks']]!=expected):
        raise ValueError('Wrong 126-condition matrix')
    for task in plan['tasks']:
        if (task['fold'],task['seed'],task['target'])!=(0,0,.6):raise ValueError('Wrong fold/seed/target')
        p=read(stage/'protocols'/task['protocol']/'protocol.json')
        if p['augmentation']!='none':raise ValueError('Augmentation is forbidden')
        if task['kind']=='foundation':
            if (len(p['train_cases'])!=5 or p['updates']!=200 or p['accumulation']!=2
                    or p['gradient_ratio_target']!=.6 or p['gradient_ratio_p95_cap']!=.75
                    or p['baseline_loss']!='Unchanged established MedSAM3 native fine-tuning loss'):
                raise ValueError('MedSAM3 recipe changed')
        else:
            cap=SEISMIC_CAP if task['dataset']=='GeoFWI3D' else CAP
            if (p['weight_ce']!=1 or p['calibration_base']!='Dice+CE' or p['calibration_target_ratio']!=.6
                    or p['calibration_p95_cap']!=.75 or p['budget_flops']!=cap):
                raise ValueError('Wrong baseline, calibration or cap')
            profile=read(stage/'protocols'/task['protocol']/'profile.json')
            s=p['models'][p['trainer_alias']]['objectives'][task['arm']]
            if s!=schedule(profile['costs'],task['arm'],s,p['train_count'],cap):
                raise ValueError('Measured FLOP schedule differs')
    return plan


def check_predecessor(plan):
    predecessor=plan['predecessor'];stage=Path(predecessor['stage'])
    if digest(stage/'manifest.json')!=predecessor['manifest_sha256']:
        raise ValueError('MSD predecessor has changed')
    return stage,msd.verify(stage)


def predecessor_complete(stage,plan):
    status=read(stage/'status.json') if (stage/'status.json').exists() else {}
    if status.get('status')!='complete' or status.get('completed')!=len(plan['tasks']):return False
    for task in plan['tasks']:
        run=stage/'runs'/task['id']
        if (read(run/'completion.json')['status']!='complete' or read(run/'cleanup.json')['status']!='complete'
                or read(run/'archive_acknowledgment.json')['manifest_sha256']!=digest(run/'archive_manifest.json')):
            raise ValueError('Predecessor result/archive is incomplete')
    return True


def call_submit(stage,continuation=None):
    previous=controller.verify;controller.verify=verify
    try:return BASE_SUBMIT(stage,continuation)
    finally:controller.verify=previous


def ensure_successor(stage,job):
    path=stage/'successor.json'
    if path.exists():
        receipt=read(path)
        if receipt['dependency']==str(job) and receipt['manifest_sha256']==digest(stage/'manifest.json'):
            if any(row[0]==receipt['job_id'] and row[1]=='PENDING' for row in controller.queue()):
                return receipt
    return call_submit(stage,str(job))


def submit(stage):
    plan=verify(stage);predecessor,old_plan=check_predecessor(plan)
    if read(stage/'preflight.json').get('manifest_sha256')!=digest(stage/'manifest.json'):
        raise ValueError('Successful matching preflight required before queue replacement')
    if read(stage/'preflight.json').get('status')!='passed':raise ValueError('Preflight did not pass')
    usage=controller.quota()
    if usage['free_gib']<plan['storage']['minimum_free_gib']:raise ValueError('Insufficient personal quota')
    # Only the predecessor's known queued continuation may be replaced.
    pending_receipt=read(predecessor/'successor.json') if (predecessor/'successor.json').exists() else None
    with (stage/'handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=controller.queue();pending=[r for r in queue if r[1]=='PENDING']
        if pending and (len(pending)!=1 or not pending_receipt or pending[0][0]!=pending_receipt['job_id']
                or pending_receipt['manifest_sha256']!=plan['predecessor']['manifest_sha256']):
            raise RuntimeError('An unrelated queued job occupies the slot')
        running=[r for r in queue if r[1] in ('RUNNING','COMPLETING','CONFIGURING')]
        own={p.stem for p in (predecessor/'submissions').glob('*.json')}
        if running and (len(running)!=1 or running[0][0] not in own):
            raise RuntimeError('An unrelated active job occupies the slot')
        previous_control=read(predecessor/'control.json')
        if previous_control.get('pause') or read(stage/'control.json').get('pause'):
            raise RuntimeError('Explicitly paused campaign cannot be submitted')
        if pending:
            subprocess.run(['scancel',pending[0][0]],check=True)
            for _ in range(30):
                if not any(r[0]==pending[0][0] for r in controller.queue()):break
                time.sleep(1)
            else:raise RuntimeError('Queued continuation has not cancelled')
        write(predecessor/'control.json',{**previous_control,'automatic_continuation_owner':str(stage)})
        try:
            receipt=call_submit(stage)
        except Exception:
            # Restore the old continuation if submission definitively failed.
            current=controller.queue()
            if pending and not any(r[1]=='PENDING' for r in current) and running:
                previous=controller.verify;controller.verify=msd.verify
                try:BASE_SUBMIT(predecessor,running[0][0])
                finally:controller.verify=previous
            write(predecessor/'control.json',previous_control)
            raise
        write(stage/'handoff.json',dict(status='queued',predecessor_stage=str(predecessor),
            replaced_continuation=pending[0][0] if pending else None,submission=receipt,
            policy='Resume exact frozen MSD trainers if needed; only then start the follow-on; one queued continuation for the whole chain'))
        return receipt


def command_for(stage,plan,task,base_path):
    command=BASE_COMMAND(stage,plan,task,base_path)
    if task['kind']=='conventional':command[2]='scripts.cross_dataset_campaign.binary_ce_constraints'
    return command


def run(stage):
    plan=verify(stage);predecessor,old_plan=check_predecessor(plan)
    job=os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_ALLOCATION_DEADLINE']=str(time.time()+plan['graceful_stop_seconds'])
    previous_status=read(predecessor/'status.json') if (predecessor/'status.json').exists() else {}
    if read(stage/'control.json').get('pause') or read(predecessor/'control.json').get('pause'):
        controller.summarize(stage,plan,'paused');return
    if previous_status.get('status')=='failed':
        controller.summarize(stage,plan,'blocked_predecessor_failed');return
    controller.submit=lambda ignored,continuation=None:ensure_successor(stage,job)
    def cancel(phase,active_job):
        status=read(phase/'status.json') if (phase/'status.json').exists() else {}
        if phase==predecessor and status.get('status')=='complete':return
        BASE_CANCEL(stage,active_job)
    controller.cancel_successor=cancel
    if not predecessor_complete(predecessor,old_plan):
        controller.verify=msd.verify
        def predecessor_command(phase,phase_plan,task,base_path):
            command=BASE_COMMAND(phase,phase_plan,task,base_path)
            command[2]='scripts.cross_dataset_campaign.msd_ce_constraints'
            path=':'.join(str(predecessor/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
            return ['env','PYTHONPATH='+path,*command]
        controller.command_for=predecessor_command
        try:
            controller.run(predecessor)
        except Exception:
            controller.summarize(stage,plan,'blocked_predecessor_failed');BASE_CANCEL(stage,job);raise
        if not predecessor_complete(predecessor,old_plan):
            status=read(predecessor/'status.json')['status']
            controller.summarize(stage,plan,status,active='predecessor:'+predecessor.name)
            return
    controller.verify=verify;controller.command_for=command_for
    controller.run(stage)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['submit','run','preflight','status','pause','resume'])
    parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage.resolve()
    if args.command=='submit':print(submit(stage))
    elif args.command=='run':run(stage)
    elif args.command=='preflight':
        from scripts.cross_dataset_campaign.preflight_followon import preflight
        preflight(stage)
    elif args.command=='status':
        print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Queued, not started')
    else:
        plan=verify(stage);predecessor=Path(plan['predecessor']['stage'])
        for phase in (stage,predecessor):
            control=read(phase/'control.json');control['pause']=args.command=='pause';write(phase/'control.json',control)
        own={p.stem for phase in (stage,predecessor) for p in (phase/'submissions').glob('*.json')}
        if args.command=='pause':
            for row in controller.queue():
                if row[0] in own and row[1]=='PENDING':subprocess.run(['scancel',row[0]],check=True)
        elif not any(row[0] in own for row in controller.queue()):print(call_submit(stage))


if __name__=='__main__':main()
