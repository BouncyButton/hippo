"""Deploy and submit one laptop-independent job within an absolute three-hour window."""
import argparse
import json
import math
import subprocess
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.mni_constraints import nnunet_swin_refine2030 as study
from scripts.mni_constraints import launch_unet_bands_timing_strength as shared

STAGE, REPORT = study.ROOT/'experiments'/study.NAME, study.ROOT/'reports'/study.NAME
DEST, PARENT = REMOTE/study.NAME, REMOTE/study.PARENT
shared.study, shared.STAGE, shared.REPORT, shared.DEST, shared.PARENT = study, STAGE, REPORT, DEST, PARENT


def submit():
    assert read(REPORT/'remote_preflight.json')['status'] == 'passed'
    assert not (REPORT/'submission.json').exists()
    assert time.time()+600 < read(STAGE/'plan.json')['deadline']
    code = '''import json,sys,time,math,subprocess,fcntl
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import quota
from scripts.mni_constraints import nnunet_swin_refine2030 as study
p=Path(sys.argv[1]);parent=Path(sys.argv[2]);plan=study.verify(p)
assert read(parent/'control.json')['pause'],'Broader sweep must remain paused.'
assert read(p/'preflight.json')['status']=='passed' and read(p/'preflight.json')['manifest_sha256']==digest(p/'manifest.json')
assert not read(p/'control.json')['pause']
with (p/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (p/'submission.json').exists()
 assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip(),'Account already has an active job.'
 now=time.time();assert now+600<plan['deadline']
 usage=quota();assert usage['free_gib']>=max(plan['storage']['minimum_free_gib'],plan['storage']['estimated_peak_mirrored_gib']+16)
 minutes=max(1,math.floor((plan['deadline']-now)/60))
 command=['sbatch','--parsable','--time='+str(minutes),'--chdir='+str(p),'--output='+str(p/'slurm-%j.out'),'--error='+str(p/'slurm-%j.err'),str(p/'run.sbatch'),str(p)]
 job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit()
 r=dict(job_id=job,submitted_at=time.time(),deadline=plan['deadline'],slurm_time_limit_minutes=minutes,quota=usage,command=command,manifest_sha256=digest(p/'manifest.json'),laptop_required=False,automatic_successor=False,broader_sweep_control=read(parent/'control.json'))
 write(p/'submission.json',r);write(p/'submissions'/f'{job}.json',r);print(json.dumps(r))'''
    env=':'.join(str(DEST/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    receipt=json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',code,DEST,PARENT],timeout=900))
    write(REPORT/'submission.json',receipt)
    # Helpful while connected; training/selection do not wait for this collector.
    shared.collector(DEST,STAGE,REPORT,'scripts.mni_constraints.collect_bounded_refinement')
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['deploy','submit'])
    action=parser.parse_args().action
    if action=='deploy':shared.deploy()
    else:submit()


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:
        print(error.output,error.stderr,flush=True);raise
