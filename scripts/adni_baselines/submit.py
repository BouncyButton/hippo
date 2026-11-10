"""Submit once, after verifying the frozen stage and cluster dataset."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .core import base
from .worker import verify_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    args = parser.parse_args()
    stage = args.stage
    if os.getuid() != 1395 or stage.name != 'adni_baselines_5pct_fold0_seed0_20261013':
        raise ValueError('Unexpected cluster account or stage')
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt = stage/'submission.json'
        if receipt.exists():
            print(receipt.read_text()); return
        plan = verify_stage(stage)
        queue = subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines()
        active = [r.split('|') for r in queue if r.strip()]
        if len(active)>1 or any(r[1]!='RUNNING' for r in active):
            raise RuntimeError('Respecting one-running/one-queued policy: '+repr(queue))
        quota = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not match or float(match[2])-float(match[1])<5:
            raise RuntimeError('Need 5 GiB free for retained model checkpoints')
        command = ['sbatch','--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:'+dependency)
        command.append(str(stage/'run.sbatch'))
        job = subprocess.check_output(command,cwd=stage,text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected sbatch response')
        record = dict(job_id=int(job),submitted_at_unix=time.time(),stage=str(stage),
            run_root='/mnt/beegfsstudents/home/3160552/adni_baselines_5pct_fold0_seed0_20261013_results',
            models=[r['model'] for r in plan['runs']],fold=0,seed=0,train_subjects=5,train_crops=10,
            val_subjects=27,val_crops=54,budget_gflops_per_model=plan['budget_flops_per_model']/1e9,
            dependency_afterany=dependency,queue_before_submission=queue,quota_before_submission=quota,
            manifest_sha256=base.digest(stage/'manifest.json'),dataset_sha256=plan['pkl_sha256'])
        base.atomic_json(receipt,record)
        print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
