"""Idempotent submission of the authorized unaugmented sequential pilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.adni_baselines.core import base
from scripts.adni_constraints.prepare import NAME
from scripts.adni_constraints.worker import verify_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    if os.getuid() != 1395 or stage.name != NAME:
        raise ValueError('Unexpected account or frozen stage')
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = stage/'submission.json'
        if receipt.exists():
            print(receipt.read_text()); return
        plan = verify_stage(stage)
        preflight = json.loads((stage/'preflight.json').read_text())
        if preflight['status'] != 'passed' or preflight['manifest_sha256'] != base.digest(stage/'manifest.json'):
            raise ValueError('Missing preflight bound to frozen manifest')
        queue = subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines()
        active = [r.split('|') for r in queue if r.strip()]
        if len(active) > 1 or any(r[1] != 'RUNNING' for r in active):
            raise RuntimeError('One-running/one-queued policy: '+repr(queue))
        quota = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not match or float(match[2])-float(match[1]) < 10:
            raise RuntimeError('Require 10 GiB free for 56 best models and one active optimizer checkpoint')
        command = ['sbatch','--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:'+dependency)
        command.append(str(stage/'run.sbatch'))
        job = subprocess.check_output(command,cwd=stage,text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected sbatch response')
        record = dict(job_id=int(job),submitted_at_unix=time.time(),stage=str(stage),
            run_root=str(stage)+'_results',models=plan['models'],objectives=plan['objectives'],
            num_runs=56,augmentation='none',fold=0,seed=0,train_subjects=5,train_crops=10,
            val_subjects=27,val_crops=54,budget_gflops_per_run=plan['budget_flops_per_run']/1e9,
            calibration_in_cap=False,dependency_afterany=dependency,queue_before_submission=queue,
            quota_before_submission=quota,manifest_sha256=base.digest(stage/'manifest.json'),
            dataset_sha256=plan['pkl_sha256'],preflight_sha256=base.digest(stage/'preflight.json'))
        base.atomic_json(receipt,record)
        print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
