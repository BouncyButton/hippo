"""Submit the authorized continuation once, with an actual-quota reserve."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recovery',type=Path,required=True)
    stage=parser.parse_args().recovery
    if os.getuid()!=1395 or stage.name!='adni_constraints_noaug_recovery_20261013':
        raise ValueError('Unexpected account or stage')
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=stage/'submission.json'
        if receipt.exists():
            print(receipt.read_text());return
        for name,sha in json.loads((stage/'manifest.json').read_text()).items():
            if digest(stage/name)!=sha:
                raise ValueError('Frozen recovery changed')
        control=json.loads((stage/'recovery_plan.json').read_text())
        clean=json.loads((stage/'cleanup_receipt.json').read_text())
        if clean['status']!='complete' or clean['archive_catalog_sha256']!=digest(stage/'checkpoint_archive.json'):
            raise ValueError('Cleanup must be completed and bound to verified backups')
        queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines()
        jobs=[r.split('|') for r in queue if r.strip()]
        if len(jobs)>1 or any(r[1]!='RUNNING' for r in jobs):
            raise RuntimeError('One-running/one-queued policy: '+repr(queue))
        quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        found=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not found or float(found[2])-float(found[1])<control['minimum_quota_free_gib']:
            raise RuntimeError('Insufficient quota reserve: '+quota)
        command=['sbatch','--parsable']
        if jobs:
            command.append('--dependency=afterany:'+jobs[0][0])
        command.append(str(stage/'run.sbatch'))
        job=subprocess.check_output(command,cwd=stage,text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm response')
        record=dict(job_id=int(job),original_job_id=698935,submitted_at_unix=time.time(),
            completed_runs_reused=22,unfinished_runs=34,resume_update=147,augmentation='none',
            fold=0,seed=0,budget_gflops_per_trajectory=control['training_cap_flops_per_completed_trajectory']/1e9,
            authorized_replay=control['replay'],quota_before_submission=quota,
            dependency_afterany=jobs[0][0] if jobs else None,manifest_sha256=digest(stage/'manifest.json'))
        temporary=receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(receipt)
        print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
