"""Submit exactly one allocation after verified preflight and queue/quota checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from campaign90 import ARMS, BUDGET, CAMPAIGN, MODELS, digest, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args()
    if os.getuid()!=1395:raise ValueError('Unexpected cluster account')
    stage=args.stage
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=stage/'submission.json'
        if receipt.exists():print(receipt.read_text());return
        source=stage/'source'
        manifest_path=stage/'source_manifest.json'
        for name,sha in json.loads(manifest_path.read_text()).items():
            if digest(source/name)!=sha:raise ValueError('Source changed: '+name)
        plan_path=source/'experiments'/CAMPAIGN/'plan.json'
        plan=json.loads(plan_path.read_text())
        preflight=json.loads((stage/'preflight.json').read_text())
        if (preflight['status']!='passed' or preflight['plan_sha256']!=digest(plan_path)
                or preflight['source_manifest_sha256']!=digest(manifest_path)
                or plan['budget_flops_per_run']!=BUDGET or plan['runs']!=24
                or plan['model_order']!=list(MODELS) or plan['arm_order']!=list(ARMS)
                or (plan['fold'],plan['seed'])!=(0,0)):
            raise ValueError('Preflight or authorized plan changed')
        queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines()
        active=[row.split('|') for row in queue if row.strip()]
        if len(active)>1 or any(row[1]!='RUNNING' for row in active):
            raise RuntimeError('One-running/one-queued limit: '+repr(queue))
        quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        limits=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not limits or float(limits[2])-float(limits[1])<30:
            raise RuntimeError('Need30GiB free for24 checkpoint pairs: '+quota)
        command=['sbatch','--parsable']
        dependency=active[0][0] if active else None
        if dependency:command.append('--dependency=afterany:'+dependency)
        command.append(str(source/'scripts/run_campaign90.sbatch'))
        response=subprocess.run(command,cwd=stage,capture_output=True,text=True,check=True)
        job=int(response.stdout.strip().split(';')[0])
        record={'job_id':job,'submitted_at_unix':time.time(),'stage':str(stage),
                'run_root':f'/mnt/beegfsstudents/home/3160552/four_models_six_31976_{job}',
                'models':list(MODELS),'objectives':list(ARMS),'total_runs':24,'fold':0,'seed':0,
                'budget_gflops_per_run':BUDGET/1e9,'single_allocation':True,
                'plan_sha256':digest(plan_path),'preflight_sha256':digest(stage/'preflight.json'),
                'source_manifest_sha256':digest(manifest_path),'dependency_afterany':dependency,
                'queue_before_submission':queue,'quota_before_submission':quota}
        write_json(receipt,record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
