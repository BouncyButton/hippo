"""Idempotent submission respecting the one-running/one-queued cluster limit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from campaign90 import digest, write_json
from scripts.capped_depth_delay.constants import CAMPAIGN
from scripts.capped_depth_delay.preflight import check_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() != 1395:
        raise ValueError('Unexpected cluster account')
    stage = args.stage
    with (stage / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = stage / 'submission.json'
        if receipt.exists():
            print(receipt.read_text())
            return
        plan, protocols = check_frozen(stage, args.pkl)
        pre = json.loads((stage / 'preflight.json').read_text())
        plan_sha = digest(stage / 'source/experiments' / CAMPAIGN / 'plan.json')
        if (pre['status'] != 'passed' or pre['plan_sha256'] != plan_sha
                or pre['source_manifest_sha256'] != digest(stage / 'source_manifest.json')):
            raise ValueError('Preflight binding differs')
        queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j'], text=True).splitlines()
        active = [row.split('|') for row in queue if row.strip()]
        if len(active) > 1 or any(row[1] != 'RUNNING' for row in active):
            raise RuntimeError('One-running/one-queued limit: ' + repr(queue))
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 2:
            raise RuntimeError('Need two GiB for a single verified export slot')
        command = ['sbatch', '--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:' + dependency)
        command.append(str(stage / 'source/scripts/capped_depth_delay/run.sbatch'))
        result = subprocess.run(command, cwd=stage, capture_output=True, text=True, check=True)
        job = int(result.stdout.strip().split(';')[0])
        record = dict(job_id=job, submitted_at_unix=time.time(), stage=str(stage),
            run_root=f'/mnt/beegfsstudents/home/3160552/capped_depth_delay_31976_{job}',
            total_runs=44, delayed_sdf_runs=12, depth_runs=32, fold=0, seed=0,
            single_allocation=True, sequential=True, budget_flops=plan['budget_flops'],
            budget_scope=plan['budget_scope'], calibration_in_cap=False,
            dependency_afterany=dependency, queue_before_submission=queue,
            quota_before_submission=quota, plan_sha256=plan_sha,
            source_manifest_sha256=digest(stage / 'source_manifest.json'),
            preflight_sha256=digest(stage / 'preflight.json'))
        write_json(receipt, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
