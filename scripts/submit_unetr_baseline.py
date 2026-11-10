"""Submit the verified single UNETR ViT-B16 run respecting the cluster queue limit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_budget_baselines import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    if os.getuid() != 1395:
        raise ValueError('Unexpected cluster account')
    with (stage / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = stage / 'submission.json'
        if receipt.exists():
            print(receipt.read_text()); return
        for name, sha in json.loads((stage / 'source_manifest.json').read_text()).items():
            if digest(stage / 'source' / name) != sha:
                raise ValueError('Frozen source changed: ' + name)
        plan_path = stage / 'source/experiments/unetr_dice_20261005/plan.json'
        plan = json.loads(plan_path.read_text())
        preflight = json.loads((stage / 'preflight.json').read_text())
        if (preflight['status'] != 'passed' or preflight['plan_sha256'] != digest(plan_path)
                or preflight['source_manifest_sha256'] != digest(stage / 'source_manifest.json')
                or plan['candidate_order'] != ['unetr_vit_b16'] or len(plan['runs']) != 1
                or plan['runs'][0]['updates'] != 477 or (plan['fold'], plan['seed']) != (0, 0)
                or plan['runs'][0]['lr'] != .0001
                or plan['budget_flops'] != 62166546655000):
            raise ValueError('Preflight/plan differs from authorized baseline')
        queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j'], text=True).splitlines()
        active = [row.split('|') for row in queue if row.strip()]
        if len(active) > 1 or any(row[1] != 'RUNNING' for row in active):
            raise RuntimeError('One-running/one-queued limit: ' + repr(queue))
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 3:
            raise RuntimeError('Need 3 GiB free for baseline: ' + quota)
        command = ['sbatch', '--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:' + dependency)
        command.append(str(stage / 'source/scripts/run_unetr_baseline.sbatch'))
        response = subprocess.run(command, cwd=stage, capture_output=True, text=True, check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = {'job_id': job, 'submitted_at_unix': time.time(), 'stage': str(stage),
                  'run_root': f'/mnt/beegfsstudents/home/3160552/unetr_dice_{job}',
                  'model': 'unetr_vit_b16', 'objective': 'dice', 'budget_gflops': 62166.546655,
                  'fold': 0, 'seed': 0, 'initial_lr': .0001, 'updates': 477, 'epochs': 47.7,
                  'plan_sha256': digest(plan_path), 'preflight_sha256': digest(stage / 'preflight.json'),
                  'source_manifest_sha256': digest(stage / 'source_manifest.json'),
                  'dependency_afterany': dependency, 'single_allocation': True,
                  'queue_before_submission': queue, 'quota_before_submission': quota}
        write_json(receipt, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
