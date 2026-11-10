"""Submit one verified bands allocation, respecting the one-running/one-queued limit."""
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
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() != 1395:
        raise ValueError('Unexpected cluster account')
    stage = args.stage
    with (stage / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = stage / 'submission.json'
        if receipt.exists():
            print(receipt.read_text()); return
        for name, sha in json.loads((stage / 'source_manifest.json').read_text()).items():
            if digest(stage / 'source' / name) != sha:
                raise ValueError('Frozen source changed: ' + name)
        campaign_path = stage / 'source/experiments/budget_bands_94603_20261005/campaign.json'
        campaign = json.loads(campaign_path.read_text())
        preflight = json.loads((stage / 'preflight.json').read_text())
        if preflight['status'] != 'passed' or preflight['campaign_sha256'] != digest(campaign_path):
            raise ValueError('Preflight does not bind this campaign')
        queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j'], text=True).splitlines()
        active = [line.split('|') for line in queue if line.strip()]
        if len(active) > 1 or any(row[1] != 'RUNNING' for row in active):
            raise RuntimeError('Wait for the current queue before submitting: ' + repr(queue))
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 5:
            raise RuntimeError('Insufficient free quota: ' + quota)
        command = ['sbatch', '--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:' + dependency)
        command.append(str(stage / 'source/scripts/run_budget_bands.sbatch'))
        response = subprocess.run(command, cwd=stage, capture_output=True, text=True, check=True)
        job_id = int(response.stdout.strip().split(';')[0])
        record = {'job_id': job_id, 'stage': str(stage),
                  'run_root': f'/mnt/beegfsstudents/home/3160552/budget_bands_94603_{job_id}',
                  'submitted_at_unix': time.time(), 'budget_gflops_per_run': 94603.78,
                  'fold': 0, 'seed': 0, 'objective': 'dice_bands', 'baseline_job_id': 684354,
                  'model_order': campaign['model_order'], 'updates': campaign['max_updates'],
                  'source_manifest_sha256': digest(stage / 'source_manifest.json'),
                  'campaign_sha256': digest(campaign_path), 'preflight_sha256': digest(stage / 'preflight.json'),
                  'single_sequential_allocation': True, 'dependency_afterany': dependency,
                  'queue_before_submission': queue, 'quota_before_submission': quota}
        temporary = receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n'); temporary.replace(receipt)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
