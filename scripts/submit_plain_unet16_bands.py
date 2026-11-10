"""Submit one bands fallback only after the completed UNETR pilot is unsuitable."""
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
    parser.add_argument('--unetr-root', type=Path, required=True)
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
        directory = stage / 'source/experiments/plain_unet16_bands_20261005/dice_bands'
        protocol = json.loads((directory / 'protocol.json').read_text())
        check = json.loads((stage / 'preflight.json').read_text())
        if (check['status'] != 'passed' or check['protocol_sha256'] != digest(directory / 'protocol.json')
                or check['source_manifest_sha256'] != digest(stage / 'source_manifest.json')
                or protocol['models']['unet']['objectives']['dice_bands']['max_steps'] != 1091
                or protocol['budget_flops'] != 62166546655000 or protocol['folds'] != [0]
                or protocol['seeds'] != [0] or protocol['candidate_name'] != 'basic_unet16'):
            raise ValueError('Unverified fallback recipe')
        summary_path = args.unetr_root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        output = args.unetr_root / 'unetr_vit_b16/dice/fold0/seed0/run'
        completion = json.loads((output / 'completion.json').read_text())
        if (summary['status'] != 'complete' or summary['model'] != 'unetr_vit_b16'
                or summary['run']['completion_sha256'] != digest(output / 'completion.json')
                or summary['run']['updates'] != 477 or summary['run']['discarded_attempts'] != 0
                or completion['metrics_sha256'] != digest(output / 'metrics.csv')
                or all(.70 <= summary['run'][k] <= .80 for k in ['best_dice', 'final_dice'])):
            raise ValueError('UNETR has not met the authorized fallback condition')
        queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j'], text=True).splitlines()
        active = [row.split('|') for row in queue if row.strip()]
        if len(active) > 1 or any(row[1] != 'RUNNING' for row in active):
            raise RuntimeError('One-running/one-queued limit: ' + repr(queue))
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 3:
            raise RuntimeError('Need 3 GiB free: ' + quota)
        command = ['sbatch', '--parsable', '--export=ALL,UNETR_REFERENCE_ROOT=' + str(args.unetr_root)]
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:' + dependency)
        command.append(str(stage / 'source/scripts/run_plain_unet16_bands.sbatch'))
        response = subprocess.run(command, cwd=stage, capture_output=True, text=True, check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = {'job_id': job, 'submitted_at_unix': time.time(), 'stage': str(stage),
                  'run_root': f'/mnt/beegfsstudents/home/3160552/plain_unet16_bands_{job}',
                  'candidate': 'basic_unet16', 'objective': 'dice_bands', 'baseline_job_id': 684625,
                  'unetr_reference_root': str(args.unetr_root), 'unetr_summary_sha256': digest(summary_path),
                  'budget_gflops': 62166.546655, 'fold': 0, 'seed': 0, 'updates': 1091, 'epochs': 109.1,
                  'protocol_sha256': digest(directory / 'protocol.json'),
                  'preflight_sha256': digest(stage / 'preflight.json'),
                  'source_manifest_sha256': digest(stage / 'source_manifest.json'),
                  'dependency_afterany': dependency, 'single_allocation': True,
                  'queue_before_submission': queue, 'quota_before_submission': quota}
        write_json(receipt, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
