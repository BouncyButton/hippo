"""Submit the authorized single allocation once, respecting the queue limit."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stage = Path(__file__).resolve().parents[1]
    assert os.getuid() == 1395, 'Use the authorized cluster account'
    receipt = stage / 'submission.json'
    with (stage / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            print(receipt.read_text());return
        manifest = json.loads((stage / 'payload_manifest.json').read_text())
        for relative, expected in manifest.items():
            assert digest(stage / relative) == expected, relative
        protocol_path = stage / 'experiments/compute_matched_constraints_20pct_20261004/protocol.json'
        protocol = json.loads(protocol_path.read_text())
        assert protocol['status'] == 'ready' and protocol['budget_flops'] == 8_138_082_968_869
        assert protocol['folds'] == protocol['seeds'] == [0]
        assert list(protocol['models']) == ['unet'] and len(protocol['models']['unet']['objectives']) == 8
        check = json.loads((stage / 'verification.json').read_text())
        assert check['cluster_tests_passed'] == 9 and check['protocol_sha256'] == digest(protocol_path)
        data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
        assert digest(data) == protocol['pkl_sha256']
        quota = subprocess.run(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'],
                               capture_output=True, text=True, check=True).stdout
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 5:
            raise RuntimeError('Need at least 5 GiB free in the actual user quota before submission: ' + quota)
        queue = subprocess.run(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j'],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        states = [line.split('|')[1] for line in queue if line.strip()]
        if states.count('PENDING') >= 1 or len(states) > 1:
            raise RuntimeError('Submission would exceed the one-running/one-queued limit: ' + repr(queue))
        sbatch = stage / 'scripts/run_compute_matched_constraints.sbatch'
        response = subprocess.run(['sbatch', '--parsable', str(sbatch)], cwd=stage,
                                  capture_output=True, text=True, check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = {'job_id': job, 'submitted_at_unix': time.time(), 'stage': str(stage),
                  'run_root': f'/mnt/beegfsstudents/home/3160552/compute_matched_constraints20_{job}',
                  'protocol_sha256': digest(protocol_path), 'sbatch_sha256': digest(sbatch),
                  'queue_before_submission': queue, 'quota_before_submission': quota,
                  'models': ['unet'], 'folds': [0], 'seeds': [0], 'runs': 8,
                  'budget_flops_per_run': protocol['budget_flops']}
        temporary = receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n');temporary.replace(receipt)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
