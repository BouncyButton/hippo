"""Idempotent, quota-aware submission; preserve at most one queued allocation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    p = json.loads((root/'protocol.json').read_text())
    if os.getuid() != 1395 or root.name != 'band_evolution_fold0_seed0_5pct_20261010_01' or p['name'] != root.name:
        raise RuntimeError('Incorrect staging location')
    receipt = root/'submission.json'
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            print(receipt.read_text()); return
        attempt = root/'submission_attempt.json'
        if attempt.exists():
            raise RuntimeError('Unresolved prior attempt; inspect scheduler before any resubmission')
        for name, expected in json.loads((root/'payload_manifest.json').read_text()).items():
            if sha(root/name) != expected:
                raise ValueError('Payload changed: '+name)
        checks = json.loads((root/'local_checks.json').read_text())
        assert checks['passed'] and checks['protocol_sha256'] == sha(root/'protocol.json')
        assert p['main_runs'] == len(p['runs']) and 100 <= p['main_runs'] < 240 and p['fold'] == p['seed'] == 0
        assert set(p['splits']) == {'5pct'} and not p['auxiliary_branch']
        assert sha(p['pkl']) == p['pkl_sha256']
        prefix = json.loads((root/'shared_warmup/provenance.json').read_text())
        for name, expected in prefix['artifacts'].items():
            assert sha(root/'shared_warmup'/name) == expected
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not match:
            raise RuntimeError('Unparseable quota: '+quota)
        free = float(match[2])-float(match[1])
        if free < p['minimum_free_gib']:
            raise RuntimeError('Insufficient free quota: '+quota)
        queue = subprocess.check_output(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j'], text=True).splitlines()
        jobs = [line.split('|') for line in queue if line.strip()]
        if len(jobs) > 1 or any(j[1] != 'RUNNING' for j in jobs):
            raise RuntimeError('Preserve one running and at most one pending allocation: '+repr(queue))
        dependencies = [j[0] for j in jobs]
        command = ['sbatch', '--parsable', f'--output={root}/slurm-%j.out', f'--error={root}/slurm-%j.err']
        if dependencies:
            command.append('--dependency=afterany:'+':'.join(dependencies))
        command += [str(root/'source/scripts/band_evolution/run.sh'), str(root)]
        request = dict(command=command, requested_at_unix=time.time(), protocol_identity=p['identity_sha'],
                       protocol_sha256=sha(root/'protocol.json'), runs=p['main_runs'], fold=0, seed=0, fractions=['5pct'],
                       dependencies=dependencies, queue_before_submission=queue, quota_before_submission=quota,
                       free_gib=free, auxiliary_branch=False, shared_warmup=True, max_epochs=p['max_epochs'])
        attempt.write_text(json.dumps(request, indent=2)+'\n')
        with attempt.open('rb') as stream:
            os.fsync(stream.fileno())
        result = subprocess.run(command, cwd=root/'source', capture_output=True, text=True)
        request.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        (root/'submission_response.json').write_text(json.dumps(request, indent=2)+'\n')
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():
            raise RuntimeError('Submission not confirmed; inspect response')
        request['job_id'] = job
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(request, indent=2)+'\n'); temporary.replace(receipt)
        print(json.dumps(request, indent=2))


if __name__ == '__main__':
    main()
