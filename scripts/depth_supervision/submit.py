"""Idempotently submit one allocation for exactly eight sequential training runs."""
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
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    source = Path(__file__).resolve().parents[2]
    root = source.parent
    if os.getuid() != 1395 or root.name != 'depth_swinunetr_fold0_seed0_20261006_01':
        raise RuntimeError('Submit only from the dedicated cluster staging directory')
    receipt = root/'submission.json'
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            print(receipt.read_text())
            return
        attempt = root/'submission_attempt.json'
        if attempt.exists():
            raise RuntimeError('Unresolved previous submission attempt; inspect Slurm before retrying')
        manifest = json.loads((root/'payload_manifest.json').read_text())
        for name, expected in manifest.items():
            if sha(root/name) != expected:
                raise RuntimeError('Payload changed: '+name)
        protocol_path = source/'docs/experiments/depth_swinunetr_20261006/protocol.json'
        p = json.loads(protocol_path.read_text())
        expected_arms = ['dice','depth','depth_sdf3d','depth_sdf_layer']
        assert p['arms'] == expected_arms and p['fold'] == p['seed'] == 0
        assert p['max_epochs'] == 75 and p['baseline_loss'] == 'dice'
        assert len(p['splits']['full']['train']) == 208 and len(p['splits']['5pct']['train']) == 10
        assert len(p['splits']['full']['val']) == len(p['splits']['5pct']['val']) == 52
        checks = json.loads((root/'local_checks.json').read_text())
        assert checks['passed'] and checks['protocol_sha256'] == sha(protocol_path)
        for name, expected in p['source_sha256'].items():
            assert sha(source/name) == expected, name
        data = Path('/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
        assert sha(data) == p['pkl_sha256']
        quota = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'], text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2])-float(limits[1]) < 10:
            raise RuntimeError('Need at least 10 GiB free for checkpoints and geometry: '+quota)
        queue = subprocess.check_output(['squeue','-u','3160552','-h','-o','%i|%T|%j'], text=True).splitlines()
        entries = [line.split('|') for line in queue if line.strip()]
        if len(entries) > 1 or any(e[1] != 'RUNNING' for e in entries):
            raise RuntimeError('Keep one running job and one queued allocation: '+repr(queue))
        script = source/'scripts/depth_supervision/run.sh'
        argv = ['sbatch','--parsable',f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
        dependencies = [e[0] for e in entries]
        if dependencies:
            argv.append('--dependency=afterany:'+':'.join(dependencies))
        argv += [str(script), str(root)]
        request = dict(command=argv, queue_before_submission=queue, quota_before_submission=quota,
                       requested_at_unix=time.time(), protocol_sha256=sha(protocol_path),
                       root=str(root), runs=8, arms=expected_arms, fractions=['5pct','full'],
                       fold=0, seed=0, max_epochs=75, dependencies=dependencies)
        attempt.write_text(json.dumps(request, indent=2)+'\n')
        with attempt.open('rb') as stream:
            os.fsync(stream.fileno())
        result = subprocess.run(argv, cwd=source, capture_output=True, text=True)
        request.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        (root/'submission_response.json').write_text(json.dumps(request, indent=2)+'\n')
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():
            raise RuntimeError('Submission not confirmed; inspect submission_response.json')
        request['job_id'] = job
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(request, indent=2)+'\n')
        temporary.replace(receipt)
        print(json.dumps(request, indent=2))


if __name__ == '__main__':
    main()
