"""Verify and submit exactly one three-seed momentum experiment."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

STAGE = Path('/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261004/nnunet_momentum095')
ARCHIVE = Path('/home/3160552/hippo-nnunet-m095-20261004.tar.gz')
PROTOCOL = STAGE / 'experiments/nnunet_momentum095_20261004'
PYTHON = '/home/3160552/.conda/envs/hippocampus/bin/python'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight():
    manifest = json.loads((PROTOCOL / 'staging_manifest.json').read_text())
    for relative, expected in manifest.items():
        assert digest(STAGE / relative) == expected, relative
    with tarfile.open(ARCHIVE) as tar:
        for member in tar.getmembers():
            if member.isfile():
                expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                assert digest(STAGE / member.name) == expected, member.name
    assert digest(Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')) == json.loads((PROTOCOL / 'protocol.json').read_text())['pkl_sha256']
    subprocess.run(['bash', '-n', 'scripts/run_nnunet_momentum095.sbatch'], cwd=STAGE, check=True)
    env = dict(os.environ, PYTHONPATH=f'{STAGE}:{STAGE}/baselines/nnUNet',
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/hippo-m095-preflight-mpl',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    subprocess.run([PYTHON, '-m', 'unittest',
                    'scripts.test_compute_matched_baselines.ScheduleTests.test_momentum_revision_isolated_and_used_by_optimizer',
                    'scripts.test_compute_matched_baselines.LossSemanticTests', '-v'],
                   cwd=STAGE, env=env, check=True)
    print('Package/data hashes, bash syntax, and three cluster preflight tests passed.', flush=True)


def submit():
    with (STAGE / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = STAGE / 'submission.json'
        if receipt.exists():
            print(receipt.read_text()); return
        attempt = STAGE / 'submission_attempt.json'
        if attempt.exists():
            raise RuntimeError('Submission outcome needs inspection; will not submit a duplicate.')
        queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i|%j|%T'], text=True)
        if len(queue.strip().splitlines()) >= 2:
            raise RuntimeError('Two jobs already present; will not exceed the user queue allowance: ' + queue)
        preflight()
        attempt.write_text(json.dumps({'attempted_at_utc': datetime.now(timezone.utc).isoformat(),
                                      'archive_sha256': digest(ARCHIVE)}, indent=2) + '\n')
        result = subprocess.run(['sbatch', '--parsable', 'scripts/run_nnunet_momentum095.sbatch'],
                                cwd=STAGE, text=True, capture_output=True, check=True)
        job_id = int(result.stdout.strip().split(';')[0])
        data = {'job_id': job_id, 'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
                'jobs_submitted': 1, 'sequential_runs': 3, 'models': ['nnunet'],
                'folds': [0], 'seeds': [0, 1, 2], 'sgd_momentum': .95,
                'warmup_updates': 0, 'updates_per_run': 327,
                'budget_gflops_per_run': 116259, 'expected_training_gflops_per_run': 116181.104090427,
                'expected_total_training_gflops': 348543.312271281,
                'cluster_stage': str(STAGE),
                'run_root': f'/mnt/beegfsstudents/home/3160552/nnunet_momentum095_{job_id}',
                'protocol_sha256': digest(PROTOCOL / 'protocol.json'),
                'trainer_sha256': digest(STAGE / 'scripts/train_compute_matched_baselines.py'),
                'sbatch_sha256': digest(STAGE / 'scripts/run_nnunet_momentum095.sbatch'),
                'archive_sha256': digest(ARCHIVE),
                'verification': {'local_regression_tests': 10, 'cluster_preflight_tests': 3,
                                 'package_dataset_hashes': 'passed', 'bash_syntax': 'passed'},
                'note': 'Post hoc fold-0 momentum diagnostic. Report all three seeds; no per-seed best-of-recipe selection.'}
        temporary = STAGE / 'submission.json.tmp'
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(receipt)
        print(json.dumps(data, indent=2), flush=True)


if __name__ == '__main__':
    submit()
