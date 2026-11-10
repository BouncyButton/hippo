"""Run a single multiclass comparison with the base checkpoint in node RAM."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    status = root/'status.json'
    try:
        if subprocess.check_output(['stat', '-f', '-c', '%T', '/dev/shm'], text=True).strip() != 'tmpfs':
            raise RuntimeError('Scratch must be tmpfs')
        if shutil.disk_usage('/dev/shm').free < 8*1024**3:
            raise RuntimeError('Insufficient RAM scratch')
        with tempfile.TemporaryDirectory(prefix='medsam3-ap-', dir='/dev/shm') as tmp:
            scratch = Path(tmp)
            os.chmod(scratch, 0o700)
            base, ready = scratch/'sam3.pt', scratch/'ready'
            atomic_json(status, dict(status='waiting_for_base', job_id=os.environ['SLURM_JOB_ID'],
                destination=str(base), ready=str(ready), node=os.environ.get('SLURMD_NODENAME')))
            deadline = time.monotonic()+1800
            while not ready.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Base checkpoint transfer timed out')
                time.sleep(2)
            for key in ['TMPDIR', 'MPLCONFIGDIR', 'HF_HOME', 'HF_XET_CACHE', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR']:
                path = scratch/key
                path.mkdir()
                os.environ[key] = str(path)
            atomic_json(status, dict(status='preflight', job_id=os.environ['SLURM_JOB_ID']))
            subprocess.run([sys.executable, 'scripts/medsam3_preflight.py', '--root', str(root),
                '--base-weights', str(base), '--minimum-free-gib', '0.9'], cwd=root/'source', check=True)
            subprocess.run([sys.executable, '-m', 'unittest', 'scripts.test_medsam3_multiclass',
                'scripts.test_medsam3_supervised_constraints'], cwd=root/'source', check=True)
            atomic_json(status, dict(status='running', job_id=os.environ['SLURM_JOB_ID']))
            subprocess.run([sys.executable, 'scripts/train_medsam3_multiclass.py',
                '--upstream', str(root/'MedSAM3'), '--base-weights', str(base),
                '--lora-weights', str(root/'weights/best_lora_weights.pt'),
                '--data', '/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD',
                '--protocol', str(root/'protocol.json'), '--output', str(root/'results')],
                cwd=root/'source', check=True)
            atomic_json(status, dict(status='complete', job_id=os.environ['SLURM_JOB_ID']))
    except Exception as exc:
        atomic_json(status, dict(status='failed', error=repr(exc), job_id=os.environ.get('SLURM_JOB_ID')))
        raise


if __name__ == '__main__':
    main()
