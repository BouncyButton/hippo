"""Single Slurm allocation: three supervised five-shot training arms."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from medsam3_cluster_job import local_storage
from medsam3_preflight import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    job = os.environ['SLURM_JOB_ID']
    status_path = root/f'job_{job}_status.json'
    broadcast_root = None
    try:
        local = local_storage(root)
        with tempfile.TemporaryDirectory(prefix=f'medsam3-losses-{job}-', dir=local) as scratch:
            broadcast_root = Path(scratch)
            base, ready = broadcast_root/'sam3.pt', broadcast_root/'ready.json'
            status_path.write_text(json.dumps(dict(status='waiting_for_base', job_id=job,
                                                   destination=str(base), ready=str(ready)))+'\n')
            deadline = time.monotonic()+1200
            while not ready.is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError('No checkpoint broadcast after 20 minutes.')
                time.sleep(2)
            expected = json.loads(ready.read_text())
            protocol = root/'source/docs/experiments/medsam3_supervised_5shot_20260928/protocol.json'
            settings = json.loads(protocol.read_text())
            if expected['sha256'] != settings['base_sha256'] or digest(base) != settings['base_sha256']:
                raise RuntimeError('Base checkpoint checksum differs from frozen protocol.')
            if base.stat().st_size != 3450062241:
                raise RuntimeError('Unexpected checkpoint size.')
            subprocess.run([sys.executable, 'scripts/medsam3_preflight.py', '--root', str(root),
                            '--minimum-free-gib', '2', '--base-weights', str(base)], check=True)
            # Keep all large weights/compiler caches outside the BeeGFS quota.
            for key, name in [('TMPDIR', 'tmp'), ('MPLCONFIGDIR', 'matplotlib'),
                              ('HF_HOME', 'hf'), ('HF_XET_CACHE', 'xet'),
                              ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton')]:
                path = broadcast_root/name
                path.mkdir()
                os.environ[key] = str(path)
            status_path.write_text(json.dumps(dict(status='running', job_id=job))+'\n')
            subprocess.run([sys.executable, 'scripts/train_medsam3_constraint_few_shot.py',
                            '--upstream', str(root/'MedSAM3'), '--base-weights', str(base),
                            '--lora-weights', str(root/'weights/best_lora_weights.pt'),
                            '--data', str(args.data), '--protocol', str(protocol),
                            '--output', str(root/f'results_{job}')], check=True)
        status_path.write_text(json.dumps(dict(status='complete', job_id=job))+'\n')
    except Exception as exc:
        status_path.write_text(json.dumps(dict(status='failed', job_id=job, error_type=type(exc).__name__))+'\n')
        raise


if __name__ == '__main__':
    main()
