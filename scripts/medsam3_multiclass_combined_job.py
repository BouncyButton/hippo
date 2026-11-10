"""Run every frozen fold and seed sequentially in one GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from medsam3_multiclass_replication import aggregate, digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    study = json.loads((root/'study.json').read_text())
    job = os.environ['SLURM_JOB_ID']
    completed = []

    def status(stage, **fields):
        record = dict(status=stage, job_id=job, fold=-1, completed_folds=completed, **fields)
        write(root/'combined_status.json', record)
        write(root/f'job_{job}_status.json', record)

    try:
        assert not sys.flags.optimize
        for name, expected in json.loads((root/'source/SOURCE_MANIFEST.json').read_text()).items():
            assert digest(root/'source'/name) == expected, name
        assert subprocess.check_output(['stat', '-f', '-c', '%T', '/dev/shm'], text=True).strip() == 'tmpfs'
        assert shutil.disk_usage('/dev/shm').free >= 8*1024**3
        with tempfile.TemporaryDirectory(prefix=f'medsam3-ap-{job}-', dir='/dev/shm') as temporary:
            scratch = Path(temporary)
            os.chmod(scratch, 0o700)
            base, ready = scratch/'sam3.pt', scratch/'ready.json'
            status('waiting_for_base', destination=str(base), ready=str(ready), node=os.environ.get('SLURMD_NODENAME'))
            deadline = time.monotonic()+1800
            while not ready.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Checkpoint delivery timed out')
                time.sleep(2)
            assert digest(base) == study['base_sha256']
            for fold in study['folds']:
                status('running', current_fold=fold)
                subprocess.run([sys.executable, str(root/'source/scripts/medsam3_multiclass_replication_job.py'),
                                '--root', str(root), '--fold', str(fold), '--base-weights', str(base)], check=True)
                completed.append(fold)
            status('aggregating')
            aggregate(root)
            status('complete')
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
