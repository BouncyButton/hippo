"""Run one pilot allocation, caching the large gated base only on node-local disk."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import hashlib

if __package__:
    from .medsam3_preflight import BASE_REVISION
else:
    from medsam3_preflight import BASE_REVISION


def local_storage(root):
    """Prefer local disk, then RAM-backed storage within the allocation's RAM budget."""
    candidates = [os.environ.get('SLURM_TMPDIR'), '/tmp', '/dev/shm']
    inventory = []
    for value in dict.fromkeys(p for p in candidates if p):
        path = Path(value)
        if not path.is_dir() or not os.access(path, os.W_OK):
            continue
        shared = path.stat().st_dev == root.stat().st_dev
        free = shutil.disk_usage(path).free
        inventory.append(dict(path=str(path), shared_with_beegfs=shared, free_bytes=free))
        if not shared and free >= 8*1024**3:
            print(json.dumps(dict(temporary_storage=inventory, selected=str(path))), flush=True)
            return path
    raise RuntimeError('No private temporary filesystem has 8 GiB available: '+json.dumps(inventory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base-weights', type=Path)
    parser.add_argument('--broadcast-base', action='store_true',
                        help='Wait for Slurm sbcast of user-supplied weights into node /tmp.')
    args = parser.parse_args()
    args.root = args.root.resolve()
    base = args.base_weights or args.root/'weights/sam3.pt'
    job = os.environ['SLURM_JOB_ID']
    status_path = args.root/f'job_{job}_status.json'
    created_broadcast_root = False
    try:
        temporary_root = local_storage(args.root)
        broadcast_root = temporary_root/f'medsam3-{os.getuid()}-{job}'
        broadcast = broadcast_root/'sam3.pt'
        ready = broadcast_root/'ready.json'
        if args.broadcast_base:
            broadcast_root.mkdir(mode=0o700)
            created_broadcast_root = True
            status_path.write_text(json.dumps(dict(status='waiting_for_base', job_id=job,
                                                  destination=str(broadcast), ready=str(ready)))+'\n')
            print(f'Waiting for Slurm broadcast: {broadcast}', flush=True)
            deadline = time.monotonic() + 1200
            while not ready.is_file():
                if time.monotonic() > deadline:
                    raise RuntimeError('Base checkpoint broadcast did not arrive within 20 minutes.')
                time.sleep(2)
            expected = json.loads(ready.read_text())
            h = hashlib.sha256()
            with broadcast.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != expected['sha256'] or broadcast.stat().st_size != expected['bytes']:
                raise RuntimeError('Broadcast checkpoint failed integrity verification.')
            base = broadcast
        subprocess.run([sys.executable, 'scripts/medsam3_preflight.py', '--root', str(args.root),
                        '--base-weights', str(base), '--allow-download'], check=True)
        # /home aliases BeeGFS here; explicitly keep the 3.45GB download in /tmp.
        with tempfile.TemporaryDirectory(prefix=f'medsam3-{job}-', dir=temporary_root) as scratch:
            os.environ['MPLCONFIGDIR'] = str(Path(scratch)/'matplotlib')
            os.environ['HF_XET_CACHE'] = str(Path(scratch)/'xet')
            if not base.is_file():
                from huggingface_hub import hf_hub_download, get_token
                base = Path(hf_hub_download('facebook/sam3', 'sam3.pt', revision=BASE_REVISION,
                                           cache_dir=str(Path(scratch)/'hf'), token=get_token()))
            subprocess.run([
                sys.executable, 'evaluation/medsam3_zero_shot.py',
                '--upstream', str(args.root/'MedSAM3'), '--base-weights', str(base),
                '--lora-weights', str(args.root/'weights/best_lora_weights.pt'),
                '--image', str(args.data/'imagesTr/hippocampus_017_0000.nii.gz'),
                '--label', str(args.data/'labelsTr/hippocampus_017.nii.gz'),
                '--output', str(args.root/f'results_{job}'), '--prompt', 'hippocampus'], check=True)
        status_path.write_text(json.dumps(dict(status='complete', job_id=job))+'\n')
    except Exception as exc:
        status_path.write_text(json.dumps(dict(status='failed', job_id=job, error_type=type(exc).__name__))+'\n')
        raise
    finally:
        if created_broadcast_root:
            shutil.rmtree(broadcast_root)


if __name__ == '__main__':
    main()
