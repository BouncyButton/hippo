"""Isolated GPU worker with local checkpoint writes and immutable async copies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sys
import time

from scripts.cross_dataset_campaign.common import ARMS, digest, read, write, install_signals


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    shutil.copyfile(source, temp)
    if digest(source) != digest(temp):
        raise IOError('Checkpoint/metadata copy changed: ' + str(source))
    temp.replace(target)


class CheckpointMirror:
    """Only closed, immutable on-disk files cross the background-thread boundary."""
    def __init__(self, output, durable, interval=180):
        self.output, self.durable, self.interval = output, durable, interval
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.last = -float('inf')

    def capture(self, checkpoint):
        if self.future is not None:
            if not self.future.done():
                return
            self.future.result()
        if time.monotonic() - self.last < self.interval:
            return
        # Trainer commits with atomic replacement: the hardlink keeps this inode.
        snapshot = self.output / 'mirror_snapshot.pt'
        if snapshot.exists():
            raise RuntimeError('Unfinished checkpoint snapshot must be preserved')
        os.link(checkpoint, snapshot)
        for name in ('protocol.json', 'calibration.json'):
            copy_verified(self.output/name, self.durable/name)
        def upload():
            try:
                copy_verified(snapshot, self.durable/'checkpoint_latest.pt')
            finally:
                snapshot.unlink()
        self.future = self.pool.submit(upload)
        self.last = time.monotonic()

    def close(self):
        self.pool.shutdown(wait=True)
        if self.future is not None:
            self.future.result()


def main():
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'output', 'base-weights', 'record'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    # Full-loss calibration also needs graceful signal handling before an
    # optimizer checkpoint exists; its partial rows are sufficient to resume.
    install_signals()
    from scripts.cross_dataset_campaign import foundation_gpu_worker as binding
    # Reuse the existing GPU binding probe without invoking its ADNI trainer.
    probe = args.output/'binding'
    sys.argv = [sys.argv[0], '--probe-directory', str(probe)]
    binding.main()
    write(args.record, read(probe/(os.environ['SLURM_PROCID']+'.json')))
    from scripts.colab_mni_medsam3 import optimized
    original, matched = optimized.original, optimized.matched
    import torch
    output = Path('/tmp')/f"hippo-mni-{os.getuid()}-{os.environ['SLURM_JOB_ID']}"/args.output.name
    output.mkdir(parents=True, exist_ok=True)
    # Restore the exact optimizer/RNG checkpoint; output sidecars may lag it.
    for source in args.output.iterdir():
        if source.is_file() and source.suffix in ('.json', '.pt'):
            copy_verified(source, output/source.name)
    checkpoint = output/'checkpoint_latest.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if read(output/'protocol.json') != {**read(args.protocol), 'arm': args.arm}:
            raise ValueError('Recovery protocol differs')
        if state['protocol_sha256'] != digest(output/'protocol.json') or state['step'] != len(state['history']):
            raise ValueError('Invalid recovery checkpoint')
        write(output/'training_history.json', state['history'])
        del state
    original.Volumes = optimized.Volumes
    original.SupportSlabs = optimized.SupportSlabs
    original.load_foundation = matched.load_foundation
    from scripts.cross_dataset_campaign import adni_foundation_full_runtime as full_runtime
    from scripts.mni_medsam3_cluster.constraint_support import install
    install(read(args.protocol))
    original.calibrate = full_runtime.calibrate
    matched.configure_retained_backward()
    optimized.install_evaluation()
    mirror = CheckpointMirror(output, args.output)
    save = original.save_checkpoint
    def commit(path, state):
        save(path, state)
        mirror.capture(path)
    original.save_checkpoint = commit
    write(output/'environment.json', dict(python=sys.version, torch=torch.__version__,
        source_manifest_sha256=os.environ['MNI_MANIFEST_SHA'],
        recovery='Local atomic checkpoints every5 updates; immutable asynchronous BeeGFS mirror every180s and full clean-stop copy',
        training='MNI native-plus-constraints engine; ADNI full-matrix calibration; paired baseline sample/LR trajectory',
        evaluation='Three-prompt image/text cache, full-first-volume equivalence audit, per-volume sentinel and automatic fallback'))
    sys.argv = [sys.argv[0], '--protocol', str(args.protocol), '--output', str(output),
                '--base-weights', str(args.base_weights), '--arm', args.arm]
    began = time.monotonic()
    stopped = False
    try:
        if args.preflight:
            full_runtime.gpu_preflight()
        else:
            original.main()
    except SystemExit as exc:
        if exc.code != 75: raise
        stopped = True
    finally:
        mirror.close()
        write(output/'worker_timing.json', dict(elapsed_seconds=time.monotonic()-began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), local_scratch=str(output)))
        # Checkpoint first; completion is copied last so partial transfer cannot
        # advertise a finished experiment. Exceptions preserve recovery files.
        if checkpoint.exists():
            copy_verified(checkpoint, args.output/checkpoint.name)
        for source in sorted(output.glob('*.json')):
            if source.name != 'completion.json':
                copy_verified(source, args.output/source.name)
        if (output/'completion.json').exists():
            copy_verified(output/'completion.json', args.output/'completion.json')

    if stopped:
        names = [n for n in ("protocol.json", "checkpoint_latest.pt", "calibration_partial.json") if (args.output/n).exists()]
        write(args.output/"requested_stop.json", dict(status="saved", files={n:digest(args.output/n) for n in names}))
        raise SystemExit(75)


if __name__ == '__main__':
    main()
