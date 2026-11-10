"""Observe the frozen entry point without changing its scientific operations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time


def atomic_local(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def timed_call(stats, key, function):
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            stats[key] = stats.get(key, 0.) + time.monotonic() - start
    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--control-sha', required=True)
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer_args = args.trainer_args
    if trainer_args[:1] == ['--']:
        trainer_args = trainer_args[1:]
    output = Path(trainer_args[trainer_args.index('--output') + 1])
    # Handle a stop even during imports, before the frozen trainer installs its handlers.
    def early_stop(signum, frame):
        os.environ['HIPPO_DEADLINE'] = '0'
    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, early_stop)
    atomic_local(args.status, dict(pid=os.getpid(), ready=True, updated_at=time.time()))
    sys.path.insert(0, str(args.bundle))
    spec = importlib.util.spec_from_file_location('frozen_adni_entry', args.bundle/'entry.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import torch
    started = time.monotonic()
    initial_ledger = output/'budget_ledger.json'
    initial_charges = json.loads(initial_ledger.read_text()).get('charged_attempts', 0) if initial_ledger.exists() else 0
    stats = dict(pid=os.getpid(), ready=True, control_sha256=args.control_sha,
                 observed_full_run=not (output/'checkpoint_latest.pt').exists() and initial_charges == 0,
                 ledger_write_seconds=0., ledger_readback_seconds=0., checkpoint_seconds=0.)
    original_write, original_read = entry.write, entry.read
    ledger_write = timed_call(stats, 'ledger_write_seconds', original_write)
    ledger_read = timed_call(stats, 'ledger_readback_seconds', original_read)
    def write(path, value):
        return (ledger_write if Path(path).name == 'charged_ledger.json' else original_write)(path, value)
    def read(path):
        return (ledger_read if Path(path).name == 'charged_ledger.json' else original_read)(path)
    entry.write, entry.read = write, read
    class ObservedRecovery(entry.RecoveryWriter):
        def capture(self, force=False):
            # A stop before the first charge can create a zero-step checkpoint but
            # no ledger. There is no optimization work to upload or replay yet.
            # The scheduler audits any existing checkpoint before allowing entry.
            if initial_charges == 0 and not (self.output/'budget_ledger.json').exists():
                stats['uncharged_startup_snapshot_skipped'] = True
                return
            return super().capture(force=force)
    entry.RecoveryWriter = ObservedRecovery
    base = entry.scientific.experiment.base
    base.save_checkpoint = timed_call(stats, 'checkpoint_seconds', base.save_checkpoint)
    stopped = threading.Event()

    def snapshot():
        value = dict(stats, elapsed_seconds=time.monotonic()-started, updated_at=time.time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if torch.cuda.is_initialized():
            value['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            value['reserved_bytes'] = torch.cuda.memory_reserved()
        ledger = output/'budget_ledger.json'
        if ledger.exists():
            value['ledger'] = json.loads(ledger.read_text())
        return value

    def observe():
        while not stopped.wait(5):
            try:
                atomic_local(args.status, snapshot())
            except Exception as exc:
                # Observation failure never alters training or its accounting.
                stats['observation_error'] = repr(exc)
    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    sys.argv = [str(args.bundle/'entry.py'), *trainer_args]
    code = 0
    try:
        entry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        raise
    except BaseException:
        code = 1
        raise
    finally:
        stopped.set()
        thread.join()
        value = dict(snapshot(), exit_code=code,
                     completed=(output/'completion.json').exists() and code == 0)
        atomic_local(args.status, value)
        # Separate invocation files preserve prior measurements when resuming.
        atomic_local(output/f'concurrency_{os.getpid()}_{time.time_ns()}.json', value)


if __name__ == '__main__':
    main()
