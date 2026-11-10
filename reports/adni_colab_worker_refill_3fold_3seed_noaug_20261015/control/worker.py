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
from functools import wraps


def atomic_local(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def timed_call(stats, key, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            stats[key] = stats.get(key, 0.) + time.monotonic() - start
    return wrapped


def phase_call(stats, phase, key, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = stats.get('phase', 'starting')
        stats['phase'], stats['phase_started_at'] = phase, time.time()
        try:
            return timed_call(stats, key, function)(*args, **kwargs)
        finally:
            stats['phase'], stats['phase_started_at'] = previous, time.time()
    return wrapped


def instrument_trainer(entry, stats):
    """Install observation after the frozen binary recipe has bound its functions."""
    experiment = entry.scientific.experiment
    if not hasattr(experiment, 'trainer'):
        return  # Tiny CPU verification fixture, not a production trainer.
    original_main = experiment.trainer.main

    @wraps(original_main)
    def main():
        base, trainer = experiment.base, experiment.trainer
        for name, phase, key in [('_load_pkl_dataframe', 'loading local data', 'data_loading_seconds'),
                                  ('evaluate', 'evaluating', 'evaluation_seconds'),
                                  ('build_model', 'initializing model', 'initialization_seconds')]:
            setattr(base, name, phase_call(stats, phase, key, getattr(base, name)))
        dataset = base.PreparedPatients
        class ObservedDataset(dataset):
            __init__ = phase_call(stats, 'preparing local data', 'data_loading_seconds', dataset.__init__)
            __getitem__ = phase_call(stats, 'loading local batch', 'batch_loading_seconds', dataset.__getitem__)
        base.PreparedPatients = ObservedDataset
        trainer.calibrate = phase_call(stats, 'calibrating', 'calibration_seconds', trainer.calibrate)
        trainer.training_loss = phase_call(stats, 'training', 'loss_forward_seconds', trainer.training_loss)
        original_builder = base.build_loss_and_optimizer
        @wraps(original_builder)
        def builder(*args, **kwargs):
            criterion, optimizer = original_builder(*args, **kwargs)
            original_step = optimizer.step
            @wraps(original_step)
            def step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                # This is an observation after the original step returns; there is
                # no CUDA synchronize, extra tensor operation or random draw.
                ledger = stats.get('ledger', {})
                stats['completed_updates'] = ledger.get('charged_attempts', stats['completed_updates']+1)
                stats['completed_training_flops'] = ledger.get('training_flops', 0)
                stats['phase'] = 'training'
                return result
            optimizer.step = step
            return criterion, optimizer
        base.build_loss_and_optimizer = builder
        original_json = base.atomic_json
        @wraps(original_json)
        def atomic_json(path, value):
            result = original_json(path, value)
            if Path(path).name == 'config.json':
                stats['max_updates'] = value['max_steps']
            return result
        base.atomic_json = atomic_json
        stats['phase'] = 'training'
        return original_main()
    experiment.trainer.main = main


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
    stats = dict(pid=os.getpid(), ready=True, control_sha256=args.control_sha, phase='starting',
                 phase_started_at=time.time(), initial_updates=initial_charges, completed_updates=initial_charges,
                 observed_full_run=not (output/'checkpoint_latest.pt').exists() and initial_charges == 0,
                 ledger_write_seconds=0., ledger_readback_seconds=0., checkpoint_seconds=0.)
    original_write, original_read = entry.write, entry.read
    ledger_write = phase_call(stats, 'waiting on Drive ledger write', 'ledger_write_seconds', original_write)
    ledger_read = phase_call(stats, 'waiting on Drive ledger readback', 'ledger_readback_seconds', original_read)
    def write(path, value):
        result = (ledger_write if Path(path).name == 'charged_ledger.json' else original_write)(path, value)
        if Path(path).name == 'charged_ledger.json':
            stats['ledger'] = dict(value)
        return result
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
        def _persist(self, snapshot):
            stats['recovery_phase'] = 'uploading and verifying recovery on Drive'
            start = time.monotonic()
            try:
                result = super()._persist(snapshot)
                stats['recovery_commits'] = stats.get('recovery_commits', 0)+1
                return result
            finally:
                stats['recovery_upload_seconds'] = stats.get('recovery_upload_seconds', 0.)+time.monotonic()-start
                stats['recovery_phase'] = 'idle'
        def close(self, force=True):
            return phase_call(stats, 'waiting for verified recovery upload', 'recovery_close_seconds',
                              super().close)(force=force)
    entry.RecoveryWriter = ObservedRecovery
    base = entry.scientific.experiment.base
    base.save_checkpoint = phase_call(stats, 'checkpointing', 'checkpoint_seconds', base.save_checkpoint)
    instrument_trainer(entry, stats)
    stopped = threading.Event()

    def snapshot():
        value = dict(stats, elapsed_seconds=time.monotonic()-started, updated_at=time.time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                     cpu_seconds=time.process_time())
        if Path('/proc/self/status').exists():
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    value['rss_bytes'] = int(line.split()[1])*1024
        if torch.cuda.is_initialized():
            value['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            value['reserved_bytes'] = torch.cuda.memory_reserved()
            value['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            value['allocated_bytes'] = torch.cuda.memory_allocated()
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
        value['phase'] = ('finished; waiting for archive' if value['completed'] else
                          'paused; recovery upload finished' if code == 75 else
                          'failed; recovery retained' if code else 'exited without completion')
        atomic_local(args.status, value)
        # Separate invocation files preserve prior measurements when resuming.
        atomic_local(output/f'concurrency_{os.getpid()}_{time.time_ns()}.json', value)


if __name__ == '__main__':
    main()
