"""Filter scheduling only; all original per-run and recovery checks remain active."""
import argparse
import importlib.util
import fcntl
import os
from pathlib import Path
import signal
import sys

from campaign import comparison, filtered_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'report'))
    for name in ('bundle', 'durable', 'control', 'snapshot', 'other-stop'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--max-runs', type=int, required=True)
    parser.add_argument('--hours', type=float, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    parser.add_argument('--session-token', required=True)
    args = parser.parse_args()
    if args.seeds not in ([0, 2], [20]):
        raise ValueError('Only retained seeds 0/2 or extension seed20 are authorized')
    lock_stat = os.fstat(args.lock_fd)
    expected_lock = (args.bundle/'work/sweep.lock').stat()
    if (lock_stat.st_dev, lock_stat.st_ino) != (expected_lock.st_dev, expected_lock.st_ino):
        raise RuntimeError('Missing coordinator-owned scientific work lock')
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(args.control))
    spec = importlib.util.spec_from_file_location('seed_phase_scheduler', args.control/'scheduler.py')
    scheduler = importlib.util.module_from_spec(spec); spec.loader.exec_module(scheduler)
    scheduler.assert_no_orphans(args.bundle)
    sweep = scheduler.Sweep(args.bundle, args.durable, args.control, args.workers,
                            0 if args.command == 'report' else args.max_runs, args.hours)
    verify_original = sweep.r.verify_bundle
    verified_plan = None
    def verify(root):
        nonlocal verified_plan
        # Filtering happens only AFTER the complete frozen bundle is verified.
        verified_plan = filtered_plan(verify_original(root), args.seeds)
        return verified_plan
    sweep.r.verify_bundle = verify
    def report(root, durable, rows=None):
        if rows is None:
            raise ValueError('Phase reports require freshly verified scheduler rows')
        if verified_plan is None:
            raise RuntimeError('Frozen plan was not verified')
        value = comparison(rows, verified_plan, args.seeds,
                           dict(source_results=str(durable), source_bundle=sweep.scientific_sha))
        value.update(session_launched=sweep.launched, session_stop_reason=sweep.stop_reason,
                     session_token=args.session_token)
        sweep.r.write(args.snapshot, value)
        print(f"Verified requested seeds {args.seeds}: {value['completed']}/{value['total']}", flush=True)
        return value['runs'], value['aggregate']
    sweep.r.report = report
    original_stop = sweep.check_stop
    def check_stop():
        original_stop()
        if args.other_stop.exists():
            sweep.request_stop('STOP requested in companion results folder')
        return sweep.stop_reason is not None
    sweep.check_stop = check_stop
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: setattr(sweep, 'interrupted', True))
    startup_check = sweep.startup_check
    startup_stop = None
    def check_startup():
        nonlocal startup_stop
        try:
            startup_check()
        except scheduler.StartupStopped as exc:
            startup_stop = str(exc)
            raise
    sweep.startup_check = check_startup
    sweep.loop()
    if startup_stop:
        # Explicitly mark an incomplete verification, never reuse an older snapshot.
        sweep.r.write(args.snapshot, dict(runs=[], session_launched=0,
            session_stop_reason='startup stopped: '+startup_stop,
            session_token=args.session_token, verification_complete=False))


if __name__ == '__main__':
    main()
