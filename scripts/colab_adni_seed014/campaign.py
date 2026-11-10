"""Two original-accounting sweeps, one seeds-0/1/4 comparison; no result migration."""
import argparse
from contextlib import ExitStack
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time


def read(path):
    return json.loads(path.read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@contextmanager
def ownership(path):
    """Persistent cross-session lease; never expire or remove another owner."""
    token = str(time.time_ns())+'_'+str(os.getpid())
    path.mkdir()  # Existing ownership always requires inspection.
    marker = path/'owner.json'
    marker.write_text(json.dumps(dict(session=token, pid=os.getpid())))
    try:
        yield
    finally:
        if read(marker).get('session') != token:
            raise RuntimeError('Controller ownership changed; marker preserved')
        marker.unlink(); path.rmdir()


def filtered_plan(plan, seeds):
    result = deepcopy(plan)
    result['tasks'] = [t for t in plan['tasks'] if t['seed'] in seeds]
    result['seeds'] = list(seeds)
    result['num_runs'] = len(result['tasks'])
    return result


def comparison(rows, plan, seeds, provenance):
    expected = {t['id']: t for t in plan['tasks']}
    seen = set()
    rows = deepcopy(rows)
    for r in rows:
        if r['id'] in seen or r['id'] not in expected or r['seed'] not in seeds:
            raise ValueError('Duplicate or out-of-scope comparison row')
        if any(r[k] != v for k, v in expected[r['id']].items()):
            raise ValueError('Comparison task identity differs')
        if not math.isfinite(r['dice']):
            raise ValueError('Nonfinite Dice')
        seen.add(r['id'])
    baseline = {(r['model'], r['fold'], r['seed']): r['dice'] for r in rows if r['arm'] == 'dice'}
    for r in rows:
        b = baseline.get((r['model'], r['fold'], r['seed']))
        r['gain_pp'] = None if b is None else r['dice'] - b
    aggregate = []
    for model in plan['models']:
        for arm in plan['arms']:
            selected = [r for r in rows if r['model'] == model and r['arm'] == arm]
            means = []
            for fold in plan['folds']:
                group = [r for r in selected if r['fold'] == fold]
                if {r['seed'] for r in group} == set(seeds):
                    means.append(statistics.mean(r['dice'] for r in group))
            complete = len(selected) == len(seeds)*len(plan['folds'])
            aggregate.append(dict(model=model, arm=arm, completed=len(selected),
                expected=len(seeds)*len(plan['folds']), complete_fold_means=means,
                mean_dice=statistics.mean(means) if complete else None,
                mean_gain_pp=statistics.mean(r['gain_pp'] for r in selected)
                    if complete and all(r['gain_pp'] is not None for r in selected) else None,
                sd_across_folds=statistics.stdev(means) if complete and len(means)>1 else None))
    return dict(completed=len(rows), total=len(expected), seeds=list(seeds), runs=rows,
                aggregate=aggregate, provenance=provenance,
                note='Seed 2 replaced after observing results. Seeds 0/1 reused; original seed2 retained separately. Exploratory, post-selection comparison; not an unbiased replication. No augmentation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    parser.add_argument('--parent-bundle', required=True, type=Path)
    parser.add_argument('--parent-results', required=True, type=Path)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--durable', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--max-runs', type=int, default=504)
    parser.add_argument('--hours', type=float, default=8.)
    args = parser.parse_args()
    if args.workers < 0 or args.max_runs < 0 or not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('Invalid resource/session limits')
    package = Path(__file__).resolve().parent
    # Both roots use separate scientific modules, so execute one phase per subprocess.
    # The phase CLI is implemented below; the coordinator holds both local locks.
    import subprocess
    deadline = time.time()+args.hours*3600
    args.durable.mkdir(parents=True, exist_ok=True)
    sources = [(args.parent_bundle, args.parent_results, package/'control01', (0, 1)),
               (args.bundle, args.durable, package/'control04', (4,))]
    with ExitStack() as stack:
        locks = {}
        for root, durable, control, seeds in sources:
            work = root/'work'; work.mkdir(exist_ok=True)
            lock = stack.enter_context((work/'sweep.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('An existing controller is active; stop it cleanly before switching')
            locks[root] = lock.fileno()
        stack.enter_context(ownership(args.durable/'.adni_seeds014_controller_lock'))
        combined_rows = []
        combined_tasks = []
        remaining = args.max_runs
        interrupted = False
        process = None
        def stop(signum, frame):
            nonlocal interrupted
            interrupted = True
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, stop)
        for index, (root, durable, control, seeds) in enumerate(sources):
            if interrupted or (args.parent_results/'STOP').exists() or (args.durable/'STOP').exists():
                print('STOP preserved; no further phase will start.', flush=True)
                return
            if index == 1:
                # Keep the old controller out across VMs while seed4 runs. Its
                # own phase has released this lease only after workers drained.
                stack.enter_context(ownership(args.parent_results/'.adni_controller_lock'))
            hours = max(.001, (deadline-time.time())/3600)
            snapshot = args.durable/('retained_seeds01.json' if index == 0 else 'seed4_snapshot.json')
            command = [sys.executable, str(package/'phase.py'), args.command,
                       '--bundle', str(root), '--durable', str(durable), '--control', str(control),
                       '--seeds', *map(str, seeds), '--snapshot', str(snapshot),
                       '--lock-fd', str(locks[root]),
                       '--workers', str(args.workers), '--max-runs', str(remaining), '--hours', str(hours),
                       '--other-stop', str(args.durable/'STOP' if index == 0 else args.parent_results/'STOP')]
            print('PHASE: verify/resume requested seeds', seeds, flush=True)
            process = subprocess.Popen(command, pass_fds=(locks[root],))
            code = process.wait(); process = None
            if code:
                raise RuntimeError(f'Seed phase exited {code}; all recovery retained. No next phase launched.')
            result = read(snapshot)
            combined_rows.extend(result['runs'])
            phase_plan = filtered_plan(read(root/'plan.json'), seeds)
            combined_tasks.extend(phase_plan['tasks'])
            remaining = max(0, remaining-result.get('session_launched', 0))
            if interrupted or result.get('session_stop_reason'):
                print('Clean stop:', result.get('session_stop_reason'), flush=True)
                return
        plan = dict(read(args.parent_bundle/'plan.json'), tasks=combined_tasks)
        value = comparison(combined_rows, plan, (0, 1, 4),
                           dict(original_results=str(args.parent_results), seed4_results=str(args.durable)))
        # All data just passed the full archive checks in both phases. Publish only
        # now; failed/partial verification never replaces the prior combined report.
        runtime = load(args.bundle/'runtime.py', 'seed4_report_runtime')
        runtime.write(args.durable/'comparison.json', value)
        print(f"Verified seeds 0/1/4 results: {value['completed']}/504", flush=True)


if __name__ == '__main__':
    main()
