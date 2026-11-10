"""Adaptive concurrent workers; frozen training, archives and recovery stay authoritative."""
import argparse
from collections import deque
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

GIB = 1024**3


def load_runtime(root):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('runtime', root/'runtime.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['runtime'] = module
    spec.loader.exec_module(module)
    module.activate(root)
    return module


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_control(control):
    manifest = json.loads((control/'control_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(control/name) != expected:
            raise ValueError('Scheduler package hash differs: '+name)
    return sha(control/'control_manifest.json'), manifest


def cpu_capacity():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 2
    return max(1, (cpus-2)//2)


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Cannot inspect available system RAM')


def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('This notebook requires exactly one visible GPU')
    util, used, total = map(float, rows[0].split(','))
    return dict(util=util, used=used*1024**2, total=total*1024**2, ram=available_ram())


def profile_key(task):
    return task['model']+'/'+task['arm']


def reservation(profile):
    # Peaks cover a completed training/calibration/evaluation run. Reserve growth
    # across folds plus CUDA context/workspace, never use current allocation alone.
    return dict(gpu=max(4*GIB, 2*profile['peak_reserved_bytes']+2*GIB),
                ram=max(4*GIB, 1.5*profile['peak_rss_bytes']+GIB))


def fits(candidate, active, profiles, sample):
    profile = profiles.get(profile_key(candidate))
    if profile is None:
        return not active and sample['total']-sample['used'] >= 16*GIB and sample['ram'] >= 8*GIB
    if any(profile_key(job['task']) not in profiles for job in active):
        return False  # An unmeasured condition gets its first real run alone.
    wanted = reservation(profile)
    future_gpu = future_ram = 0
    for job in active:
        budget = reservation(profiles[profile_key(job['task'])])
        status = job.get('status', {})
        future_gpu += max(0, budget['gpu']-status.get('reserved_bytes', 0))
        future_ram += max(0, budget['ram']-status.get('peak_rss_bytes', 0))
    return (sample['total']-sample['used'] >= wanted['gpu']+future_gpu+max(8*GIB, .15*sample['total'])
            and sample['ram'] >= wanted['ram']+future_ram+4*GIB)


def has_work(root, durable, task):
    output = root/'work'/task['id']
    recovery = durable/'recovery'/task['id']
    return (any((output/name).exists() for name in ('budget_ledger.json', 'checkpoint_latest.pt',
                'completion.json', 'archive_manifest.json'))
            or any((recovery/name).exists() for name in ('charged_ledger.json', 'latest.json')))


def candidates(pending, resumed, done):
    existing = [task for task in pending if task['id'] in resumed]
    if existing:
        return existing
    baselines = [task for task in pending if task['arm'] == 'dice']
    if baselines:
        return baselines
    matched = {(task['model'], task['fold'], task['seed']) for task in done if task['arm']=='dice'}
    return [task for task in pending if (task['model'], task['fold'], task['seed']) in matched]


def trainer_arguments(root, task, protocol):
    return ['--model', protocol['trainer_alias'], '--objective', task['arm'],
        '--fold', str(task['fold']), '--seed', str(task['seed']),
        '--pkl', str(root/'data/adni_hippocampus_full.pkl'), '--output', str(root/'work'/task['id']),
        '--protocol-dir', str(root/'protocols'/task['protocol']), '--resume', '--threads', '2']


def verify_resume(runtime, output, recovery):
    """Reject lost charged updates before spending more compute on an interrupted task."""
    ledger_file = output/'budget_ledger.json'
    ledger = runtime.read(ledger_file) if ledger_file.exists() else {}
    durable_ledger = recovery/'charged_ledger.json'
    if durable_ledger.exists() and runtime.read(durable_ledger) != ledger:
        raise RuntimeError('Local/durable ledgers differ; recovery audit required: '+output.name)
    checkpoint = output/'checkpoint_latest.pt'
    charged = ledger.get('charged_attempts', 0)
    if not checkpoint.exists():
        if charged or ledger.get('calibration_attempts', 0):
            raise RuntimeError('Charged work without a checkpoint; recovery audit required: '+output.name)
        return
    import torch
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False, mmap=True)
    if state['charged_attempts'] != charged or state['step'] != charged:
        raise RuntimeError('Checkpoint is behind charged updates; recovery audit required: '+output.name)
    if ledger.get('calibration_attempts', 0):
        calibration = output/'calibration.json'
        if not calibration.exists() or runtime.digest(calibration) != ledger.get('calibration_sha256'):
            raise RuntimeError('Charged calibration is incomplete; recovery audit required: '+output.name)


def assert_no_orphans(root):
    # flock prevents two controllers; this also catches workers orphaned by a killed controller.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            command = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        words = [part.decode(errors='replace') for part in command]
        if str(root/'entry.py') in words or (any(w.endswith('/worker.py') for w in words) and str(root) in words):
            raise RuntimeError(f'Existing trainer PID {path.parent.name} still owns this bundle. '
                               'Wait for the old run to stop cleanly; do not launch a second controller.')


class Governor:
    def __init__(self, limit):
        self.limit, self.target = limit, 1
        self.samples = deque(maxlen=12)
        self.changed = time.monotonic()
        self.reference_rate = None
        self.cooldown = 0

    def observe(self, now, util, rate, count):
        self.samples.append(util)
        if len(self.samples) < 12 or now-self.changed < 60 or count < self.target:
            return None
        average = statistics.mean(self.samples)
        # A sustained throughput drop after adding a worker backs off future launches.
        # Existing work is drained, never cancelled or replayed to tune utilization.
        if self.reference_rate and rate is not None and rate < .8*self.reference_rate and self.target > 1:
            self.target -= 1
            self.reference_rate = None
            self.changed, self.cooldown = now, now+180
            return 'Throughput declined; draining to '+str(self.target)+' workers'
        if average < 80 and self.target < self.limit and now >= self.cooldown:
            self.reference_rate = rate if rate and rate > 0 else None
            self.target += 1
            self.changed = now
            return 'Idle GPU capacity; target '+str(self.target)+' workers (subject to memory/storage checks)'
        return None


class Sweep:
    def __init__(self, root, durable, control, max_workers, max_runs, hours):
        self.root, self.durable, self.control = root, durable, control
        self.r = load_runtime(root)
        self.control_sha, manifest = verify_control(control)
        if sha(root/'bundle_manifest.json') != manifest['scientific_bundle_sha256']:
            raise RuntimeError('Companion does not match the frozen scientific bundle')
        self.limit = min(max_workers or cpu_capacity(), cpu_capacity())
        self.max_runs, self.deadline = max_runs, time.time()+hours*3600
        self.active, self.profiles, self.pending, self.rows = [], {}, [], []
        self.done_tasks, self.resumed = [], set()
        self.stop_reason = None
        self.interrupted = False
        self.errors, self.last_charge, self.rates = [], {}, deque(maxlen=12)
        self.launched = self.archived = 0
        self.governor = Governor(self.limit)
        self.status_dir = root/'parallel_status'
        self.status_dir.mkdir(exist_ok=True)
        self.session = str(time.time_ns())
        self.audit_dir = durable/'parallel'/self.session
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.last_display = 0.
        self.last_tick = time.monotonic()

    def request_stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
            print('Stopping launches:', reason, '— waiting for all workers and verified recovery.', flush=True)
        for job in self.active:
            if job['proc'].poll() is None and not job.get('stop_sent'):
                # The frozen trainer polls this at exactly the same safe boundaries
                # as its signal flag. A file also works during startup and avoids
                # signalling a process while its Python signal handlers shut down.
                self.r.write(job['control_file'], dict(pause=True))
                job['stop_sent'] = True

    def load(self):
        plan = self.r.verify_bundle(self.root)
        value = self.r.binding(self.root, self.durable)
        if self.r.read(self.durable/'preflight.json') != dict(status='passed', **value):
            raise RuntimeError('Run the existing GPU preflight first; identical receipts are reused')
        self.rows, _ = self.r.report(self.root, self.durable)
        ids = {row['id'] for row in self.rows}
        self.done_tasks = [t for t in plan['tasks'] if t['id'] in ids]
        self.pending = [t for t in plan['tasks'] if t['id'] not in ids]
        self.resumed = {t['id'] for t in self.pending if has_work(self.root, self.durable, t)}
        # Learn from completed, hash-verified archives, never from failed or partial peaks.
        for task in self.done_tasks:
            for path in (self.durable/'runs'/task['id']).glob('concurrency_*.json'):
                info = self.r.read(path)
                if info.get('completed') and info.get('observed_full_run') and info.get('control_sha256') == self.control_sha:
                    self.learn(task, info)
        for name in ('scheduler.py', 'worker.py', 'control_manifest.json'):
            self.r.copy_verified(self.control/name, self.audit_dir/name)
        self.r.write(self.audit_dir/'preservation.json', dict(existing_verified_ids=sorted(ids),
            resume_first=sorted(self.resumed), scientific_bundle_sha256=value['bundle_sha256'],
            control_sha256=self.control_sha, total_tasks=len(plan['tasks']), max_new_runs=self.max_runs,
            worker_ceiling=self.limit, augmentation='none'))
        print(f'Preserved {len(ids)}/504 verified runs; {len(self.resumed)} tasks with existing work. '
              f'Adaptive workers: 1–{self.limit}. Same results, recipes and FLOP caps.', flush=True)

    def learn(self, task, info):
        if not info.get('peak_reserved_bytes') or not info.get('peak_rss_bytes'):
            return
        key = profile_key(task)
        old = self.profiles.get(key, {})
        self.profiles[key] = {field:max(old.get(field, 0), info[field])
            for field in ('peak_reserved_bytes', 'peak_rss_bytes')}

    def storage_ready(self):
        count = len(self.active)+1
        self.r.check_drive_headroom(self.durable)
        quota = self.r.read(self.durable/'storage_quota.json')
        drive_ok = 'limit' not in quota or int(quota['limit'])-int(quota['usage']) >= 8*GIB*count
        disk_ok = shutil.disk_usage(self.root).free >= 12*GIB*count
        if not self.active and not (drive_ok and disk_ok):
            raise RuntimeError('Insufficient Drive or runtime SSD headroom; existing work preserved')
        return drive_ok and disk_ok

    def launch(self, task):
        output = self.root/'work'/task['id']
        recovery = self.durable/'recovery'/task['id']
        if not output.exists() or not any(output.iterdir()):
            self.r.restore_recovery(output, recovery)
        output.mkdir(exist_ok=True)
        if (output/'completion.json').exists():
            self.archive(task, output, recovery)
        else:
            # A partial archive without its local artifacts must be recovered, not rerun.
            if (self.durable/'runs'/task['id']/'archive_manifest.json').exists():
                raise RuntimeError('Partial archive needs recovery before training: '+task['id'])
            verify_resume(self.r, output, recovery)
            # ADNI's frozen model builder checks this exact model/fold/seed baseline
            # even on resume. Restore only its verified initialization metadata.
            self.r.restore_matched_baseline(self.root, self.durable, self.root/'work', task)
            protocol = self.r.read(self.root/'protocols'/task['protocol']/'protocol.json')
            env = self.r.child_environment(self.root)
            control_file = self.status_dir/(task['id']+'.control.json')
            self.r.write(control_file, dict(pause=False))
            env.update(HIPPO_COLAB_RECOVERY=str(recovery), HIPPO_DEADLINE=str(self.deadline-60),
                       HIPPO_COLAB_BUNDLE=str(self.root), HIPPO_CONTROL=str(control_file))
            status_file = self.status_dir/(task['id']+'.json')
            status_file.unlink(missing_ok=True)
            command = [sys.executable, '-u', str(self.control/'worker.py'), '--bundle', str(self.root),
                '--status', str(status_file), '--control-sha', self.control_sha, '--',
                *trainer_arguments(self.root, task, protocol)]
            log = (output/'train.log').open('a')
            try:
                proc = subprocess.Popen(command, cwd=self.root/'source', env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            self.active.append(dict(task=task, proc=proc, log=log, output=output, recovery=recovery,
                status_file=status_file, control_file=control_file, status={}, started=time.time()))
            print(('RESUME' if task['id'] in self.resumed else 'RUN'), task['id'],
                  f'PID={proc.pid} workers={len(self.active)}', flush=True)
            if profile_key(task) not in self.profiles:
                print('Measuring full memory peak with this pending run; it runs alone. No additional experiment.', flush=True)
        self.pending.remove(task)
        self.launched += 1

    def archive(self, task, output, recovery):
        self.r.archive_completed(self.root, output, self.durable, recovery, task)
        if not self.r.completed(self.durable, task):
            raise RuntimeError('Archive did not verify: '+task['id'])
        self.rows.append(self.r.result_row(self.durable, task))
        self.done_tasks.append(task)
        self.r.report(self.root, self.durable, rows=self.rows)
        self.archived += 1
        print('ARCHIVED', task['id'], flush=True)

    def reap(self):
        for job in list(self.active):
            code = job['proc'].poll()
            if code is None:
                continue
            job['log'].close()
            # Remove before archive; all remaining children are still drained on archive failure.
            self.active.remove(job)
            if code == 0:
                self.archive(job['task'], job['output'], job['recovery'])
                if job['status_file'].exists():
                    info = self.r.read(job['status_file'])
                    if info.get('completed') and info.get('observed_full_run'):
                        self.learn(job['task'], info)
            elif code == 75:
                self.request_stop('worker saved for continuation')
            else:
                self.r.copy_verified(job['output']/'train.log',
                    self.durable/'failures'/job['task']['id']/'train.log')
                self.errors.append(job['task']['id']+f' failed (exit {code}); '+(job['output']/'train.log').read_text()[-2000:])
                self.request_stop('worker failure; preserving all active work')

    def observe(self):
        now = time.monotonic()
        charged_delta = 0
        for job in self.active:
            if job['status_file'].exists():
                job['status'] = self.r.read(job['status_file'])
                value = job['status'].get('ledger', {}).get('training_flops', 0)
                previous = self.last_charge.get(job['task']['id'], value)
                charged_delta += max(0, value-previous)
                self.last_charge[job['task']['id']] = value
        self.rates.append((charged_delta, now-self.last_tick))
        self.last_tick = now
        rate = sum(v for v, _ in self.rates)/max(.001, sum(s for _, s in self.rates))
        sample = gpu_sample()
        message = self.governor.observe(now, sample['util'], rate, len(self.active))
        if message:
            print(message, flush=True)
        if now-self.last_display >= 60:
            self.last_display = now
            progress = [dict(id=j['task']['id'], **j['status']) for j in self.active]
            report = dict(time=time.time(), workers=len(self.active), target=self.governor.target,
                ceiling=self.limit, gpu=sample, recent_charged_flops_per_second=rate,
                completed=len(self.rows), active=progress, profiles=self.profiles)
            self.r.write(self.audit_dir/'status.json', report)
            with (self.audit_dir/'timing.jsonl').open('a') as stream:
                stream.write(json.dumps(report, allow_nan=False)+'\n')
            steps = ', '.join(f"{j['task']['id']}:{j['status'].get('ledger', {}).get('charged_attempts', '?')}"
                              for j in self.active)
            print(f"PROGRESS {len(self.rows)}/504; workers={len(self.active)}/{self.governor.target}; "
                  f"GPU={sample['util']:.0f}% memory={sample['used']/GIB:.1f}/{sample['total']/GIB:.0f} GiB; "
                  f"charged updates [{steps}]", flush=True)
        return sample

    def loop(self):
        self.load()
        while self.pending or self.active:
            try:
                if self.interrupted:
                    self.request_stop('interrupt requested')
                if self.stop_reason:
                    self.request_stop(self.stop_reason)
                self.reap()
                if time.time() >= self.deadline-120:
                    self.request_stop('session deadline')
                if (self.durable/'STOP').exists():
                    self.request_stop('STOP file is present (kept until you deliberately remove it)')
                if self.stop_reason and not self.active:
                    break
                sample = self.observe()
                if not self.stop_reason and self.launched < self.max_runs and len(self.active) < self.governor.target:
                    # Existing work has priority. Do not fill with new tasks while any resumed task is active.
                    eligible = candidates(self.pending, self.resumed, self.done_tasks)
                    if any(j['task']['id'] in self.resumed for j in self.active):
                        eligible = [t for t in eligible if t['id'] in self.resumed]
                    task = next((t for t in eligible if fits(t, self.active, self.profiles, sample)), None)
                    if task and self.storage_ready():
                        self.launch(task)
                    elif not self.active and eligible:
                        raise RuntimeError('No pending task fits the memory headroom policy; preserved for continuation')
                    elif not self.active and self.pending:
                        raise RuntimeError('Pending constraints lack their matched baseline; no work was discarded')
                if not self.active and (self.launched >= self.max_runs or not self.pending):
                    break
            except (KeyboardInterrupt, SystemExit):
                self.request_stop('interrupted')
            except Exception as exc:
                self.errors.append(str(exc))
                self.request_stop('error: '+str(exc))
                if not self.active:
                    break
            time.sleep(5)
        self.r.report(self.root, self.durable, rows=self.rows)
        self.r.write(self.audit_dir/'session.json', dict(archived=self.archived, launched=self.launched,
            completed=len(self.rows), stop_reason=self.stop_reason, errors=self.errors,
            control_sha256=self.control_sha, profiles=self.profiles))
        if self.errors:
            raise RuntimeError('\n'.join(self.errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=0, help='0: adaptive CPU-derived ceiling')
    parser.add_argument('--max-runs', type=int, default=504)
    parser.add_argument('--hours', type=float, default=22.)
    args = parser.parse_args()
    if args.workers < 0 or args.max_runs < 0 or not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('workers/max-runs must be nonnegative; hours must be positive and finite')
    root, durable = args.bundle.resolve(), args.durable.resolve()
    work = root/'work'
    work.mkdir(exist_ok=True)
    with (work/'sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The original or another adaptive sweep is running. Stop it once and wait for recovery.')
        assert_no_orphans(root)
        sweep = Sweep(root, durable, Path(__file__).resolve().parent,
                      args.workers, args.max_runs, args.hours)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda signum, frame: setattr(sweep, 'interrupted', True))
        sweep.loop()


if __name__ == '__main__':
    main()
