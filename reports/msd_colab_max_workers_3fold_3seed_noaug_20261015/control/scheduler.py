"""Keep all admissible worker slots occupied; frozen science/recovery stay authoritative."""
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
import subprocess
import sys
import threading
import time

try:
    from . import local_storage, archive_verification
except ImportError:  # Direct execution from the small Colab control package.
    import local_storage
    import archive_verification

GIB = 1024**3


class StartupStopped(Exception):
    """Stop startup without publishing a partial inventory or launching work."""


class StartupProgress:
    """Report slow startup even while the main thread waits for Drive I/O."""
    def __init__(self, path, interval=30):
        self.path, self.interval = path, interval
        self.started = self.changed = time.monotonic()
        self.last_progress = self.started
        self.announced_events = -1
        self.state = dict(phase='starting scheduler', current_task=None, current_file=None,
                          checked=0, verified=0, progress_events=0)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def update(self, phase=None, announce=True, progress=False, **fields):
        with self.lock:
            phase_changed = phase is not None and phase != self.state['phase']
            if phase is not None:
                self.state['phase'] = phase
                self.changed = time.monotonic()
            self.state.update(fields)
            now = time.monotonic()
            if progress or phase_changed:
                self.last_progress = now
                self.state['progress_events'] += 1
            value = dict(self.state, elapsed_seconds=round(now-self.started, 1),
                         phase_seconds=round(now-self.changed, 1), time=time.time(),
                         seconds_since_progress=round(now-self.last_progress, 1))
            # This status lives on runtime SSD, never on the Drive mount being checked.
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value)+'\n')
            temporary.replace(self.path)
            if announce:
                current = ' | '+value['current_task'] if value['current_task'] else ''
                if value.get('current_file'):
                    current += '/'+value['current_file']
                files = (f" | files={value['archive_files_copied']}/{value.get('archive_files_total') or '?'}"
                         if value.get('current_task') and 'archive_files_copied' in value else '')
                label = 'PROGRESS' if value['progress_events'] != self.announced_events else 'STILL WAITING'
                self.announced_events = value['progress_events']
                print(f"STARTUP {label}: {value['phase']} | checked={value['checked']}/{value.get('total', '?')}"
                      f" verified={value['verified']}{files} | elapsed={value['elapsed_seconds']:.0f}s"
                      f" last progress={value['seconds_since_progress']:.0f}s ago{current}", flush=True)

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            self.update()

    def __enter__(self):
        self.update()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=1)


def directory_names(path):
    """One listing instead of probing hundreds of absent paths on Drive FUSE."""
    try:
        with os.scandir(path) as entries:
            return {entry.name for entry in entries}
    except FileNotFoundError:
        return set()


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


def compatible_profile_controls(durable, scientific_sha, allowed_workers):
    """Reuse full-run peaks only from hash-verified, explicitly compatible observers."""
    accepted = set()
    if not allowed_workers:
        return accepted
    for name in sorted(directory_names(durable/'parallel')):
        folder = durable/'parallel'/name
        marker = folder/'control_manifest.json'
        if not marker.is_file():
            continue
        manifest = json.loads(marker.read_text())
        if (manifest.get('scientific_bundle_sha256') != scientific_sha
                or manifest.get('files', {}).get('worker.py') not in allowed_workers):
            continue
        control_sha, _ = verify_control(folder)
        accepted.add(control_sha)
    return accepted


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


def task_reservation(task, profiles, status=None):
    """Use provisional headroom for unknown arms, without a solo measurement run.

    These are scheduling allowances, not measured upper bounds. Full-run peaks
    can tighten known-arm allowances. Same-model and live peaks can enlarge them.
    """
    profile = profiles.get(profile_key(task))
    observed = [profile] if profile is not None else [
        p for key, p in profiles.items() if key.startswith(task['model']+'/')]
    if status:
        observed.append(status)
    peak = {name:max((p.get(name, 0) for p in observed), default=0)
            for name in ('peak_reserved_bytes', 'peak_rss_bytes')}
    budget = reservation(peak)
    if profile is None:
        budget = dict(gpu=max(12*GIB, budget['gpu']), ram=max(8*GIB, budget['ram']))
    return budget


def fits(candidate, active, profiles, sample):
    wanted = task_reservation(candidate, profiles)
    future_gpu = future_ram = 0
    for job in active:
        status = job.get('status', {})
        budget = task_reservation(job['task'], profiles, status)
        future_gpu += max(0, budget['gpu']-status.get('reserved_bytes', 0))
        future_ram += max(0, budget['ram']-status.get('peak_rss_bytes', 0))
    return (sample['total']-sample['used'] >= wanted['gpu']+future_gpu+max(8*GIB, .15*sample['total'])
            and sample['ram'] >= wanted['ram']+future_ram+4*GIB)


def has_work(root, durable, task, recovery_names=None):
    output = root/'work'/task['id']
    recovery = durable/'recovery'/task['id']
    return (any((output/name).exists() for name in ('budget_ledger.json', 'checkpoint_latest.pt',
                'completion.json', 'archive_manifest.json'))
            or ((recovery_names is None or task['id'] in recovery_names)
                and any((recovery/name).exists() for name in ('charged_ledger.json', 'latest.json'))))


def candidates(pending, resumed, done):
    """Prioritise recovery and baselines while filling otherwise unused slots."""
    existing = [task for task in pending if task['id'] in resumed]
    fresh = [task for task in pending if task['id'] not in resumed]
    baselines = [task for task in fresh if task['arm'] == 'dice']
    matched = {(task['model'], task['fold'], task['seed']) for task in done if task['arm']=='dice'}
    constraints = [task for task in fresh if task['arm'] != 'dice'
                   and (task['model'], task['fold'], task['seed']) in matched]
    return existing + baselines + constraints


def trainer_arguments(root, task, protocol):
    return ['--model', protocol['trainer_alias'], '--objective', task['arm'],
        '--fold', str(task['fold']), '--seed', str(task['seed']),
        '--pkl', str(root/'data/msd_hippocampus_full.pkl'), '--output', str(root/'work'/task['id']),
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


class Sweep:
    def __init__(self, root, durable, control, max_workers, max_runs, hours):
        self.root, self.durable, self.control = root, durable, control
        self.r = load_runtime(root)
        self.control_sha, manifest = verify_control(control)
        if sha(root/'bundle_manifest.json') != manifest['scientific_bundle_sha256']:
            raise RuntimeError('Companion does not match the frozen scientific bundle')
        self.scientific_sha = manifest['scientific_bundle_sha256']
        self.allowed_profile_workers = manifest.get('compatible_profile_worker_sha256', [])
        self.profile_controls = {self.control_sha}
        self.work_store = None
        self.limit = min(max_workers or cpu_capacity(), cpu_capacity())
        self.max_runs, self.deadline = max_runs, time.time()+hours*3600
        self.active, self.profiles, self.pending, self.rows = [], {}, [], []
        self.done_tasks, self.resumed = [], set()
        self.stop_reason = None
        self.interrupted = False
        self.errors, self.last_charge, self.rates = [], {}, deque(maxlen=12)
        self.launched = self.archived = 0
        self.launch_wait_reason = None
        self.status_dir = root/'parallel_status'
        self.status_dir.mkdir(exist_ok=True)
        self.archive_cache = self.status_dir/'archive_verification'
        self.session = str(time.time_ns())
        self.audit_dir = durable/'parallel'/self.session
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

    def startup_check(self):
        if self.interrupted:
            raise StartupStopped('interrupt requested')
        if time.time() >= self.deadline-120:
            raise StartupStopped('session deadline')

    def load(self):
        with StartupProgress(self.status_dir/'startup.json') as progress:
            self.load_verified_inventory(progress)

    def load_verified_inventory(self, progress):
        self.startup_check()
        progress.update('verifying local scientific bundle')
        plan = self.r.verify_bundle(self.root)
        self.startup_check()
        progress.update('checking GPU/software binding and saved preflight')
        value = self.r.binding(self.root, self.durable)
        self.binding_value = value
        if self.r.read(self.durable/'preflight.json') != dict(status='passed', **value):
            raise RuntimeError('Run the existing GPU preflight first; identical receipts are reused')
        self.startup_check()
        progress.update('checking compatible prior memory observations')
        self.profile_controls.update(compatible_profile_controls(
            self.durable, self.scientific_sha, self.allowed_profile_workers))
        self.startup_check()
        progress.update('listing saved run folders on Drive')
        run_names = directory_names(self.durable/'runs')
        saved = [task for task in plan['tasks'] if task['id'] in run_names]
        progress.update(f'verifying {len(saved)} saved run folders; GPU workers start after verification', total=len(saved))
        for index, task in enumerate(saved):
            self.startup_check()
            progress.update(announce=False, current_task=task['id'])
            with archive_verification.verified_snapshot(self.r, self.durable, self.archive_cache,
                    task, value, progress.update, self.startup_check) as local:
                if local is not None:
                    self.rows.append(self.r.result_row(local, task))
                    # Consume only metadata in the verified snapshot; no Drive rereads.
                    folder = local/'runs'/task['id']
                    metadata = self.r.read(folder/'archive_manifest.json')['metadata']
                    for name in metadata:
                        if Path(name).name == name and name.startswith('concurrency_') and name.endswith('.json'):
                            info = self.r.read(folder/name)
                            if (info.get('completed') and info.get('observed_full_run')
                                    and info.get('control_sha256') in self.profile_controls):
                                self.learn(task, info)
            progress.update(progress=True, checked=index+1, verified=len(self.rows), current_file=None)
        self.startup_check()
        ids = {row['id'] for row in self.rows}
        self.done_tasks = [t for t in plan['tasks'] if t['id'] in ids]
        self.pending = [t for t in plan['tasks'] if t['id'] not in ids]
        progress.update('checking existing recovery and local work', current_task=None)
        recovery_names = directory_names(self.durable/'recovery')
        for task in self.pending:
            self.startup_check()
            if has_work(self.root, self.durable, task, recovery_names):
                self.resumed.add(task['id'])
        self.startup_check()
        progress.update('saving verified inventory and scheduler provenance')
        self.work_store = local_storage.choose_work_store(self.root, self.durable, self.scientific_sha)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        control_files = self.r.read(self.control/'control_manifest.json')['files']
        for name in (*control_files, 'control_manifest.json'):
            self.r.copy_verified(self.control/name, self.audit_dir/name)
        self.r.write(self.audit_dir/'preservation.json', dict(existing_verified_ids=sorted(ids),
            resume_first=sorted(self.resumed), scientific_bundle_sha256=value['bundle_sha256'],
            control_sha256=self.control_sha, total_tasks=len(plan['tasks']), max_new_runs=self.max_runs,
            worker_ceiling=self.limit, worker_policy='fill_slots_no_solo_profiling',
            throughput_backoff=False, unknown_gpu_allowance_bytes=12*GIB,
            unknown_ram_allowance_bytes=8*GIB, augmentation='none',
            new_work_store=str(self.work_store) if self.work_store is not None else str(self.root/'work'),
            existing_work_moved=False, compatible_profile_controls=sorted(self.profile_controls),
            archive_verification='Fresh single-read Drive snapshots; unchanged full local verifier; no cross-session trust'))
        self.startup_check()
        self.r.report(self.root, self.durable, rows=self.rows)
        progress.update('startup verification complete', current_task=None)
        print(f'Preserved {len(ids)}/504 verified runs; {len(self.resumed)} tasks with existing work. '
              f'Worker target: {self.limit}; fill all slots that fit memory/storage. '
              'No solo profiling or throughput backoff. Same results, recipes and FLOP caps.', flush=True)

    def learn(self, task, info):
        if not info.get('peak_reserved_bytes') or not info.get('peak_rss_bytes'):
            return
        key = profile_key(task)
        old = self.profiles.get(key, {})
        self.profiles[key] = {field:max(old.get(field, 0), info[field])
            for field in ('peak_reserved_bytes', 'peak_rss_bytes')}

    def storage_ready(self, task):
        count = len(self.active)+1
        self.r.check_drive_headroom(self.durable)
        quota = self.r.read(self.durable/'storage_quota.json')
        drive_ok = 'limit' not in quota or int(quota['limit'])-int(quota['usage']) >= 8*GIB*count
        destination = local_storage.output_location(self.root, task, self.work_store)
        disk_ok = shutil.disk_usage(destination if destination.exists() else destination.parent).free >= 12*GIB*count
        if not self.active and not (drive_ok and disk_ok):
            raise RuntimeError('Insufficient Drive or runtime SSD headroom; existing work preserved')
        return drive_ok and disk_ok

    def launch(self, task):
        output = local_storage.ensure_output(self.root, task, self.work_store)
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
            if self.check_stop() or any(job['proc'].poll() is not None for job in self.active):
                return False
            local_storage.record_location(output, self.work_store)
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
                  f'PID={proc.pid} workers={len(self.active)}/{self.limit}', flush=True)
            if profile_key(task) not in self.profiles:
                budget = task_reservation(task, self.profiles)
                print(f"Using provisional allowance: GPU={budget['gpu']/GIB:.1f} GiB "
                      f"RAM={budget['ram']/GIB:.1f} GiB; memory observation runs concurrently.", flush=True)
        self.pending.remove(task)
        self.launched += 1
        return True

    def archive(self, task, output, recovery):
        self.r.archive_completed(self.root, output, self.durable, recovery, task)
        with archive_verification.verified_snapshot(self.r, self.durable, self.archive_cache,
                task, self.binding_value) as local:
            if local is None:
                raise RuntimeError('Archive did not verify: '+task['id'])
            self.rows.append(self.r.result_row(local, task))
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
        if now-self.last_display >= 60:
            self.last_display = now
            progress = [dict(id=j['task']['id'], **j['status']) for j in self.active]
            report = dict(time=time.time(), workers=len(self.active), target=self.limit,
                ceiling=self.limit, gpu=sample, recent_charged_flops_per_second=rate,
                completed=len(self.rows), active=progress, profiles=self.profiles,
                policy='fill_slots_no_solo_profiling', launch_wait_reason=self.launch_wait_reason)
            self.r.write(self.audit_dir/'status.json', report)
            with (self.audit_dir/'timing.jsonl').open('a') as stream:
                stream.write(json.dumps(report, allow_nan=False)+'\n')
            steps = ', '.join(f"{j['task']['id']}:{j['status'].get('ledger', {}).get('charged_attempts', '?')}"
                              for j in self.active)
            print(f"PROGRESS {len(self.rows)}/504; workers={len(self.active)}/{self.limit}; "
                  f"GPU={sample['util']:.0f}% memory={sample['used']/GIB:.1f}/{sample['total']/GIB:.0f} GiB; "
                  f"charged updates [{steps}]; waiting={self.launch_wait_reason or 'none'}", flush=True)
        return sample

    def check_stop(self):
        if self.interrupted:
            self.request_stop('interrupt requested')
        if time.time() >= self.deadline-120:
            self.request_stop('session deadline')
        if not self.stop_reason and (self.durable/'STOP').exists():
            self.request_stop('STOP file is present (kept until you deliberately remove it)')
        if self.stop_reason:
            self.request_stop(self.stop_reason)
        return self.stop_reason is not None

    def wait_for_slot(self, reason):
        if reason != self.launch_wait_reason and reason is not None:
            print(f'WAIT workers={len(self.active)}/{self.limit}: {reason}', flush=True)
        self.launch_wait_reason = reason

    def fill_slots(self, sample):
        """Refill promptly; recheck resources and stop/failure state between launches."""
        while self.pending and self.launched < self.max_runs and len(self.active) < self.limit:
            if self.check_stop():
                return
            if any(job['proc'].poll() is not None for job in self.active):
                self.wait_for_slot('handling finished workers and their archives')
                return  # Reap failures before spending more compute.
            eligible = candidates(self.pending, self.resumed, self.done_tasks)
            if not eligible:
                self.wait_for_slot('remaining constraints await their matched baseline archives')
                if not self.active:
                    raise RuntimeError('Pending constraints lack their matched baseline; no work was discarded')
                return
            task = next((t for t in eligible if fits(t, self.active, self.profiles, sample)), None)
            if task is None:
                self.wait_for_slot('GPU/RAM headroom including reserved future growth')
                if not self.active:
                    raise RuntimeError('No pending task fits the memory headroom policy; preserved for continuation')
                return
            if not self.storage_ready(task):
                self.wait_for_slot('Drive/local storage headroom')
                return
            # Quota checks and recovery reads may take time. Respect a stop or a
            # worker exit observed after them, before another process is launched.
            if self.check_stop() or any(job['proc'].poll() is not None for job in self.active):
                return
            self.wait_for_slot(None)
            if not self.launch(task):
                return
            sample = self.observe()
        self.wait_for_slot(None)

    def loop(self):
        try:
            self.load()
        except StartupStopped as exc:
            print(f'Stopped during startup: {exc}. No workers launched; existing results and recovery preserved.', flush=True)
            return
        while self.pending or self.active:
            try:
                self.check_stop()
                self.reap()
                self.check_stop()
                if self.stop_reason and not self.active:
                    break
                sample = self.observe()
                if not self.stop_reason:
                    self.fill_slots(sample)
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


def report_verified(root, durable, control):
    """Use the same full archive checks for the notebook's results cell, without training."""
    runtime = load_runtime(root)
    _, manifest = verify_control(control)
    if sha(root/'bundle_manifest.json') != manifest['scientific_bundle_sha256']:
        raise RuntimeError('Companion does not match the frozen scientific bundle')
    plan = runtime.verify_bundle(root)
    status = root/'parallel_status'
    status.mkdir(exist_ok=True)
    rows = []
    with StartupProgress(status/'report_verification.json') as progress:
        progress.update('listing saved run folders for report')
        saved_names = directory_names(durable/'runs')
        saved = [task for task in plan['tasks'] if task['id'] in saved_names]
        if saved:
            binding = runtime.read(durable/'environment_binding.json')
            if binding['bundle_sha256'] != manifest['scientific_bundle_sha256']:
                raise ValueError('Saved results belong to a different scientific bundle')
        progress.update('verifying archives for report', total=len(saved))
        for index, task in enumerate(saved, 1):
            with archive_verification.verified_snapshot(runtime, durable, status/'archive_verification',
                    task, binding, progress.update) as local:
                if local is not None:
                    rows.append(runtime.result_row(local, task))
            progress.update(progress=True, checked=index, verified=len(rows), current_file=None)
        # Publish only after all saved archives pass; retain previous report on any failure.
        return runtime.report(root, durable, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', choices=('run', 'report'), default='run')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=0, help='0: fill CPU-supported slots; positive value caps the target')
    parser.add_argument('--max-runs', type=int, default=504)
    parser.add_argument('--hours', type=float, default=22.)
    args = parser.parse_args()
    if args.workers < 0 or args.max_runs < 0 or not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('workers/max-runs must be nonnegative; hours must be positive and finite')
    print('STARTUP acquiring scheduler lock; verifying saved results before any training.', flush=True)
    # Resolving a Drive path can itself block; report startup before touching it.
    root, durable = args.bundle.resolve(), args.durable.resolve()
    work = root/'work'
    work.mkdir(exist_ok=True)
    with (work/'sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The original or another adaptive sweep is running. Stop it once and wait for recovery.')
        assert_no_orphans(root)
        if args.command == 'report':
            report_verified(root, durable, Path(__file__).resolve().parent)
            return
        sweep = Sweep(root, durable, Path(__file__).resolve().parent,
                      args.workers, args.max_runs, args.hours)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda signum, frame: setattr(sweep, 'interrupted', True))
        sweep.loop()


if __name__ == '__main__':
    main()
