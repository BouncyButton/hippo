"""Fill admissible worker slots; frozen ADNI science and recovery stay authoritative."""
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
import time
import threading
import tempfile
try:
    from .telemetry import LiveStatus, directories
    from . import local_storage, archive_verification, acceleration_gate
except ImportError:
    from telemetry import LiveStatus, directories
    import local_storage, archive_verification, acceleration_gate

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


def compatible_profile_controls(durable, scientific_sha, allowed_workers):
    """Reuse full-run peaks only from hash-verified, explicitly compatible observers."""
    accepted = set()
    if not allowed_workers:
        return accepted
    for name in sorted(directories(durable/'parallel')):
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
    ram = available_ram()
    return dict(util=util, used=used*1024**2, total=total*1024**2, ram=ram,
                used_gib=round(used/1024, 1), total_gib=round(total/1024, 1),
                ram_gib=round(ram/GIB, 1))


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
    if profile is None or profile.get('acceleration_observed') is False:
        budget = dict(gpu=max(12*GIB, budget['gpu']), ram=max(8*GIB, budget['ram']))
    return budget


def fits(candidate, active, profiles, sample):
    live_peaks = {field:max((j.get('status', {}).get(field, 0) for j in active
                           if j['task']['model'] == candidate['model']), default=0)
                  for field in ('peak_reserved_bytes', 'peak_rss_bytes')}
    wanted = task_reservation(candidate, profiles, live_peaks)
    future_gpu = future_ram = 0
    for job in active:
        status = job.get('status', {})
        budget = task_reservation(job['task'], profiles, status)
        future_gpu += max(0, budget['gpu']-status.get('reserved_bytes', 0))
        future_ram += max(0, budget['ram']-status.get('rss_bytes', 0))
    return (sample['total']-sample['used'] >= wanted['gpu']+future_gpu+max(8*GIB, .15*sample['total'])
            and sample['ram'] >= wanted['ram']+future_ram+4*GIB)



def has_work(root, durable, task):
    output = root/'work'/task['id']
    recovery = durable/'recovery'/task['id']
    return (any((output/name).exists() for name in ('budget_ledger.json', 'checkpoint_latest.pt',
                'completion.json', 'archive_manifest.json'))
            or any((recovery/name).exists() for name in ('charged_ledger.json', 'latest.json')))


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


class WorkMeter:
    """Observation only. Count returned optimizer work through task turnover.

    The scientific charge ledger remains authoritative and is never modified.
    Final worker counters are recorded before archiving/removing a worker.
    Startup, checkpoint and archive delays remain in elapsed wall time.
    """
    def __init__(self):
        self.seen, self.pending = {}, {}
        self.window = deque(maxlen=12)
        self.cohort = None
        self.last_tick = time.monotonic()

    def start(self, task, initial_flops=0):
        self.seen[task['id']] = initial_flops

    def record(self, task, status):
        value = status.get('completed_training_flops')
        if value is None:
            return
        previous = self.seen.get(task['id'], value)
        key = profile_key(task)
        self.pending[key] = self.pending.get(key, 0)+max(0, value-previous)
        self.seen[task['id']] = max(value, previous)

    def observe(self, now, keys):
        cohort = tuple(sorted(set(keys))) or self.cohort or ()
        if cohort != self.cohort:
            self.window.clear()
            self.cohort = cohort
        delta = sum(self.pending.get(key, 0) for key in cohort)
        self.pending.clear()
        self.window.append((delta, max(.001, now-self.last_tick)))
        self.last_tick = now
        return sum(v for v, _ in self.window)/sum(t for _, t in self.window)


class Sweep:
    def __init__(self, root, durable, control, max_workers, max_runs, hours):
        self.root, self.durable, self.control = root, durable, control
        self.r = load_runtime(root)
        self.control_sha, manifest = verify_control(control)
        self.profile_controls = {self.control_sha, *manifest.get('compatible_profile_control_sha256', [])}
        self.allowed_profile_workers = manifest.get('compatible_profile_worker_sha256', [])
        self.scientific_sha = manifest['scientific_bundle_sha256']
        self.acceleration_spec = manifest.get('acceleration')
        self.acceleration_receipt = None
        if sha(root/'bundle_manifest.json') != manifest['scientific_bundle_sha256']:
            raise RuntimeError('Companion does not match the frozen scientific bundle')
        self.limit = min(max_workers or cpu_capacity(), cpu_capacity())
        self.max_runs, self.deadline = max_runs, time.time()+hours*3600
        self.active, self.profiles, self.pending, self.rows = [], {}, [], []
        self.done_tasks, self.resumed = [], set()
        self.stop_reason = None
        self.interrupted = False
        self.errors = []
        self.launched = self.archived = 0
        self.meter = WorkMeter()
        self.status_dir = root/'parallel_status'
        self.status_dir.mkdir(exist_ok=True)
        self.session = str(time.time_ns())
        self.audit_dir = durable/'parallel'/self.session
        self.local_audit = self.status_dir/self.session
        self.live = LiveStatus(self.local_audit, provider=self.telemetry)
        self.archive_cache = self.local_audit/'verified_archives'
        self.lease = durable/'.adni_controller_lock'
        self.owns_lease = False
        self.latest_sample = None
        self.cpu_previous = None
        self.worker_progress = {}
        self.started_ids = set()
        self.baseline_bytes = {}
        self.work_store = None
        self.launch_wait_reason = None
        self.peak_workers = 0
        self.live.update(target=self.limit, ceiling=self.limit)
        self.stop_lock = threading.RLock()

    def request_stop(self, reason):
        with self.stop_lock:
            self._request_stop(reason)

    def _request_stop(self, reason):
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
        with self.live.phase('verify frozen scientific bundle on local SSD'):
            plan = self.r.verify_bundle(self.root)
        with self.live.phase('acquire Drive controller ownership'):
            try:
                self.lease.mkdir()
            except FileExistsError:
                raise RuntimeError('A Drive controller lock already exists. Do not run two runtimes against these results. '
                                   'If the prior VM died, its ownership and recovery need inspection; no automatic lock expiry.')
            self.owns_lease = True
            self.r.write(self.lease/'owner.json', dict(session=self.session, pid=os.getpid(),
                hostname=__import__('socket').gethostname(), started_at=time.time()))
        with self.live.phase('check environment binding and existing GPU preflight on Drive'):
            if self.acceleration_spec:
                value, self.acceleration_receipt = acceleration_gate.existing(
                    self.r, self.control, self.root, self.durable)
                self.r.write(self.local_audit/'acceleration_receipt.json', self.acceleration_receipt)
                print('Validated A100 acceleration enabled; original preflight and scientific recipe retained.', flush=True)
            else:
                value = self.r.binding(self.root, self.durable)
            self.binding_value = value
            if self.r.read(self.durable/'preflight.json') != dict(status='passed', **value):
                raise RuntimeError('Run the existing GPU preflight first; identical receipts are reused')
            self.r.write(self.local_audit/'environment_binding.json', value)
            self.r.write(self.local_audit/'controller.json', dict(control_sha256=self.control_sha,
                scientific_bundle_sha256=self.scientific_sha, worker_target=self.limit,
                max_new_runs=self.max_runs, session=self.session))
        with self.live.phase('list existing archive, recovery and local work directories once'):
            self.run_dirs = directories(self.durable/'runs')
            recovery_dirs = directories(self.durable/'recovery')
            local_dirs = directories(self.root/'work')
        with self.live.phase('verify compatible ADNI observer provenance'):
            self.profile_controls.update(compatible_profile_controls(
                self.durable, self.scientific_sha, self.allowed_profile_workers))
        existing = [t for t in plan['tasks'] if t['id'] in self.run_dirs]
        self.live.update(archive_total=len(existing), planned_total=len(plan['tasks']))
        for checked, task in enumerate(existing, 1):
            if self.interrupted:
                raise RuntimeError('Stopped during archive verification; all scientific work preserved')
            with self.live.phase('verify archived results from Drive', task['id']):
                with self.verified_snapshot(task) as local:
                    if local is not None:
                        self.consume_archive(local, task)
                        self.live.update(progress=True, verified=len(self.rows), completed=len(self.rows))
                self.live.update(progress=True, checked=checked)
            print(f"ARCHIVE CHECK {checked}/{len(existing)}; verified={len(self.rows)}; {task['id']}", flush=True)
        ids = {row['id'] for row in self.rows}
        self.done_tasks = [t for t in plan['tasks'] if t['id'] in ids]
        self.pending = [t for t in plan['tasks'] if t['id'] not in ids]
        # Only actual directories are considered. Empty/initialization-only directories
        # are cheap to audit and may safely be treated as resume candidates.
        self.resumed = {t['id'] for t in self.pending if t['id'] in local_dirs | recovery_dirs | self.run_dirs}
        with self.live.phase('select local disk for new checkpoints and recovery staging'):
            self.work_store = local_storage.choose_work_store(self.root, self.durable, self.scientific_sha)
            if self.work_store:
                physical_dirs = directories(self.work_store)
                self.resumed.update(t['id'] for t in self.pending if t['id'] in physical_dirs)
        with self.live.phase('save controller provenance and preservation inventory to Drive'):
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            manifest = self.r.read(self.control/'control_manifest.json')
            for name in (*manifest['files'], 'control_manifest.json'):
                self.r.copy_verified(self.control/name, self.audit_dir/name)
            if self.acceleration_receipt is not None:
                self.r.copy_verified(self.local_audit/'acceleration_receipt.json', self.audit_dir/'acceleration_receipt.json')
            self.r.write(self.audit_dir/'preservation.json', dict(existing_verified_ids=sorted(ids),
            resume_first=sorted(self.resumed), scientific_bundle_sha256=value['bundle_sha256'],
            control_sha256=self.control_sha, total_tasks=len(plan['tasks']), max_new_runs=self.max_runs,
            worker_ceiling=self.limit, augmentation='none',
            compatible_profile_controls=sorted(self.profile_controls),
            policy='fill_slots_no_solo_profiling', work_store=str(self.work_store), existing_work_moved=False))
        print(f'Preserved {len(ids)}/504 verified runs; {len(self.resumed)} tasks with existing work. '
              f'Worker target: {self.limit}; fill every slot that fits memory/storage. Same results, recipes and FLOP caps.', flush=True)

    def verified_snapshot(self, task):
        def progress(**fields):
            fields.pop('announce', None)
            fields.pop('current_task', None)
            self.live.update(**fields)
        return archive_verification.verified_snapshot(self.r, self.durable,
            self.archive_cache, task, self.binding_value, progress=progress)

    def consume_archive(self, local, task):
        """Parse only the bytes that just passed the unchanged full verifier."""
        self.rows.append(self.r.result_row(local, task))
        directory = local/'runs'/task['id']
        for path in directory.glob('concurrency_*.json'):
            info = self.r.read(path)
            if (info.get('completed') and info.get('observed_full_run')
                    and info.get('control_sha256') in self.profile_controls):
                self.learn(task, info)
        if task['arm'] == 'dice':
            # Session-only verified bytes retain the frozen baseline restoration
            # check without another Drive read. Never load these from a prior
            # session. The disposable filesystem snapshot is removed on exit.
            self.baseline_bytes[task['id']] = {
                str(p.relative_to(directory)):p.read_bytes()
                for p in directory.rglob('*') if p.is_file()}

    def restore_baseline(self, task):
        if task['arm'] == 'dice':
            return
        identity = f"ADNI_{task['model']}_dice_f{task['fold']}s{task['seed']}"
        if identity not in self.baseline_bytes:
            raise ValueError('A verified fresh matched baseline is required before constraints')
        with tempfile.TemporaryDirectory(prefix='baseline_', dir=self.archive_cache) as temporary:
            local = Path(temporary)
            self.r.write(local/'environment_binding.json', self.binding_value)
            for name, value in self.baseline_bytes[identity].items():
                path = local/'runs'/identity/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            # All hashes/identity/readability/source/cleanup gates run again on
            # the same verified bytes, then the frozen initialization check.
            self.r.restore_matched_baseline(self.root, local, self.root/'work', task)

    def learn(self, task, info):
        if any(not isinstance(info.get(field), (int, float)) or not math.isfinite(info[field])
               or info[field] <= 0 for field in ('peak_reserved_bytes', 'peak_rss_bytes')):
            return
        # Pre-acceleration records remain useful hints, but are not measurements
        # of the new full run. Reserve an extra GiB for caches and retain the
        # unknown-path floor until an accelerated full run actually completes.
        measured = (not self.acceleration_spec or info.get('acceleration', {}).get('candidate_sha256')
                    == self.acceleration_spec['candidate_sha256'])
        info = dict(info)
        if not measured:
            info['peak_reserved_bytes'] += GIB
        key = profile_key(task)
        old = self.profiles.get(key, {})
        self.profiles[key] = {field:max(old.get(field, 0), info[field])
            for field in ('peak_reserved_bytes', 'peak_rss_bytes')}
        self.profiles[key]['acceleration_observed'] = measured or old.get('acceleration_observed', False)

    def storage_ready(self, task):
        count = len(self.active)+1
        self.r.check_drive_headroom(self.durable)
        quota = self.r.read(self.durable/'storage_quota.json')
        drive_ok = 'limit' not in quota or int(quota['limit'])-int(quota['usage']) >= 8*GIB*count
        # Reserve remaining checkpoint/snapshot space on EACH occupied device,
        # including old runtime-disk jobs kept beside new NVMe jobs.
        disks = {}
        for item in [j['task'] for j in self.active]+[task]:
            destination = local_storage.output_location(self.root, item, self.work_store)
            path = destination if destination.exists() else destination.parent
            device = path.stat().st_dev
            previous = disks.get(device, (path, 0))
            disks[device] = (path, previous[1]+12*GIB)
        disk_ok = (all(shutil.disk_usage(path).free >= allowance for path, allowance in disks.values())
                   and shutil.disk_usage(self.root).free >= 4*GIB)
        if not self.active and not (drive_ok and disk_ok):
            raise RuntimeError('Insufficient Drive or runtime SSD headroom; existing work preserved')
        return drive_ok and disk_ok

    def launch(self, task):
        if task['id'] in self.started_ids or task not in self.pending:
            raise RuntimeError('Duplicate task launch refused: '+task['id'])
        output = local_storage.ensure_output(self.root, task, self.work_store)
        recovery = self.durable/'recovery'/task['id']
        if not output.exists() or not any(output.iterdir()):
            self.r.restore_recovery(output, recovery)
        output.mkdir(exist_ok=True)
        if (output/'completion.json').exists():
            self.archive(task, output, recovery)
        else:
            # A partial archive without its local artifacts must be recovered, not rerun.
            if task['id'] in self.run_dirs and (self.durable/'runs'/task['id']/'archive_manifest.json').exists():
                raise RuntimeError('Partial archive needs recovery before training: '+task['id'])
            verify_resume(self.r, output, recovery)
            local_storage.record_location(output, self.work_store)
            initial_ledger = self.r.read(output/'budget_ledger.json') if (output/'budget_ledger.json').exists() else {}
            # ADNI's frozen model builder checks this exact model/fold/seed baseline
            # even on resume. Restore only its verified initialization metadata.
            self.restore_baseline(task)
            if self.check_stop():
                return False
            # Recovery/Drive may have blocked while other jobs allocated memory
            # or failed. Recheck before spending compute on another task.
            if any(j['proc'].poll() is not None for j in self.active):
                self.wait_for_slot('handling finished workers and their archives')
                return False
            if not fits(task, self.active, self.profiles, self.observe()):
                self.wait_for_slot('GPU/RAM headroom changed during recovery reads')
                return False
            protocol = self.r.read(self.root/'protocols'/task['protocol']/'protocol.json')
            env = self.r.child_environment(self.root)
            with self.stop_lock:
                if self.stop_reason or self.interrupted:
                    return False
                control_file = self.status_dir/(task['id']+'.control.json')
                self.r.write(control_file, dict(pause=False))
                env.update(HIPPO_COLAB_RECOVERY=str(recovery), HIPPO_DEADLINE=str(self.deadline-60),
                           HIPPO_COLAB_BUNDLE=str(self.root), HIPPO_CONTROL=str(control_file))
                status_file = self.status_dir/(task['id']+'.json')
                status_file.unlink(missing_ok=True)
                command = [sys.executable, '-u', str(self.control/'worker.py'), '--bundle', str(self.root),
                    '--status', str(status_file), '--control-sha', self.control_sha]
                if self.acceleration_spec:
                    if self.acceleration_receipt is None:
                        raise RuntimeError('Acceleration evidence was not verified; refusing launch')
                    command += ['--acceleration-receipt', str(self.local_audit/'acceleration_receipt.json')]
                command += ['--', *trainer_arguments(self.root, task, protocol)]
                log = (output/'train.log').open('a')
                try:
                    proc = subprocess.Popen(command, cwd=self.root/'source', env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                self.active.append(dict(task=task, proc=proc, log=log, output=output, recovery=recovery,
                    status_file=status_file, control_file=control_file, status={}, started=time.time()))
                self.meter.start(task, initial_ledger.get('training_flops', 0))
                self.started_ids.add(task['id'])
                self.peak_workers = max(self.peak_workers, len(self.active))
                self.live.update(workers=len(self.active), peak_workers=self.peak_workers, target=self.limit)
                self.live.publish()
            print(('RESUME' if task['id'] in self.resumed else 'RUN'), task['id'],
                  f'PID={proc.pid} workers={len(self.active)}', flush=True)
            if profile_key(task) not in self.profiles:
                allowance = task_reservation(task, self.profiles)
                print(f"Provisional allowance: GPU={allowance['gpu']/GIB:.1f} GiB RAM={allowance['ram']/GIB:.1f} GiB; "
                      'full-run memory observations collected concurrently. No extra experiment.', flush=True)
        self.pending.remove(task)
        self.launched += 1
        return True

    def archive(self, task, output, recovery):
        with self.live.phase('archiving, verifying and recording cleanup on Drive', task['id']):
            self.r.archive_completed(self.root, output, self.durable, recovery, task)
            with self.verified_snapshot(task) as local:
                if local is None:
                    raise RuntimeError('Archive did not verify: '+task['id'])
                self.consume_archive(local, task)
        self.done_tasks.append(task)
        self.live.update(progress=True, completed=len(self.rows))
        with self.live.phase('save updated comparison on Drive', task['id']):
            self.r.report(self.root, self.durable, rows=self.rows)
        self.archived += 1
        print('ARCHIVED', task['id'], flush=True)

    def reap(self):
        for job in list(self.active):
            code = job['proc'].poll()
            if code is None:
                continue
            if job['status_file'].exists():
                job['status'] = self.r.read(job['status_file'])
                self.meter.record(job['task'], job['status'])
            job['log'].close()
            # Remove before archive; all remaining children are still drained on archive failure.
            self.active.remove(job)
            if code == 0:
                self.archive(job['task'], job['output'], job['recovery'])
            elif code == 75:
                self.request_stop('worker saved for continuation')
            else:
                self.request_stop('worker failure; preserving all active work')
                self.r.copy_verified(job['output']/'train.log',
                    self.durable/'failures'/job['task']['id']/'train.log')
                self.errors.append(job['task']['id']+f' failed (exit {code}); '+(job['output']/'train.log').read_text()[-2000:])

    def telemetry(self):
        """Called by the independent heartbeat thread; never touches Drive."""
        try:
            sample = gpu_sample()
            fields = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
            total, idle = sum(fields), fields[3]+fields[4]
            if self.cpu_previous:
                old_total, old_idle = self.cpu_previous
                sample['cpu_percent'] = round(100*(1-(idle-old_idle)/max(1,total-old_total)), 1)
            self.cpu_previous = total, idle
            sample['observed_at'] = time.time()
            self.latest_sample = sample
        except Exception as exc:
            self.live.update(observation_error=repr(exc))
            sample = self.latest_sample or {}
        progress = []
        for job in list(self.active):
            try:
                value = self.r.read(job['status_file'])
            except (FileNotFoundError, json.JSONDecodeError):
                value = dict(phase='starting', updated_at=job['started'])
            job['status'] = value
            signature = (value.get('completed_updates'), value.get('phase'), value.get('recovery_commits'))
            old = self.worker_progress.get(job['task']['id'])
            if signature != old:
                self.worker_progress[job['task']['id']] = signature
                self.live.update(progress=True)
            progress.append(dict(id=job['task']['id'], **value))
        self.live.update(gpu=sample, active=progress, workers=len(progress), target=self.limit,
                         ceiling=self.limit, completed=len(self.rows), profiles=self.profiles)
        # A controller blocked in Drive cannot delay a requested clean stop.
        if self.interrupted or time.time() >= self.deadline-120:
            self.request_stop('interrupt requested' if self.interrupted else 'session deadline')

    def observe(self):
        now = time.monotonic()
        keys = {profile_key(j['task']) for j in self.active}
        for job in self.active:
            # Read the local observer snapshot, including workers still starting.
            # Their startup time is part of real throughput, not a reason to
            # restart the entire measurement window.
            try:
                job['status'] = self.r.read(job['status_file'])
            except FileNotFoundError:
                pass
            status = job.get('status', {})
            self.meter.record(job['task'], status)
        rate = self.meter.observe(now, keys)
        signature = self.meter.cohort
        sample = gpu_sample()
        self.live.update(recent_completed_flops_per_second=rate,
                         target=self.limit, throughput_cohort=signature)
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
        self.live.update(launch_gate=reason or 'all available slots filled')

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
            self.wait_for_slot('checking Drive/local storage headroom for '+task['id'])
            with self.live.phase('check Drive quota and local disk headroom', task['id']):
                storage_ready = self.storage_ready(task)
            if not storage_ready:
                self.wait_for_slot('Drive/local storage headroom')
                return
            # Quota checks and recovery reads may take time. Respect a stop or a
            # worker exit observed after them, before another process is launched.
            if self.check_stop() or any(job['proc'].poll() is not None for job in self.active):
                return
            self.wait_for_slot(None)
            with self.live.phase('restore and audit recovery, baseline and ledgers', task['id']):
                if not self.launch(task):
                    return
            sample = self.observe()
        self.wait_for_slot('session run limit reached' if self.launched >= self.max_runs else
                           'no pending tasks; waiting for active workers' if not self.pending else None)

    def loop(self):
        self.live.start()
        try:
            self.run_loop()
        except BaseException as exc:
            if str(exc) not in self.errors:
                self.errors.append(str(exc))
            raise
        finally:
            self.live.update(phase='stopped' if not self.active else 'controller failed with active workers',
                             stop_reason=self.stop_reason, errors=self.errors, workers=len(self.active),
                             active=[dict(id=j['task']['id'], **j.get('status', {})) for j in self.active])
            try:
                if self.owns_lease and not self.active:
                    with self.live.phase('release controller ownership on Drive'):
                        owner = self.r.read(self.lease/'owner.json')
                        if owner['session'] != self.session:
                            raise RuntimeError('Controller ownership changed; lock preserved for inspection')
                        (self.lease/'owner.json').unlink()
                        self.lease.rmdir()
                        self.owns_lease = False
            finally:
                self.live.close()

    def run_loop(self):
        self.load()
        # Archive verification is not pending-run throughput. Per-run startup,
        # quota checks, recovery and archiving remain in the measured session.
        self.meter.last_tick = time.monotonic()
        self.live.update(phase='scheduling pending scientific runs')
        while (self.pending or self.active) and (self.max_runs > 0 or self.active):
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
        with self.live.phase('save final comparison and local telemetry to Drive'):
            self.r.report(self.root, self.durable, rows=self.rows)
            self.live.publish()
            for name in ('status.json', 'timing.jsonl'):
                if (self.local_audit/name).exists():
                    # Freeze a LOCAL copy before verified upload, while live reporting continues.
                    snapshot = self.local_audit/('final_'+name)
                    with self.live.lock:
                        shutil.copyfile(self.local_audit/name, snapshot)
                    self.r.copy_verified(snapshot, self.audit_dir/name)
        self.r.write(self.audit_dir/'session.json', dict(archived=self.archived, launched=self.launched,
            completed=len(self.rows), stop_reason=self.stop_reason, errors=self.errors,
            control_sha256=self.control_sha, profiles=self.profiles))
        if self.errors:
            raise RuntimeError('\n'.join(self.errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', choices=('run', 'report'), default='run')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=0, help='0: fill CPU-supported slots; a positive value caps the fixed target')
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
                      args.workers, 0 if args.command == 'report' else args.max_runs, args.hours)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda signum, frame: setattr(sweep, 'interrupted', True))
        sweep.loop()


if __name__ == '__main__':
    main()
