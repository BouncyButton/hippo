"""Adaptive process supervisor for the unchanged, frozen 108-run Colab bundle.

Scheduling, telemetry and supplementary recovery live here. An optional audited
evaluation extension leaves training, task identities, optimizer and archives intact.
"""
import argparse
from collections import deque
from dataclasses import dataclass, field
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import uuid

FROZEN_MANIFEST = '716f2e774fb9de3140baf1077eabff75fc0dc144f9a9e6752ec620dc5ddd6ac2'
GIB = 1024**3


def transient_quota_error(error):
    """Retry quota reads only for rate limits and transient HTTP server errors."""
    status = getattr(getattr(error, 'resp', None), 'status', None)
    if status in (429, 500, 502, 503, 504):
        return True
    if status != 403:
        return False
    try:
        details = json.loads(error.content)['error']['errors']
        return any(item.get('reason') in ('rateLimitExceeded', 'userRateLimitExceeded')
                   for item in details)
    except (AttributeError, ValueError, KeyError, TypeError):
        return False


def load_runtime(root):
    spec = importlib.util.spec_from_file_location('medsam3_frozen_runtime', root/'runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    if runtime.digest(root/'bundle_manifest.json') != FROZEN_MANIFEST:
        raise ValueError('This supervisor requires the exact existing 108-run package.')
    runtime.activate(root)
    return runtime


class SweepLock:
    """Use the serial scheduler's lock too; never bypass an existing worker."""
    def __init__(self, root):
        self.path = Path(root)/'work/sweep.lock'
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError('The previous sweep is still active. Stop its cell once and wait for recovery upload.') from None
        return self

    def __exit__(self, *_):
        self.stream.close()


def gpu_snapshot():
    """Read resource telemetry without initializing a CUDA model in this process."""
    args = ['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits', '--id=0']
    result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=15)
    total, used, utilization = map(float, result.stdout.strip().split(','))
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                             '--format=csv,noheader,nounits', '--id=0'],
                            capture_output=True, text=True, check=True, timeout=15)
    processes = {}
    for line in result.stdout.splitlines():
        fields = line.split(',')
        if len(fields) == 2:
            try:
                processes[int(fields[0])] = float(fields[1]) / 1024
            except ValueError:
                pass
    import psutil
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 4
    return dict(total_gib=total/1024, used_gib=used/1024, utilization=utilization,
                processes=processes, host_free_gib=psutil.virtual_memory().available/GIB,
                cpu_workers=max(1, cpus//4))


def progress(runtime, output):
    ledger = output/'checkpoint_ledger.json'
    step = runtime.read(ledger)['step'] if ledger.exists() else 0
    evaluation = output/'evaluation.json'
    volumes = len(runtime.read(evaluation)['records']) if evaluation.exists() else 0
    phase = 'loading' if not ledger.exists() else 'training'
    if step == 200:
        phase = 'evaluating' if (output/'training_summary.json').exists() else 'saving'
    if (output/'completion.json').exists():
        phase = 'archiving'
    value = dict(step=step, volumes=volumes, phase=phase)
    detail = output/'evaluation_runtime.json'
    if phase == 'evaluating' and detail.exists():
        value['evaluation'] = runtime.read(detail)
    return value


@dataclass
class Worker:
    task: dict
    process: object
    log: object
    started: float
    initial: dict
    peak_gib: float = 0.
    state: dict = field(default_factory=dict)
    stop_sent: bool = False
    last_stop: float = -float('inf')

    def ready(self):
        return (self.state.get('step', 0) > self.initial['step']
                or self.state.get('volumes', 0) > self.initial['volumes'])


class AdaptiveLimit:
    """Probe real pending work, comparing stable homogeneous throughput windows.

    A mixed training/evaluation window cannot support a fair rate comparison.
    In that case low utilization and progress can justify a bounded extra worker.
    All observations and decisions are saved; no pilot run is thrown away.
    """
    def __init__(self, maximum=0, window=180):
        self.maximum = maximum
        self.window = window
        self.target = 1
        self.ceiling = None
        self.memory_per_worker = 16.
        self.samples = deque()
        self.rates = {}
        self.performance_limits = {}
        self.last_change = -float('inf')
        self.last_signature = None

    def resource_limit(self, gpu):
        cap = min(gpu['cpu_workers'], max(1, int((gpu['total_gib']-8)/self.memory_per_worker)))
        if self.maximum:
            cap = min(cap, self.maximum)
        if self.ceiling:
            cap = min(cap, self.ceiling)
        return max(1, cap)

    def can_launch(self, gpu, workers):
        # Reserve unseen peaks of current processes, not just their current VRAM.
        reserve = sum(max(0., self.memory_per_worker-gpu['processes'].get(w.process.pid, 0.)) for w in workers)
        return (len(workers) < min(self.target, self.resource_limit(gpu))
                and gpu['total_gib']-gpu['used_gib'] >= 8+self.memory_per_worker+reserve
                and gpu['host_free_gib'] >= 8+12*(1+sum(not w.ready() for w in workers))
                and all(w.ready() and time.monotonic()-w.started >= 60 for w in workers))

    def observe(self, now, gpu, workers):
        for w in workers:
            w.peak_gib = max(w.peak_gib, gpu['processes'].get(w.process.pid, 0.))
        # Never shrink the initial reservation until a worker has trained ten updates.
        trained = [w.peak_gib for w in workers if w.state.get('step', 0)-w.initial['step'] >= 10 and w.peak_gib > 0]
        if trained:
            self.memory_per_worker = max(14., 1.25*max(trained), self.memory_per_worker if self.target > 1 else 0)
        limit = self.resource_limit(gpu)
        if self.target > limit:
            self.target = limit
            return 'Resource limit reduced; existing workers finish or checkpoint safely.'
        signature = tuple(sorted((w.task['id'], w.state.get('phase')) for w in workers))
        if signature != self.last_signature:
            self.samples.clear()
            self.last_signature = signature
        if not workers or len(workers) != self.target or any(not w.ready() for w in workers):
            self.samples.clear()
            return None
        self.samples.append((now, sum(w.state['step'] for w in workers),
                             sum(w.state['volumes'] for w in workers), gpu['utilization']))
        if now-self.samples[0][0] < self.window or now-self.last_change < self.window:
            return None
        first = self.samples[0]
        dt = now-first[0]
        phases = {w.state['phase'] for w in workers}
        arms = tuple(sorted({w.task['arm'] for w in workers}))
        average_gpu = sum(s[3] for s in self.samples)/len(self.samples)
        key = None
        if phases == {'training'} and self.samples[-1][1]-first[1] >= 10:
            key, rate = ('training', arms), 60*(self.samples[-1][1]-first[1])/dt
        elif phases == {'evaluating'} and self.samples[-1][2]-first[2] >= 3:
            key, rate = ('evaluation', arms), 60*(self.samples[-1][2]-first[2])/dt
        else:
            rate = None
        reason = None
        if key is not None:
            previous = self.rates.get(key)
            measured_limit = self.performance_limits.get(key)
            if measured_limit is not None and self.target >= measured_limit:
                if self.target > measured_limit:
                    self.target = measured_limit
                    reason = f'Reusing measured {key[0]} limit of {self.target} workers as active runs finish.'
            elif previous and previous[0] < self.target and rate < previous[1]*1.08:
                self.target = previous[0]
                self.performance_limits[key] = previous[0]
                reason = f'No clear throughput gain ({rate:.2f} vs {previous[1]:.2f}/min); returning to {self.target} workers as runs finish.'
            else:
                self.rates[key] = (self.target, rate)
                if self.target < limit:
                    self.target += 1
                    reason = f'Measured {key[0]} throughput {rate:.2f}/min; trying {self.target} workers.'
        elif average_gpu < 80 and self.target < limit and (
                self.samples[-1][1] > first[1] or self.samples[-1][2] > first[2]):
            self.target += 1
            reason = f'Mixed phases, GPU average {average_gpu:.0f}%; trying {self.target} workers within resource reserves.'
        self.samples.clear()
        if reason:
            self.last_change = now
        return reason


def verify_recovery(runtime, output, recovery):
    """Confirm a clean stop uploaded the same ledger AND checkpoint as local SSD."""
    runtime.check_ledger(output, recovery)
    if not (output/'checkpoint_latest.pt').exists():
        return
    marker = runtime.read(recovery/'latest.json')
    archive = recovery/runtime.relative(marker['file'])
    if runtime.digest(archive) != marker['sha256']:
        raise ValueError('Recovery archive hash mismatch after clean stop')
    with tarfile.open(archive) as tar:
        ledger = json.load(tar.extractfile('checkpoint_ledger.json'))
        manifest = json.load(tar.extractfile('snapshot_manifest.json'))
    if (ledger != runtime.read(output/'checkpoint_ledger.json')
            or manifest['checkpoint_latest.pt'] != runtime.digest(output/'checkpoint_latest.pt')):
        raise ValueError('Drive recovery is older than the local checkpoint; preserve this VM for recovery.')


def save_supplement(runtime, root, durable, task):
    """Retain completed evaluation volumes and logs when stopping between runs."""
    output = root/'work'/task['id']
    if not (output/'checkpoint_latest.pt').exists():
        return
    target = durable/'parallel_resume'/task['id']
    files = {}
    for name in ('evaluation.json', 'training_history.json', 'training_summary.json', 'slab_schedule.json', 'runtime_recipe.json', 'train.log'):
        if (output/name).exists():
            runtime.copy_verified(output/name, target/name)
            files[name] = runtime.digest(target/name)
    runtime.write(target/'manifest.json', dict(bundle_sha256=FROZEN_MANIFEST, task=task,
        checkpoint_sha256=runtime.digest(output/'checkpoint_latest.pt'),
        ledger=runtime.read(output/'checkpoint_ledger.json'), files=files))


def restore_supplement(runtime, root, durable, task):
    target = durable/'parallel_resume'/task['id']
    if not (target/'manifest.json').exists():
        return
    marker = runtime.read(target/'manifest.json')
    output = root/'work'/task['id']
    if marker['task'] != task or marker['bundle_sha256'] != FROZEN_MANIFEST:
        raise ValueError('Supplementary recovery identity differs')
    if marker['ledger'] != runtime.read(output/'checkpoint_ledger.json'):
        return  # A newer local checkpoint wins. Never roll it back.
    if marker['checkpoint_sha256'] != runtime.digest(output/'checkpoint_latest.pt'):
        raise ValueError('Supplementary evaluation belongs to a different checkpoint')
    for name, sha in marker['files'].items():
        source = target/runtime.relative(name)
        if runtime.digest(source) != sha:
            raise ValueError('Supplementary recovery file changed')
        if not (output/name).exists():
            runtime.copy_verified(source, output/name)


def restore_training_history(runtime, output, recovery):
    """Recover a legacy omitted sidecar from the exact, ledger-bound checkpoint."""
    path = output/'training_history.json'
    checkpoint = output/'checkpoint_latest.pt'
    if path.exists() or not checkpoint.exists():
        return
    runtime.check_ledger(output, recovery)
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    ledger = runtime.read(output/'checkpoint_ledger.json')
    history = state['history']
    if (state['protocol_sha256'] != runtime.digest(output/'protocol.json')
            or state['protocol_sha256'] != ledger['protocol_sha256']
            or state['step'] != ledger['step'] or len(history) != state['step']
            or [row['update'] for row in history] != list(range(1, state['step']+1))):
        raise ValueError('Recovery checkpoint recipe, ledger and training history disagree')
    completion = output/'completion.json'
    if completion.exists() and runtime.read(completion)['checkpoint_sha256'] != runtime.digest(checkpoint):
        raise ValueError('Completed evaluation belongs to a different checkpoint')
    runtime.write(path, history)


def prepare_task(runtime, root, durable, task):
    output, recovery = root/'work'/task['id'], durable/'recovery'/task['id']
    if not output.exists() or not any(output.iterdir()):
        runtime.restore(output, recovery)
    output.mkdir(exist_ok=True)
    runtime.check_ledger(output, recovery)
    if (output/'checkpoint_latest.pt').exists():
        restore_supplement(runtime, root, durable, task)
        restore_training_history(runtime, output, recovery)
    return output, recovery


class Supervisor:
    def __init__(self, runtime, root, durable, maximum=0, max_runs=108, hours=22, poll=5, telemetry=gpu_snapshot,
                 fast_evaluation=False):
        self.rt, self.root, self.durable = runtime, Path(root), Path(durable)
        self.tuner, self.max_runs, self.poll, self.telemetry = AdaptiveLimit(maximum), max_runs, poll, telemetry
        self.deadline = time.time()+hours*3600
        self.active = {}
        self.finished = set()
        self.launched = 0
        self.stopping = False
        self.failure = None
        self.previous_handlers = {}
        self.session = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        self.record = self.durable/'parallel_sessions'/self.session
        self.last_display = -float('inf')
        self.fast_evaluation = fast_evaluation
        self.quota_retry_at = 0.
        self.quota_failures = 0

    def event(self, message, **data):
        print(message, flush=True)
        self.record.mkdir(parents=True, exist_ok=True)
        with (self.record/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), message=message, **data))+'\n')

    def request_stop(self, *_):
        self.stopping = True

    def stop_workers(self):
        for w in self.active.values():
            ready = self.root/'work'/w.task['id']/'supervisor_ready.json'
            if (w.process.poll() is None and time.monotonic()-w.last_stop >= 5
                    and ready.exists() and self.rt.read(ready)['pid'] == w.process.pid):
                w.process.send_signal(signal.SIGUSR1)
                w.stop_sent = True
                w.last_stop = time.monotonic()

    def launch(self, task):
        # Keep reaping/checkpointing active workers while a quota read backs off.
        # Never use an old quota receipt to admit a new worker.
        if self.stopping or time.monotonic() < self.quota_retry_at:
            return
        try:
            self.rt.quota(self.durable)
        except Exception as error:
            if not transient_quota_error(error) or self.quota_failures >= 10:
                raise
            self.quota_failures += 1
            delay = min(64., 2. ** self.quota_failures + uuid.uuid4().int % 1000 / 1000)
            self.quota_retry_at = time.monotonic() + delay
            self.event(f'Drive quota check temporarily unavailable; retrying in {delay:.1f}s. Active workers continue.',
                       quota_retry=self.quota_failures, retry_seconds=delay)
            return
        if self.quota_failures:
            self.event('Drive quota check recovered; continuing pending work.')
        self.quota_failures = 0
        self.quota_retry_at = 0.
        if self.stopping or (self.durable/'STOP').exists() or time.time() >= self.deadline-300:
            self.stopping = True
            return
        quota = self.rt.read(self.durable/'storage_quota.json')
        required = (8+2*(len(self.active)+1))*GIB
        if 'limit' in quota and int(quota['limit'])-int(quota['usage']) < required:
            raise RuntimeError('Drive headroom insufficient for parallel recovery. Existing checkpoints are retained.')
        import shutil
        if shutil.disk_usage(self.root).free < required:
            raise RuntimeError('Runtime SSD headroom insufficient for parallel recovery.')
        output, _ = prepare_task(self.rt, self.root, self.durable, task)
        args, env = self.rt.child(self.root, self.durable, 'train', '--task', task['id'], deadline=self.deadline-300)
        # A tiny launcher ignores stop requests only during Python/module startup.
        # The original engine then installs its own clean-stop handler unchanged.
        # This prevents SIGUSR1 from killing a just-started process before import.
        args = [args[0], str(Path(__file__).resolve()), '_worker', str(output/'supervisor_ready.json'), *args[1:]]
        # Cache directories are non-scientific; keep font cache writes independent.
        env['MPLCONFIGDIR'] = str(output/'matplotlib_cache')
        env['HIPPO_AUDITED_FAST_EVAL'] = '1' if self.fast_evaluation else '0'
        log = (output/'train.log').open('a')
        initial = progress(self.rt, output)
        try:
            process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.active[task['id']] = Worker(task, process, log, time.monotonic(), initial, state=initial)
        self.launched += 1
        self.event(f"RUN {task['id']} — resume update {initial['step']}/200, evaluated {initial['volumes']} volumes", pid=process.pid)

    def reap(self):
        for task_id, w in list(self.active.items()):
            code = w.process.poll()
            if code is None:
                continue
            w.log.close()
            del self.active[task_id]
            output, recovery = self.root/'work'/task_id, self.durable/'recovery'/task_id
            if code == 0:
                restore_training_history(self.rt, output, recovery)
                self.rt.archive_run(self.root, output, self.durable, recovery, w.task)
                self.finished.add(task_id)
                self.event('ARCHIVED '+task_id)
            elif code == 75:
                verify_recovery(self.rt, output, recovery)
                save_supplement(self.rt, self.root, self.durable, w.task)
                self.event('CHECKPOINT VERIFIED ON DRIVE '+task_id, **progress(self.rt, output))
                self.stopping = True
            else:
                self.rt.copy_verified(output/'train.log', self.durable/'failures'/task_id/'train.log')
                self.failure = f'{task_id} exited {code}. Recovery preserved. '+(output/'train.log').read_text()[-4000:]
                self.event('FAILED '+task_id, returncode=code)
                self.stopping = True

    def inspect(self, gpu):
        for task_id, w in self.active.items():
            w.state = progress(self.rt, self.root/'work'/task_id)
        now = time.monotonic()
        reason = self.tuner.observe(now, gpu, list(self.active.values()))
        if reason:
            self.event(reason)
        if now-self.last_display >= 30:
            workers = [dict(id=w.task['id'], pid=w.process.pid, peak_gib=round(w.peak_gib, 2), **w.state) for w in self.active.values()]
            record = dict(time=time.time(), session=self.session, completed=len(self.finished), total=108,
                target_workers=self.tuner.target, resource_limit=self.tuner.resource_limit(gpu),
                gpu={k: v for k, v in gpu.items() if k != 'processes'}, workers=workers, stopping=self.stopping)
            self.rt.write(self.durable/'parallel_status.json', record)
            with (self.record/'telemetry.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            self.event(f"GPU {gpu['utilization']:.0f}% | VRAM {gpu['used_gib']:.1f}/{gpu['total_gib']:.1f} GiB | workers {len(workers)}/{self.tuner.target} | archived {len(self.finished)}/108")
            for w in workers:
                detail = w.get('evaluation')
                suffix = (f" | {detail['backend']}: {detail['case']} slice {detail['slice']}/{detail['slices']}"
                          if detail else '')
                print(f"  {w['id']}: {w['phase']}, update {w['step']}/200, evaluated {w['volumes']} volumes{suffix}", flush=True)
            self.last_display = now
        if gpu['total_gib']-gpu['used_gib'] < 4 or gpu['host_free_gib'] < 4:
            self.event('Low memory headroom: saving all workers before continuing with a lower limit.')
            self.stopping = True
            self.rt.write(self.durable/'parallel_resource_limit.json', dict(max_workers=max(1, len(self.active)-1), reason='Low memory headroom'))

    def run(self):
        with SweepLock(self.root):
            plan = self.rt.verify_bundle(self.root)
            binding = self.rt.binding(self.root, self.durable)
            if self.rt.read(self.durable/'preflight.json') != dict(status='passed', **binding):
                raise RuntimeError('The existing GPU preflight must be passed in this exact environment.')
            for fold in self.rt.FOLDS:
                self.rt.fold_calibration(self.root, self.durable, fold)
            for task in plan['tasks']:
                if self.rt.completed(self.root, self.durable, task):
                    self.finished.add(task['id'])
                elif (self.root/'work'/task['id']/'completion.json').exists():
                    restore_training_history(self.rt, self.root/'work'/task['id'], self.durable/'recovery'/task['id'])
                    self.rt.archive_run(self.root, self.root/'work'/task['id'], self.durable, self.durable/'recovery'/task['id'], task)
                    self.finished.add(task['id'])
            # Interrupted local work takes priority, but all fresh baselines still finish first.
            tasks = sorted(enumerate(plan['tasks']), key=lambda it: (it[1]['arm'] != 'baseline',
                not (self.root/'work'/it[1]['id']/'checkpoint_latest.pt').exists()
                    and not (self.durable/'recovery'/it[1]['id']/'latest.json').exists(), it[0]))
            tasks = [t for _, t in tasks]
            limit_file = self.durable/'parallel_resource_limit.json'
            if limit_file.exists():
                self.tuner.ceiling = self.rt.read(limit_file)['max_workers']
            self.event(f'Preserved {len(self.finished)} verified completed runs. Same frozen engine, calibration and 108 task identities.', bundle_sha256=FROZEN_MANIFEST)
            if self.fast_evaluation:
                self.event('Evaluation feature reuse is enabled only after a checkpoint-specific numerical and speed check. Training remains unchanged.')
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
                self.previous_handlers[sig] = signal.signal(sig, self.request_stop)
            try:
                while True:
                    self.reap()
                    if (self.durable/'STOP').exists() or time.time() >= self.deadline-300:
                        self.stopping = True
                    if self.stopping:
                        self.stop_workers()
                    if not self.active and (self.stopping or self.launched >= self.max_runs or len(self.finished) == len(tasks)):
                        break
                    gpu = self.telemetry()
                    self.inspect(gpu)
                    if self.stopping:
                        self.stop_workers()
                    elif self.launched < self.max_runs and self.tuner.can_launch(gpu, list(self.active.values())):
                        pending = [t for t in tasks if t['id'] not in self.finished and t['id'] not in self.active]
                        baselines = [t for t in pending if t['arm'] == 'baseline']
                        if baselines:
                            self.launch(baselines[0])
                        elif not any(w.task['arm'] == 'baseline' for w in self.active.values()) and pending:
                            self.launch(pending[0])
                    elif not self.active and not self.stopping:
                        raise RuntimeError('Not enough free GPU/host memory for the first worker plus reserves. Existing work is preserved.')
                    time.sleep(self.poll)
            except BaseException as error:
                self.failure = self.failure or f'{type(error).__name__}: {error}'
                self.stopping = True
            finally:
                # Never release the sweep lock while an engine is still running.
                if self.active:
                    print('Saving every active worker to Drive. Keep this runtime connected until this cell returns.', flush=True)
                while self.active:
                    self.stop_workers()
                    try:
                        self.reap()
                    except Exception as error:
                        self.failure = self.failure or f'{type(error).__name__}: {error}'
                    if self.active:
                        time.sleep(self.poll)
                for sig, handler in self.previous_handlers.items():
                    signal.signal(sig, handler)
            result = self.rt.report(self.root, self.durable)
            self.rt.write(self.record/'result.json', dict(status='failed' if self.failure else 'complete' if result['new_completed'] == 108 else 'stopped',
                error=self.failure, new_completed=result['new_completed'], launched=self.launched, bundle_sha256=FROZEN_MANIFEST))
            if self.failure:
                raise RuntimeError(self.failure)
            self.event('All requested workers finished or stopped cleanly. Resume with the same Run cell and RESULTS folder.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--max-workers', type=int, default=0, help='0: automatic, bounded by VRAM, CPU and measured throughput')
    parser.add_argument('--max-runs', type=int, default=108)
    parser.add_argument('--hours', type=float, default=22)
    parser.add_argument('--fast-evaluation', action='store_true', help='Audited evaluation-only image/text feature reuse')
    args = parser.parse_args()
    if args.max_workers < 0 or not 1 <= args.max_runs <= 108 or args.hours <= 0:
        parser.error('Invalid workers, runs or session duration')
    root, durable = args.bundle.resolve(), args.durable.resolve()
    Supervisor(load_runtime(root), root, durable, args.max_workers, args.max_runs, args.hours,
               fast_evaluation=args.fast_evaluation).run()


def launch_frozen_worker(marker, argv):
    import runpy
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    marker = Path(marker)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(pid=os.getpid())))
    temporary.replace(marker)
    sys.argv = argv
    sys.path.insert(0, str(Path(argv[0]).resolve().parent))
    if os.environ.get('HIPPO_AUDITED_FAST_EVAL') == '1':
        spec = importlib.util.spec_from_file_location('medsam3_frozen_engine', argv[0])
        engine = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = engine
        spec.loader.exec_module(engine)
        from fast_eval import install
        durable = Path(argv[argv.index('--durable')+1])
        install(engine, durable)
        engine.main()
    else:
        runpy.run_path(argv[0], run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        launch_frozen_worker(sys.argv[2], sys.argv[3:])
    else:
        main()
