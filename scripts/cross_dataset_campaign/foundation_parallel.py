"""Two isolated GPU experiment workers with per-run archive/cleanup gates."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest, install_signals, stopping
from scripts.cross_dataset_campaign.storage import prepare, retire, quota


def step_prefix(tasks=1):
    return ['srun', '--exclusive', '--exact', '--nodes=1', f'--ntasks={tasks}',
        '--cpus-per-task=4', f'--mem={40*tasks}G', f'--gres=gpu:4g.40gb:{tasks}',
        '--gpus-per-task=4g.40gb:1', '--gpu-bind=single:1', '--kill-on-bad-exit=1']


def worker_command(stage, plan, task, base_path, lane):
    command = controller.command_for(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.foundation_gpu_worker'
    command += ['--record', str(stage/'runs'/task['id']/'gpu_worker.json')]
    return [*step_prefix(), 'env', f'HIPPO_GPU_WORKER={lane}', *command]


def verify_gpu_probe(stage, plan):
    job = os.environ['SLURM_JOB_ID']
    folder = stage/'gpu_probe'/job
    subprocess.run([*step_prefix(2), plan['foundation_python'], '-m',
        'scripts.cross_dataset_campaign.foundation_gpu_worker', '--probe-directory', str(folder)],
        cwd=stage/'source', check=True)
    records = [read(folder/f'{i}.json') for i in range(2)]
    if (any(r['status'] != 'passed' or r['visible_devices'] != 1 or r['slurm_job_id'] != job for r in records)
            or len({r['device_uuid'] for r in records}) != 2):
        raise RuntimeError('Two distinct isolated GPUs were not assigned')
    write(stage/'gpu_allocation.json', dict(status='passed', workers=records,
        manifest_sha256=digest(stage/'manifest.json'), job_id=job))


def next_task(stage, tasks):
    for task in tasks:
        path = stage/'runs'/task['id']/'cleanup.json'
        if not path.exists() or read(path).get('status') != 'complete':
            return task
    return None


def signal_workers(workers):
    for worker in workers.values():
        process = worker.get('process')
        if process is not None and process.poll() is None and not worker.get('signalled'):
            process.send_signal(signal.SIGUSR1)
            worker['signalled'] = True


def dispatch(stage, plan, base_path, publish):
    """Advance each dataset independently; each lane owns at most one GPU step."""
    groups = {i: [t for t in plan['tasks'] if t['dataset'] == dataset]
              for i, dataset in enumerate(plan['parallelism']['datasets'])}
    workers = {}
    reason = None
    failure = None
    try:
        while True:
            changed = False
            if reason is None and stopping():
                reason = 'paused' if read(stage/'control.json').get('pause') else 'continuing_at_time_limit'
                changed = True
            if reason:
                signal_workers(workers)
            for lane, worker in list(workers.items()):
                output = worker['output']
                process = worker.get('process')
                if worker['phase'] == 'running':
                    code = process.poll()
                    if code is None:
                        continue
                    worker['stream'].close()
                    if code == 75:
                        reason = reason or ('paused' if read(stage/'control.json').get('pause') else 'continuing_at_time_limit')
                        del workers[lane]; changed = True
                        continue
                    if code:
                        failure = failure or f"{worker['task']['id']}: trainer exited {code}"
                        reason = 'failed'; del workers[lane]; changed = True
                        continue
                    prepare(output, worker['task'])
                    worker.update(phase='waiting_for_verified_archive', archive_started=time.monotonic())
                    changed = True
                if worker['phase'] == 'waiting_for_verified_archive':
                    if (output/'archive_acknowledgment.json').exists():
                        retire(output)
                        del workers[lane]; changed = True
                    elif reason:
                        del workers[lane]; changed = True
                    elif time.monotonic()-worker['archive_started'] >= plan['archive_wait_timeout_seconds']:
                        reason = 'waiting_for_archive_offline'
                        del workers[lane]; changed = True
            if reason:
                signal_workers(workers)
                if not workers:
                    publish(reason, {})
                    if failure:
                        raise RuntimeError(failure)
                    return reason
            else:
                for lane, tasks in groups.items():
                    if lane in workers:
                        continue
                    task = next_task(stage, tasks)
                    if task is None:
                        continue
                    if quota()['free_gib'] < plan['storage']['minimum_free_gib']:
                        raise RuntimeError('Insufficient personal quota before concurrent experiment')
                    output = stage/'runs'/task['id']; output.mkdir(parents=True, exist_ok=True)
                    worker = dict(task=task, output=output)
                    if (output/'completion.json').exists():
                        prepare(output, task)
                        worker.update(phase='waiting_for_verified_archive', archive_started=time.monotonic())
                    else:
                        stream = (output/'train.log').open('a')
                        try:
                            process = subprocess.Popen(worker_command(stage, plan, task, base_path, lane),
                                cwd=stage/'source', stdout=stream, stderr=subprocess.STDOUT)
                        except Exception:
                            stream.close(); raise
                        worker.update(phase='running', process=process, stream=stream)
                    workers[lane] = worker; changed = True
                if not workers:
                    publish('complete', {})
                    return 'complete'
            if changed:
                publish(reason or 'running', workers)
            time.sleep(2)
    finally:
        # srun forwards USR1 to its task. Preserve checkpoint/resume on failures.
        signal_workers(workers)
        for worker in workers.values():
            process = worker.get('process')
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
            if worker.get('stream'):
                worker['stream'].close()


def run(stage, plan, ensure_successor, cancel_successor):
    install_signals()
    job = os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_CONTROL'] = str(stage/'control.json')
    os.environ['HIPPO_DEADLINE'] = str(min(time.time()+plan['graceful_stop_seconds'],
        float(os.environ.get('HIPPO_ALLOCATION_DEADLINE', 'inf'))))
    base = Path('/tmp')/f'hippo-sam3-{os.getuid()}-{job}.pt'
    def publish(status, workers):
        active = [w['task']['id'] for w in workers.values()]
        controller.summarize(stage, plan, status, active=active)
        write(stage/'parallel_status.json', dict(status=status, job_id=job, updated_at=time.time(),
            workers={str(i): dict(task=w['task']['id'], phase=w['phase']) for i,w in workers.items()}))
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(stage/'control.json').get('pause'):
            publish('paused', {}); return
        ensure_successor()
        try:
            verify_gpu_probe(stage, plan)
            if not base.exists():
                write(stage/'base_request.json', dict(job_id=job, destination=str(base), expected_sha256=plan['base_sha256']))
                publish('waiting_for_base_checkpoint', {})
                began = time.monotonic()
                while not base.exists() or not (stage/f'base_ready_{job}.json').exists():
                    if stopping():
                        status = 'paused' if read(stage/'control.json').get('pause') else 'continuing_at_time_limit'
                        publish(status, {})
                        if status == 'paused': cancel_successor()
                        return
                    if time.monotonic()-began >= plan['archive_wait_timeout_seconds']:
                        publish('waiting_for_base_offline', {}); cancel_successor(); return
                    time.sleep(5)
            if digest(base) != plan['base_sha256']:
                raise RuntimeError('Base checkpoint delivery hash mismatch')
            result = dispatch(stage, plan, base, publish)
            if result != 'continuing_at_time_limit':
                cancel_successor()
        except Exception:
            import traceback
            write(stage/'failure.json', dict(traceback=traceback.format_exc(), job_id=job, time=time.time()))
            publish('failed', {}); cancel_successor(); raise
        finally:
            if base.exists() and digest(base) == plan['base_sha256']:
                base.unlink()
