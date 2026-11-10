"""Two Slurm GPU workers around the unchanged SAM3 126-run recipe."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

PYTHON = '/mnt/beegfsstudents/home/3160552/medsam3_runtime_20260928/venv/bin/python'
BUNDLE_SHA = 'bd128e499273277477f204a8d182906b669b377eae008c6d169d8ad4fe0c69cd'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def activate(stage):
    root = stage/'bundle'
    sys.path.insert(0, str(root))
    import runtime as rt
    rt.activate(root)
    return root, rt


def assignments(stage):
    assignment = read(stage/'assignment.json')
    ids = list(assignment['cluster_tasks'])
    handoff = stage/'colab_handoff.json'
    if handoff.exists():
        value = read(handoff)
        reserved = set(assignment['colab_tasks'])
        extra = set(value['additional_task_ids'])
        done = set(value['colab_completed'])
        if (extra & done or extra | done != reserved or
                value['assignment_sha256'] != digest(stage/'assignment.json') or
                time.time() < assignment['deadline_epoch']):
            raise ValueError('Invalid or premature Colab handoff')
        ids.extend(t for t in assignment['colab_tasks'] if t in extra)
    if len(ids) != len(set(ids)) or set(ids) & set(assignment.get('completed_ids', [])):
        raise ValueError('Duplicate or already completed assignment')
    return ids


def blocked_tasks(stage):
    path = stage/'colab_handoff.json'
    if not path.exists():
        return {}
    handoff = read(path)
    blocked = handoff.get('blocked_tasks', {})
    if not set(blocked).issubset(handoff['additional_task_ids']):
        raise ValueError('Blocked task is not owned by this handoff')
    return blocked


class LocalTransport:
    """Per-task persistent ledgers; an off-BeeGFS receipt gates retirement."""
    def __init__(self, stage, root, task, binding, calibration_lock=None):
        self.stage, self.root, self.task = stage, root, task
        self.source = digest(root/'bundle_manifest.json')
        self.state_path = stage/'task_state'/(task['id']+'.json')
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            completed={}, charges={}, objects={}, binding=binding, bundle_sha256=self.source)
        if self.state['binding'] != binding or self.state['bundle_sha256'] != self.source:
            raise ValueError('Worker environment or scientific source changed during resume')
        self.calibration_lock = calibration_lock
        self.save()

    def save(self):
        write(self.state_path, self.state)

    def charge(self, name, value):
        self.state['charges'][name] = value
        self.save()

    def restore(self, key):
        # Recovery is the atomic checkpoint on persistent BeeGFS, never a
        # volatile-node filesystem. restore_run still validates charge equality.
        return False

    def release_calibration(self):
        if self.calibration_lock is not None:
            fcntl.flock(self.calibration_lock, fcntl.LOCK_UN)
            self.calibration_lock.close(); self.calibration_lock = None

    def publish(self, key, paths):
        if key.startswith('calibration/'):
            self.release_calibration()
        if not key.startswith('completed/'):
            return {}
        from scripts.cross_dataset_campaign.storage import prepare
        from scripts.cross_dataset_campaign.common import stopping
        output = self.root/'work'/self.task['id']
        old = output/'archive_manifest.json'
        if old.exists() and 'metadata' not in read(old):
            old.replace(output/'replication_archive_manifest.json')
        if not old.exists():
            cell = f'f{self.task["fold"]}s{self.task["seed"]}'
            for source, name in ((self.root/'durable/calibration'/cell, 'shared_calibration'),
                                 (self.root/'durable/probes'/self.task['id'], 'probe_snapshots')):
                if source.exists():
                    shutil.copytree(source, output/name, dirs_exist_ok=True)
            write(output/'execution_provenance.json', dict(platform='bocconi-cluster',
                source_bundle_sha256=self.source, adapter_sha256=digest(Path(__file__)),
                assignment_sha256=digest(self.stage/'assignment.json'),
                hardware_binding=self.state['binding'],
                colab_binding=read(self.stage/'colab_binding.json'),
                policy='Unchanged recipe, data, weights and update budget; cluster paths and GPU binding recorded separately.'))
        prepare(output, dict(self.task, kind='foundation', dataset='GeoFWI3D', model='SAM3'))
        began = time.monotonic()
        while not (output/'archive_acknowledgment.json').exists():
            if stopping() or time.monotonic()-began > 600:
                write(self.stage/'archive_wait.json', dict(task=self.task['id'], time=time.time()))
                raise SystemExit(75)
            time.sleep(5)
        return {'id': self.task['id'], 'local_sha256': digest(output/'archive_manifest.json')}

    def retire(self, name):
        from scripts.cross_dataset_campaign.storage import retire
        if name not in self.state['completed']:
            raise ValueError('Cannot retire an incomplete run')
        retire(self.root/'work'/name)
        self.state['charges'].pop(name, None)
        self.save()


def worker(stage, name):
    root, rt = activate(stage)
    import replication
    from scripts.cross_dataset_campaign.common import install_signals
    install_signals()
    if name not in assignments(stage):
        raise ValueError('Task is not owned by the cluster')
    if name in blocked_tasks(stage):
        raise ValueError('Task requires recovery audit; charged updates must not be replayed')
    task = next(t for t in replication.tasks() if t['id'] == name)
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker must have exactly one GPU')
    props = torch.cuda.get_device_properties(0)
    out = root/'work'/name; out.mkdir(parents=True, exist_ok=True)
    write(out/'gpu_worker.json', dict(job_id=os.environ['SLURM_JOB_ID'],
        device_uuid=str(props.uuid), gpu=props.name, memory=props.total_memory,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), time=time.time()))
    (root/'durable').mkdir(exist_ok=True)
    with (stage/'binding.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binding = rt.binding(root, root/'durable')
    reference = read(stage/'colab_binding.json')['environment']
    for package, version in reference['packages'].items():
        if binding['environment']['packages'][package].split('+')[0] != version.split('+')[0]:
            raise ValueError('Scientific package version changed: '+package)
    if binding['environment']['cuda'] != reference['cuda'] or binding['environment']['cudnn'] != reference['cudnn']:
        raise ValueError('CUDA/cuDNN version differs from Colab')
    cell = f'f{task["fold"]}s{task["seed"]}'
    lock = (stage/f'calibration_{cell}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (root/'durable/calibration'/cell/'calibration.json').exists():
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close(); lock = None
    transport = LocalTransport(stage, root, task, binding, lock)
    try:
        if (out/'cleanup.json').exists() and read(out/'cleanup.json')['status'] == 'complete':
            return
        if ((out/'completion.json').exists() and (out/'archive_manifest.json').exists()
                and 'metadata' in read(out/'archive_manifest.json')):
            replication.validate_complete(rt, out, rt.read(out/'protocol.json'))
            transport.publish('completed/'+name, [])
            transport.state['completed'][name] = rt.read(out/'completion.json')
            transport.save(); transport.retire(name)
            return
        replication.run_one(root, task, transport)
    finally:
        transport.release_calibration()


def step_prefix(tasks=1):
    return ['srun', '--exclusive', '--exact', '--nodes=1', f'--ntasks={tasks}',
        '--cpus-per-task=4', f'--mem={40*tasks}G', f'--gres=gpu:4g.40gb:{tasks}',
        '--gpus-per-task=4g.40gb:1', '--gpu-bind=single:1', '--kill-on-bad-exit=1']


def verified_retirement(stage, name):
    """A launcher error cannot invalidate an already verified, archived run."""
    from scripts.cross_dataset_campaign.storage import verify_archive
    output = stage/'bundle/work'/name
    try:
        cleanup = read(output/'cleanup.json')
        ack = read(output/'archive_acknowledgment.json')
        manifest = read(output/'archive_manifest.json')
        done = read(output/'completion.json')
        state = read(stage/'task_state'/(name+'.json'))
        if (cleanup['status'] != 'complete' or ack['status'] != 'verified'
                or not ack['source_verified']
                or ack['manifest_sha256'] != digest(output/'archive_manifest.json')
                or manifest['task']['id'] != name
                or manifest['completion_sha256'] != digest(output/'completion.json')
                or done['status'] != 'complete'
                or state['completed'].get(name) != done
                or name in state['charges']):
            return False
        verify_archive(output, manifest)
        return True
    except (OSError, ValueError, KeyError):
        return False


def queue():
    text = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%T'], text=True)
    return [line.split('|') for line in text.splitlines() if line.strip()]


def submit(stage, dependency=None):
    from scripts.cross_dataset_campaign.storage import quota
    q = quota()
    if q['free_gib'] < 12:
        raise RuntimeError('Need 12 GiB personal BeeGFS headroom')
    active = queue()
    if sum(state == 'PENDING' for _, state in active) >= 1 or sum(state == 'RUNNING' for _, state in active) > 1:
        raise RuntimeError('Respect at most one running and one queued job')
    running = [job for job, state in active if state == 'RUNNING']
    dep = dependency or (running[0] if running else None)
    cmd = ['sbatch', '--parsable', '--chdir='+str(stage),
           '--output='+str(stage/'slurm-%j.out'), '--error='+str(stage/'slurm-%j.err')]
    if dep:
        cmd.append('--dependency=afterany:'+str(dep))
    cmd += [str(stage/'run.sbatch'), str(stage)]
    job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Invalid Slurm submission response')
    write(stage/'submissions'/(job+'.json'), dict(job_id=job, dependency=dep,
        quota=q, command=cmd, submitted_at=time.time(), manifest_sha256=digest(stage/'manifest.json')))
    return job


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Deployment source differs: '+name)
    root, _ = activate(stage)
    if digest(root/'bundle_manifest.json') != BUNDLE_SHA:
        raise ValueError('Scientific bundle differs')
    import replication
    replication.verify(root)


def run(stage):
    root, _ = activate(stage)
    from scripts.cross_dataset_campaign.common import install_signals, stopping
    from scripts.cross_dataset_campaign.storage import quota
    install_signals()
    os.environ['HIPPO_CONTROL'] = str(stage/'control.json')
    os.environ['HIPPO_DEADLINE'] = str(time.time()+23.5*3600)
    job = os.environ['SLURM_JOB_ID']
    active = {}
    successor = None
    def publish(status):
        ids = assignments(stage)
        completed = {n: read(root/'work'/n/'completion.json') for n in ids
            if (root/'work'/n/'cleanup.json').exists() and read(root/'work'/n/'cleanup.json')['status'] == 'complete'}
        write(stage/'status.json', dict(status=status, job_id=job, updated_at=time.time(),
            completed=len(completed), total=len(ids), results=completed,
            active={str(lane): item['id'] for lane,item in active.items()},
            blocked_tasks=blocked_tasks(stage),
            awaiting_colab_handoff=not (stage/'colab_handoff.json').exists()))
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(stage)
        if read(stage/'control.json').get('pause'):
            publish('paused'); return
        probe = stage/'gpu_probe'/job
        probe_cmd = step_prefix(2)+[PYTHON, '-m', 'scripts.cross_dataset_campaign.foundation_gpu_worker', '--probe-directory', str(probe)]
        subprocess.run(probe_cmd, check=True)
        receipts = [read(probe/f'{i}.json') for i in range(2)]
        if len({r['device_uuid'] for r in receipts}) != 2:
            raise ValueError('GPU workers did not receive distinct devices')
        write(stage/'gpu_allocation.json', dict(job_id=job, workers=receipts))
        successor = submit(stage, dependency=job)
        write(stage/'successor.json', dict(job_id=successor, predecessor=job))
        try:
            while True:
                for lane, item in list(active.items()):
                    code = item['process'].poll()
                    if code is None:
                        if stopping() and not item.get('signalled'):
                            item['process'].send_signal(signal.SIGUSR1); item['signalled'] = True
                        continue
                    item['stream'].close(); del active[lane]
                    if code != 0 and verified_retirement(stage, item['id']):
                        write(stage/'launcher_exit_audit'/(job+'_'+item['id']+'.json'),
                            dict(job_id=job, task=item['id'], launcher_exit=code,
                                 disposition='Completed evaluation and verified durable archive take precedence.',
                                 time=time.time()))
                        continue
                    if code not in (0, 75):
                        raise RuntimeError(item['id']+f' failed with exit {code}')
                    if code == 75 and not stopping():
                        # Preserve files and end this allocation if the off-BeeGFS
                        # collector is unavailable. The monitor resumes once back.
                        raise RuntimeError('Archive collector unavailable or worker requested early stop: '+item['id'])
                if stopping():
                    if not active:
                        if read(stage/'control.json').get('pause'):
                            subprocess.run(['scancel', successor], check=True); successor = None
                            publish('paused')
                        else:
                            publish('continuing_at_time_limit')
                        return
                else:
                    ids = assignments(stage)
                    running = {w['id'] for w in active.values()}
                    blocked = blocked_tasks(stage)
                    pending = [n for n in ids if n not in running and n not in blocked and not (
                        (root/'work'/n/'cleanup.json').exists() and read(root/'work'/n/'cleanup.json')['status']=='complete')]
                    for lane in (0, 1):
                        if lane in active or not pending:
                            continue
                        if quota()['free_gib'] < 12:
                            raise RuntimeError('Personal BeeGFS quota headroom exhausted')
                        name = pending.pop(0)
                        log = stage/'worker_logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
                        stream = log.open('a')
                        command = step_prefix()+[PYTHON, str(stage/'cluster_runner.py'), 'worker', '--stage', str(stage), '--task', name]
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                        active[lane] = dict(id=name, stream=stream, process=process)
                    if not active:
                        if (stage/'colab_handoff.json').exists():
                            subprocess.run(['scancel', successor], check=True); successor = None
                            publish('waiting_for_recovery_audit' if blocked else 'complete'); return
                        publish('waiting_for_colab_handoff'); time.sleep(30); continue
                publish('running')
                time.sleep(10)
        except BaseException:
            write(stage/'failure.json', dict(job_id=job, traceback=traceback.format_exc(), time=time.time()))
            for item in active.values():
                if item['process'].poll() is None:
                    item['process'].send_signal(signal.SIGUSR1)
            for item in active.values():
                item['process'].wait(timeout=600); item['stream'].close()
            if successor:
                subprocess.run(['scancel', successor], check=False)
            publish('failed')
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run','worker','verify','submit'])
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--task')
    args = parser.parse_args(); stage = args.stage.resolve()
    activate(stage)
    if args.command == 'worker': worker(stage, args.task)
    elif args.command == 'verify': verify(stage)
    elif args.command == 'submit': verify(stage); print(submit(stage))
    else: run(stage)


if __name__ == '__main__':
    main()
