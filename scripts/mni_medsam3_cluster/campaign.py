"""Frozen MNI three-seed plan and queue admission under the account job limit."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign import controller, adni_foundation_full_pool as pool
from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.cross_dataset_campaign.storage import quota, retire as retire_archived

MODULE = 'scripts.mni_medsam3_cluster.campaign'


def verify(stage):
    for name, expected in read(stage/'manifest.json').items():
        if digest(stage/name) != expected:
            raise ValueError('Frozen campaign changed: '+name)
    plan = read(stage/'plan.json')
    tasks = plan['tasks']
    seeds = plan.get('seeds', [0, 1, 2])
    updates = plan.get('updates_per_run', 200)
    workers = plan['parallelism']['workers']
    if [t['seed'] for t in tasks] != seeds or workers != min(2, len(seeds)):
        raise ValueError('Task seeds or GPU worker count differs from frozen plan')
    protocols = [read(stage/'protocols'/t['protocol']/'protocol.json') for t in tasks]
    for task, p in zip(tasks, protocols):
        if (p['seed'], p['fold'], p['updates'], p['train_subjects'], p['train_crops']) != (task['seed'], 0, updates, 5, 10):
            raise ValueError('Unexpected experiment scope')
        if p['augmentation'] != 'none' or task['arm'] != 'dice':
            raise ValueError('Unexpected loss/augmentation')
        if p['train_cases'] != protocols[0]['train_cases'] or p['evaluation_cases'] != protocols[0]['evaluation_cases']:
            raise ValueError('Seeds must use identical subject splits')
        if {x.rsplit('_', 1)[0] for x in p['train_cases']} & {x.rsplit('_', 1)[0] for x in p['evaluation_cases']}:
            raise ValueError('Subject leakage')
    return plan


def preflight(stage):
    plan = verify(stage)
    protocol = read(stage/'protocols'/plan['tasks'][0]['protocol']/'protocol.json')
    for path, expected in ((plan['base_staging'], plan['base_sha256']), (protocol['lora_weights'], protocol['lora_sha256'])):
        if digest(Path(path)) != expected:
            raise ValueError('Initialization hash differs: '+path)
    for name, expected in read(stage/'upstream_manifest.json').items():
        if digest(Path(protocol['upstream'])/name) != expected:
            raise ValueError('Upstream differs: '+name)
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    check = subprocess.run(['sbatch', '--test-only', str(stage/'run.sbatch'), str(stage)],
        text=True, capture_output=True, check=True)
    usage = quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient quota')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        quota=usage, scheduler=check.stdout+check.stderr, checked_at=time.time(),
        gpu_validation=f"At job start: {plan['parallelism']['workers']} isolated GPU worker(s). Per run: finite native/bands/edge backward before optimizer updates. Evaluation cache audited against reference predictions."))


def dependencies_ready(plan):
    records = []
    for entry in plan.get('wait_for_campaigns', []):
        root = Path(entry['stage'])
        state = read(root/'status.json') if (root/'status.json').exists() else {}
        same_plan = (root/'plan.json').exists() and digest(root/'plan.json') == entry['plan_sha256']
        done = state.get('training_completed', state.get('completed', 0))
        ready = (same_plan and state.get('status') in ('complete', 'complete_pending_archive')
                 and done == entry['expected_runs'] and not state.get('active'))
        records.append(dict(stage=str(root), ready=ready, same_plan=same_plan,
            status=state.get('status', 'not_started'), completed=done, total=entry['expected_runs']))
    return all(r['ready'] for r in records), records


def submit(stage):
    if not dependencies_ready(verify(stage))[0]:
        raise RuntimeError('The preceding sweep has not finished')
    controller.verify = verify
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    current = controller.queue()
    existing = [row for row in current if row[0] in own]
    if existing:
        return dict(status='already_queued', jobs=existing)
    receipt = controller.submit(stage)
    # Another campaign may submit between our queue check and sbatch. Undo
    # only this new pending job if it creates a third account job.
    current = controller.queue()
    if len(current) > 2:
        ours = next((r for r in current if r[0] == receipt['job_id']), None)
        if ours and ours[1] == 'PENDING':
            subprocess.run(['scancel', receipt['job_id']], check=True)
            write(stage/'submission_race.json', dict(cancelled=receipt, queue=current, time=time.time()))
            raise RuntimeError('Submission race: own pending job withdrawn to preserve account limit')
    return receipt


def admit(stage):
    """Lightweight, finite login-side waiter; no extra Slurm/GPU job slot."""
    with (stage/'admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify(stage)
        while True:
            if read(stage/'control.json').get('pause'):
                write(stage/'admission.json', dict(status='paused', time=time.time()))
                return
            state = read(stage/'status.json') if (stage/'status.json').exists() else {}
            if state.get('status') in ('complete', 'failed', 'paused'):
                return
            ready, dependencies = dependencies_ready(plan)
            if not ready:
                write(stage/'admission.json', dict(status='waiting_for_sweep_completion',
                    dependencies=dependencies, pid=os.getpid(), time=time.time()))
                time.sleep(30)
                continue
            current = controller.queue()
            own = {p.stem for p in (stage/'submissions').glob('*.json')}
            if any(r[0] in own for r in current):
                write(stage/'admission.json', dict(status='submitted', queue=current, time=time.time()))
                time.sleep(30)
                continue
            # Completed runs must be archived before another GPU allocation.
            if state.get('status') in ('waiting_for_archive_offline', 'waiting_for_base_offline'):
                time.sleep(30)
                continue
            write(stage/'admission.json', dict(status='waiting_for_account_slot', queue=current,
                pid=os.getpid(), time=time.time(), policy='At most one running and one queued job; existing jobs unchanged'))
            if len(current) <= 1 and all(r[1] in ('RUNNING', 'COMPLETING', 'CONFIGURING') for r in current):
                try:
                    receipt = submit(stage)
                    write(stage/'admission.json', dict(status='submitted', receipt=receipt, time=time.time()))
                except Exception:
                    write(stage/'admission_error.json', dict(traceback=traceback.format_exc(), time=time.time()))
            time.sleep(30)


def worker_command(stage, plan, task, base, lane):
    command = controller.command_for(stage, plan, task, base)
    command[2] = 'scripts.mni_medsam3_cluster.worker'
    return [*pool.step_prefix(), 'env', f'HIPPO_GPU_WORKER={lane}',
        f'MNI_MANIFEST_SHA={digest(stage/"manifest.json")}', *command,
        '--record', str(stage/'runs'/task['id']/'gpu_worker.json')]


def retire(output):
    # The stock gate checks full evaluations, durable metadata and checkpoint hashes.
    receipt = retire_archived(output)
    local = Path(read(output/'worker_timing.json')['local_scratch'])
    expected = Path('/tmp')/f"hippo-mni-{os.getuid()}-{os.environ['SLURM_JOB_ID']}"/output.name
    if local != expected:
        # Recovery can originate in a previous allocation; never touch that path.
        return receipt
    checkpoint = local/'checkpoint_latest.pt'
    if checkpoint.exists():
        expected_sha = read(output/'completion.json')['checkpoint_sha256']
        if digest(checkpoint) != expected_sha:
            raise ValueError('Scratch checkpoint changed; preserve it')
        receipt['scratch_deleted'] = dict(path=str(checkpoint), sha256=expected_sha, bytes=checkpoint.stat().st_size)
        write(output/'cleanup.json', receipt)
        checkpoint.unlink()
    return receipt


def run(stage):
    plan = verify(stage)
    pool.worker_command = worker_command
    pool.retire = retire
    if plan['parallelism']['workers'] == 1:
        pool.verify_gpu_probe = verify_single_gpu
    # MNI's unchanged matched.calibrate performs the native-loss gradient check
    # for each seed. The separate ADNI all-loss preflight is not applicable.
    pool.gpu_preflight = lambda stage, plan, base, publish: True
    # This small campaign fits one six-hour allocation; don't reserve an extra
    # continuation ahead of the user's other campaigns.
    pool.run(stage, plan, lambda: None, lambda: None)


def verify_single_gpu(stage, plan):
    job = os.environ['SLURM_JOB_ID']
    folder = stage/'gpu_probe'/job
    subprocess.run([*pool.step_prefix(1), plan['foundation_python'], '-m',
        'scripts.cross_dataset_campaign.foundation_gpu_worker', '--probe-directory', str(folder)],
        cwd=stage/'source', check=True)
    record = read(folder/'0.json')
    if record['status'] != 'passed' or record['visible_devices'] != 1 or record['slurm_job_id'] != job:
        raise RuntimeError('Single-GPU binding check failed')
    write(stage/'gpu_allocation.json', dict(status='passed', workers=[record],
        manifest_sha256=digest(stage/'manifest.json'), job_id=job))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'preflight', 'submit', 'admit', 'run'])
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.command](args.stage.resolve())
    if result is not None:
        print(result, flush=True)


if __name__ == '__main__':
    main()
