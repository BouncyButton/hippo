"""Six two-sample foundation comparisons on the account's maximum two GPUs."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.cross_dataset_campaign import controller, followon_job as chain
from scripts.cross_dataset_campaign import foundation_checks_job as parent
from scripts.cross_dataset_campaign import sam3_two_volume_job as previous
from scripts.cross_dataset_campaign import foundation_parallel as parallel
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT

NAME = 'foundation_two_sample_2gpu_20261015'
MODULE = 'scripts.cross_dataset_campaign.foundation_two_sample_job'
PARENT_SHA = previous.PARENT_SHA
PREVIOUS_SHA = '51116e266eea8fe802cfbafd211ea97248807994a3f7d19430789e6e811fdc30'
ARMS = parent.ARMS
DATASETS = ('GeoFWI3D', 'ADNI')
ADDITIONS = (*previous.ADDITIONS, 'foundation_two_sample_job.py', 'foundation_parallel.py',
             'foundation_gpu_worker.py', 'test_foundation_parallel.py', 'manage_foundation_two_sample.py')
BASE_COMMAND = controller.command_for
BASE_CANCEL = controller.cancel_successor
BASE_SUBMIT = controller.submit


def tasks():
    return [dict(id=f'{dataset}_{"SAM3" if dataset == "GeoFWI3D" else "MedSAM3"}_2sample_{arm}_f0s0',
        kind='foundation', dataset=dataset, model='SAM3' if dataset == 'GeoFWI3D' else 'MedSAM3',
        fold=0, seed=0, arm=arm, target=.1, protocol=dataset) for arm in ARMS for dataset in DATASETS]


def reduced_protocol(original):
    result = deepcopy(original)
    result.update(name=NAME, train_cases=original['train_cases'][:2],
        support_selection='First two samples from the frozen five-sample support; selected before the comparison')
    if result['dataset'] == 'ADNI':
        result['support_subjects'] = original['support_subjects'][:2]
    return result


def prepare():
    stage = ROOT/'experiments'/NAME
    source_stage = ROOT/'experiments'/parent.NAME
    plan = deepcopy(parent.verify(source_stage))
    if digest(source_stage/'manifest.json') != PARENT_SHA:
        raise ValueError('Working source provenance changed')
    if stage.exists(): raise ValueError('Never overwrite a frozen campaign')
    stage.mkdir(parents=True)
    shutil.copytree(source_stage/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ADDITIONS:
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/filename,
                     stage/'source/scripts/cross_dataset_campaign'/filename)
    shutil.copy2(source_stage/'manifest.json', stage/'predecessor_manifest.json')
    shutil.copy2(source_stage/'original_recipe.json', stage/'original_recipe.json')
    for dataset in DATASETS:
        original = read(source_stage/'protocols'/dataset/'protocol.json')
        write(stage/'parent_protocols'/dataset/'protocol.json', original)
        write(stage/'protocols'/dataset/'protocol.json', reduced_protocol(original))
    for key in ('budget_flops_by_dataset', 'recovery_of'):
        plan.pop(key, None)
    plan.update(name=NAME, tasks=tasks(), num_runs=6, archive_resume_module=MODULE,
        archive_dependency_stages=[str(REMOTE/parent.NAME)],
        predecessor=dict(name=parent.NAME, stage=str(REMOTE/parent.NAME), manifest_sha256=PARENT_SHA),
        supersedes=dict(name=previous.NAME, stage=str(REMOTE/previous.NAME), manifest_sha256=PREVIOUS_SHA, job_id='702445'),
        authorization='User explicitly requests two-sample SAM3 GeoFWI3D and MedSAM3 ADNI comparisons, both baseline/bands/bands+edge, fold0 seed0 no augmentation, 200 updates, and the maximum number of concurrent GPU workers permitted by the account.',
        exploration_followup='Only these six two-sample foundation experiments',
        comparison_policy='Working frozen training engine, native losses, optimizer, precision, sampling and resume unchanged. Independent experiments run on isolated GPUs, never distributed optimization. Final update checkpoint primary.',
        calibration_policy='Original data-dependent sequential 10% reference target and 50% p95 ceiling; four calibration pairs per training sample; do not force achieved ratios.',
        parallelism=dict(workers=2, datasets=list(DATASETS), gpus_per_worker=1, cpus_per_worker=4,
            gpu_type='4g.40gb', allocation_cpu_count=8, allocation_memory_gib=80,
            account_max_gpus=2, qos_max_gpus_per_job=4,
            policy='One independent experiment per dataset concurrently; baseline before bands before bands+edge within each dataset'))
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), minimum_free_gib=10,
        estimated_peak_logical_gib=4, mirrored_peak_gib=8, reserve_gib=2)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    write(stage/'source_version.json', {**read(source_stage/'source_version.json'),
        'working_parent_manifest_sha256': PARENT_SHA,
        'training_source_policy': 'Every working source file is byte-identical; add only orchestration and a GPU binding audit wrapper'})
    batch = (source_stage/'run.sbatch').read_text().replace('hippo-foundation-original10', 'hippo-foundation-2gpu')
    batch = batch.replace('#SBATCH --gres=gpu:4g.40gb:1', '#SBATCH --nodes=1\n#SBATCH --ntasks=2\n#SBATCH --gres=gpu:4g.40gb:2')
    batch = batch.replace('#SBATCH --mem=40G', '#SBATCH --mem=80G').replace(parent.MODULE+' run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)): digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha: raise ValueError('Frozen campaign changed: '+name)
    if digest(stage/'predecessor_manifest.json') != PARENT_SHA: raise ValueError('Parent source changed')
    manifest = read(stage/'predecessor_manifest.json')
    for name, sha in manifest.items():
        if name.startswith('source/') and digest(stage/name) != sha:
            raise ValueError('Working trainer source changed: '+name)
    for dataset in DATASETS:
        ref = stage/'parent_protocols'/dataset/'protocol.json'
        if digest(ref) != manifest[f'protocols/{dataset}/protocol.json']: raise ValueError('Parent recipe changed')
        p = read(stage/'protocols'/dataset/'protocol.json')
        if p != reduced_protocol(read(ref)): raise ValueError('Unrequested fine-tuning recipe change')
        if (len(set(p['train_cases'])) != 2 or len(set(p['evaluation_cases'])) != (96 if dataset == 'GeoFWI3D' else 54)
                or set(p['train_cases']) & set(p['evaluation_cases'])
                or (p['fold'], p['seed'], p['updates'], p['augmentation']) != (0,0,200,'none')):
            raise ValueError('Wrong support or scientific settings')
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 6
            or plan['archive_resume_module'] != MODULE or plan['archive_wait_timeout_seconds'] != 300
            or plan['max_job_hours'] != 24 or plan['augmentation'] != 'none'
            or plan['parallelism']['workers'] != 2 or plan['parallelism']['datasets'] != list(DATASETS)
            or plan['parallelism']['cpus_per_worker'] != 4 or plan['parallelism']['gpus_per_worker'] != 1
            or plan['predecessor'] != dict(name=parent.NAME, stage=str(REMOTE/parent.NAME), manifest_sha256=PARENT_SHA)
            or plan['supersedes'] != dict(name=previous.NAME, stage=str(REMOTE/previous.NAME), manifest_sha256=PREVIOUS_SHA, job_id='702445')):
        raise ValueError('Requested scope or resource contract changed')
    return plan


check_predecessor = previous.check_predecessor
predecessor_complete = previous.predecessor_complete


def configure_chain():
    chain.verify = verify
    chain.check_predecessor = check_predecessor
    chain.predecessor_complete = predecessor_complete
    chain.msd = parent


def replace_queued(stage):
    plan = verify(stage); predecessor, _ = check_predecessor(plan)
    old = Path(plan['supersedes']['stage'])
    if digest(old/'manifest.json') != PREVIOUS_SHA: raise ValueError('Queued campaign provenance changed')
    previous.verify(old)
    preflight = read(stage/'preflight.json')
    if preflight.get('status') != 'passed' or preflight.get('manifest_sha256') != digest(stage/'manifest.json'):
        raise ValueError('Matching passed preflight is required')
    if controller.quota()['free_gib'] < plan['storage']['minimum_free_gib']:
        raise ValueError('Insufficient personal quota')
    with (stage/'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = controller.queue()
        pending = [r for r in rows if r[1] == 'PENDING']
        old_receipt = read(old/'submission.json')
        if (len(pending) != 1 or pending[0][0] != old_receipt['job_id']
                or old_receipt['job_id'] != '702445' or old_receipt['manifest_sha256'] != PREVIOUS_SHA):
            raise RuntimeError('Expected original queued job is no longer pending; do not cancel active or unrelated jobs')
        own_parent = {f.stem for f in (predecessor/'submissions').glob('*.json')}
        running = [r for r in rows if r[1] in ('RUNNING','COMPLETING','CONFIGURING')]
        if running and (len(running) != 1 or running[0][0] not in own_parent):
            raise RuntimeError('Unrelated running job occupies the account')
        if any((old/'runs').glob('*/protocol.json')): raise RuntimeError('Do not revise an experiment that has started')
        controls = {p: read(p/'control.json') for p in (predecessor, old, stage)}
        if any(c.get('pause') for c in controls.values()): raise RuntimeError('Manual pause must be preserved')
        subprocess.run(['scancel', old_receipt['job_id']], check=True)
        for _ in range(30):
            if not any(r[0] == old_receipt['job_id'] for r in controller.queue()): break
            time.sleep(1)
        else: raise RuntimeError('Queued job cancellation has not completed')
        for p in (predecessor, old):
            write(p/'control.json', {**controls[p], 'automatic_continuation_owner': str(stage)})
        try:
            receipt = chain.call_submit(stage)
        except Exception:
            for p in (predecessor, old): write(p/'control.json', controls[p])
            # A definitive failure restores the previously authorized campaign.
            if not any(r[1] == 'PENDING' for r in controller.queue()):
                saved = controller.verify; controller.verify = previous.verify
                try: BASE_SUBMIT(old)
                finally: controller.verify = saved
            raise
        write(stage/'handoff.json', dict(status='queued', replaced_job=old_receipt['job_id'],
            predecessor_stage=str(predecessor), superseded_stage=str(old), submission=receipt,
            policy='Finish the original foundation campaign including evaluation, verified local archives and cleanup, then run six two-sample experiments on two isolated GPUs'))
        write(old/'status.json', dict(status='superseded', replacement_stage=str(stage), replacement_job=receipt['job_id']))
        return receipt


def submit(stage, continuation=None):
    configure_chain()
    own = {f.stem for f in (stage/'submissions').glob('*.json')}
    active = [r for r in controller.queue() if r[0] in own]
    if continuation is None and active: return read(stage/'submissions'/f'{active[0][0]}.json')
    if continuation is not None: return chain.call_submit(stage, continuation)
    return replace_queued(stage)


def run(stage):
    configure_chain()
    plan = verify(stage); predecessor, old_plan = check_predecessor(plan)
    job = os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_ALLOCATION_DEADLINE'] = str(time.time()+plan['graceful_stop_seconds'])
    if read(stage/'control.json').get('pause') or read(predecessor/'control.json').get('pause'):
        controller.summarize(stage, plan, 'paused'); return
    if (predecessor/'status.json').exists() and read(predecessor/'status.json').get('status') == 'failed':
        controller.summarize(stage, plan, 'blocked_predecessor_failed'); return
    controller.submit = lambda ignored, continuation=None: chain.ensure_successor(stage, job)
    def cancel(phase, active):
        if phase == predecessor and read(phase/'status.json').get('status') == 'complete': return
        BASE_CANCEL(stage, active)
    controller.cancel_successor = cancel
    if not predecessor_complete(predecessor, old_plan):
        controller.verify = parent.verify
        def previous_command(phase, phase_plan, task, base):
            command = BASE_COMMAND(phase, phase_plan, task, base)
            command[2] = 'scripts.cross_dataset_campaign.foundation_msd'
            env = ':'.join(str(predecessor/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
            return [*parallel.step_prefix(), 'env', 'PYTHONPATH='+env, *command]
        controller.command_for = previous_command
        try: controller.run(predecessor)
        except Exception:
            controller.summarize(stage, plan, 'blocked_predecessor_failed'); BASE_CANCEL(stage, job); raise
        if not predecessor_complete(predecessor, old_plan):
            controller.summarize(stage, plan, read(predecessor/'status.json')['status'], active='predecessor:'+predecessor.name)
            return
    controller.verify = verify; controller.command_for = BASE_COMMAND
    parallel.run(stage, plan, lambda: chain.ensure_successor(stage, job), lambda: BASE_CANCEL(stage, job))


def preflight(stage):
    plan = verify(stage); check_predecessor(plan)
    if digest(plan['base_staging']) != plan['base_sha256']: raise ValueError('Base checkpoint differs')
    for dataset in DATASETS:
        p = read(stage/'protocols'/dataset/'protocol.json')
        if dataset == 'ADNI':
            if digest(p['data']) != p['dataset_sha256'] or digest(p['lora_weights']) != p['lora_sha256']:
                raise ValueError('ADNI data or medical adapter differs')
        else:
            m = p['manifest']
            if p['train_cases'] != m['splits']['training_subsets']['5pct'][:2] or p['evaluation_cases'] != m['splits']['validation']:
                raise ValueError('GeoFWI3D cohort changed')
            for case in p['train_cases']+p['evaluation_cases']:
                for kind in ('image','label'):
                    if digest(Path(p['data'])/m['cases'][case][kind]) != m['cases'][case][kind+'_sha256']:
                        raise ValueError('GeoFWI3D source changed')
        hashes = read(stage/'source_version.json')['upstream_provenance']['python_source_sha256']
        if not hashes: raise ValueError('Missing upstream provenance')
        for name, sha in hashes.items():
            if digest(Path(p['upstream'])/name) != sha: raise ValueError('Upstream source changed')
    subprocess.run([plan['foundation_python'], '-m', 'scripts.cross_dataset_campaign.test_foundation_parallel', '--stage', str(stage)], check=True)
    subprocess.run(['bash','-n',str(stage/'run.sbatch')], check=True)
    qos = subprocess.check_output(['sacctmgr','-n','-P','show','qos','stud',
        'format=Name,MaxTRESPerUser%80,MaxTRESPJ%80,MaxJobsPU,MaxSubmitPU'], text=True)
    fields = next(x.strip().split('|') for x in qos.splitlines() if x.strip().startswith('stud|'))
    limits = dict(x.split('=') for x in fields[1].split(','))
    if int(limits['gres/gpu']) != 2 or int(limits['cpu']) < 8:
        raise ValueError('Account limits changed; reassess maximum parallelism')
    smoke = subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)], text=True, capture_output=True, check=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']: raise ValueError('Insufficient quota')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        checks=['Working training source byte-identical', 'Two nested training samples and unchanged validation for both datasets',
                'Data, pretrained model, medical adapter and upstream hashes', 'Independent GPU worker, resume, archive and handoff tests',
                'Actual support geometry and matched 400-pair schedules', 'Two GPUs are the account maximum; Slurm test submission passed'],
        quota=usage, qos=qos, scheduler_test=smoke.stdout+smoke.stderr,
        gpu_gate='At allocation start, two Slurm tasks must each see one distinct GPU; original native-loss/backward probe before training'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare','preflight','submit','run','status'])
    parser.add_argument('--stage', type=Path); args = parser.parse_args()
    if args.command == 'prepare': print(prepare()); return
    stage = args.stage.resolve()
    if args.command == 'preflight': preflight(stage)
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'run': run(stage)
    else: print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Queued, not started')


if __name__ == '__main__': main()
