"""Queue a nested two-volume SAM3 comparison after the six foundation runs."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.cross_dataset_campaign import controller, followon_job as chain
from scripts.cross_dataset_campaign import foundation_checks_job as parent
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT

NAME = 'sam3_geofwi_2vol_bands_20261015'
MODULE = 'scripts.cross_dataset_campaign.sam3_two_volume_job'
ARMS = parent.ARMS
PARENT_SHA = '7badba27bdbe00ae18f63dc553be5a256d9546dde3b2ac53b5d0a00b4def2edf'
SELECTION = 'First two volumes in the frozen five-volume SAM3 support, selected before the comparison'
ADDITIONS = ('sam3_two_volume_job.py', 'manage_sam3_two_volume.py', 'test_sam3_two_volume.py')
BASE_COMMAND = controller.command_for
BASE_CANCEL = controller.cancel_successor


def prepare():
    stage = ROOT/'experiments'/NAME
    previous = ROOT/'experiments'/parent.NAME
    previous_plan = parent.verify(previous)
    if digest(previous/'manifest.json') != PARENT_SHA:
        raise ValueError('Working foundation source changed')
    if stage.exists():
        raise ValueError('Never overwrite a frozen campaign')
    stage.mkdir(parents=True)
    # Keep every working training file byte-identical; add orchestration only.
    shutil.copytree(previous/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ADDITIONS:
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/filename,
                     stage/'source/scripts/cross_dataset_campaign'/filename)
    for filename in ('original_recipe.json',):
        shutil.copy2(previous/filename, stage/filename)
    shutil.copy2(previous/'manifest.json', stage/'predecessor_manifest.json')
    original = read(previous/'protocols/GeoFWI3D/protocol.json')
    write(stage/'parent_protocol.json', original)
    protocol = deepcopy(original)
    protocol.update(name=NAME, train_cases=original['train_cases'][:2], support_selection=SELECTION)
    write(stage/'protocols/GeoFWI3D/protocol.json', protocol)
    plan = deepcopy(previous_plan)
    for key in ('budget_flops_by_dataset', 'recovery_of'):
        plan.pop(key, None)
    tasks = [dict(id=f'GeoFWI3D_SAM3_2vol_{arm}_f0s0', kind='foundation',
        dataset='GeoFWI3D', model='SAM3', fold=0, seed=0, arm=arm, target=.1,
        protocol='GeoFWI3D') for arm in ARMS]
    plan.update(name=NAME, tasks=tasks, num_runs=3,
        archive_resume_module=MODULE, archive_dependency_stages=[str(REMOTE/parent.NAME)],
        predecessor=dict(name=parent.NAME, stage=str(REMOTE/parent.NAME), manifest_sha256=PARENT_SHA),
        authorization='Queue baseline, bands and bands+edge with two SAM3 training volumes after the current runs; 200 updates, the same 96 validation volumes, fold0 seed0, no augmentation.',
        exploration_followup='Only these three nested-support SAM3 comparisons',
        comparison_policy='Same successful frozen SAM3 trainer and original recipe; only support reduced from five to two volumes. Final update checkpoint primary. Identical initialization and sampling across arms.',
        calibration_policy='Original sequential 10% reference target and 50% p95 ceiling, four calibration pairs per support volume; preserve data-dependent realized ratios, do not force 10%.')
    plan['storage']['archive'] = str(ROOT/'reports'/NAME)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    source = read(previous/'source_version.json')
    write(stage/'source_version.json', {**source, 'working_parent_manifest_sha256': PARENT_SHA,
        'training_source_policy': 'Every source file from the working predecessor is byte-identical; only new orchestration modules added'})
    batch = (previous/'run.sbatch').read_text().replace('hippo-foundation-original10', 'hippo-sam3-2vol').replace(parent.MODULE+' run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)): digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen two-volume comparison changed: '+name)
    if digest(stage/'predecessor_manifest.json') != PARENT_SHA:
        raise ValueError('Working source provenance changed')
    predecessor_manifest = read(stage/'predecessor_manifest.json')
    for name, sha in predecessor_manifest.items():
        if name.startswith('source/') and digest(stage/name) != sha:
            raise ValueError('Working training source changed: '+name)
    if digest(stage/'parent_protocol.json') != predecessor_manifest['protocols/GeoFWI3D/protocol.json']:
        raise ValueError('Parent protocol changed')
    original = read(stage/'parent_protocol.json')
    expected = {**original, 'name': NAME, 'train_cases': original['train_cases'][:2], 'support_selection': SELECTION}
    p = read(stage/'protocols/GeoFWI3D/protocol.json')
    if p != expected:
        raise ValueError('Recipe differs beyond the authorized support reduction')
    if (len(set(p['train_cases'])) != 2 or len(set(p['evaluation_cases'])) != 96
            or set(p['train_cases']) & set(p['evaluation_cases'])
            or (p['fold'], p['seed'], p['updates'], p['augmentation']) != (0, 0, 200, 'none')):
        raise ValueError('Wrong support, validation cohort, or training settings')
    plan = read(stage/'plan.json')
    expected_tasks = [dict(id=f'GeoFWI3D_SAM3_2vol_{arm}_f0s0', kind='foundation',
        dataset='GeoFWI3D', model='SAM3', fold=0, seed=0, arm=arm, target=.1, protocol='GeoFWI3D') for arm in ARMS]
    if (plan['name'] != NAME or plan['tasks'] != expected_tasks or plan['num_runs'] != 3
            or plan['augmentation'] != 'none' or plan['archive_resume_module'] != MODULE
            or plan['archive_wait_timeout_seconds'] != 300 or plan['max_job_hours'] != 24
            or plan['predecessor'] != dict(name=parent.NAME, stage=str(REMOTE/parent.NAME), manifest_sha256=PARENT_SHA)):
        raise ValueError('Comparison scope or continuation contract changed')
    return plan


def check_predecessor(plan):
    dep = plan['predecessor']; stage = Path(dep['stage'])
    if dep['name'] != parent.NAME or digest(stage/'manifest.json') != PARENT_SHA:
        raise ValueError('Foundation prerequisite changed')
    return stage, parent.verify(stage)


def predecessor_complete(stage, plan):
    status = read(stage/'status.json') if (stage/'status.json').exists() else {}
    if (read(stage/'control.json').get('pause') or status.get('status') != 'complete'
            or status.get('completed') != len(plan['tasks'])):
        return False
    for task in plan['tasks']:
        run = stage/'runs'/task['id']
        ack = read(run/'archive_acknowledgment.json')
        if (read(run/'completion.json')['status'] != 'complete'
                or read(run/'cleanup.json')['status'] != 'complete'
                or ack['status'] != 'verified' or not ack['source_verified']
                or ack['manifest_sha256'] != digest(run/'archive_manifest.json')
                or ack['source_manifest_sha256'] != digest(stage/'manifest.json')):
            raise ValueError('Foundation prerequisite archive or cleanup incomplete')
    return True


def configure_chain():
    chain.verify = verify
    chain.check_predecessor = check_predecessor
    chain.predecessor_complete = predecessor_complete
    chain.msd = parent


def submit(stage, continuation=None):
    configure_chain()
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    active = [row for row in controller.queue() if row[0] in own]
    if continuation is None and active:
        return read(stage/'submissions'/f'{active[0][0]}.json')
    if continuation is not None:
        return chain.call_submit(stage, continuation)
    if (stage/'status.json').exists() and read(stage/'status.json').get('status') == 'complete':
        raise RuntimeError('Two-volume comparison already complete')
    receipt = chain.submit(stage)
    handoff = read(stage/'handoff.json')
    handoff['policy'] = 'Finish the exact frozen six-run foundation campaign, including evaluation, verified local archives and cleanup; then run the three two-volume SAM3 arms. One queued continuation for the chain.'
    write(stage/'handoff.json', handoff)
    return receipt


def command_for(stage, plan, task, base_path):
    command = BASE_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.foundation_msd'
    return command


def run(stage):
    configure_chain()
    plan = verify(stage); predecessor, previous = check_predecessor(plan)
    job = os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_ALLOCATION_DEADLINE'] = str(time.time()+plan['graceful_stop_seconds'])
    if read(stage/'control.json').get('pause') or read(predecessor/'control.json').get('pause'):
        controller.summarize(stage, plan, 'paused'); return
    if (predecessor/'status.json').exists() and read(predecessor/'status.json').get('status') == 'failed':
        controller.summarize(stage, plan, 'blocked_predecessor_failed'); return
    controller.submit = lambda ignored, continuation=None: chain.ensure_successor(stage, job)
    def cancel(phase, active_job):
        if phase == predecessor and read(phase/'status.json').get('status') == 'complete':
            return
        BASE_CANCEL(stage, active_job)
    controller.cancel_successor = cancel
    if not predecessor_complete(predecessor, previous):
        controller.verify = parent.verify
        def previous_command(phase, phase_plan, task, base_path):
            command = command_for(phase, phase_plan, task, base_path)
            env = ':'.join(str(predecessor/p) for p in ('source', 'source/scripts', 'source/baselines/nnUNet'))
            return ['env', 'PYTHONPATH='+env, *command]
        controller.command_for = previous_command
        try:
            controller.run(predecessor)
        except Exception:
            controller.summarize(stage, plan, 'blocked_predecessor_failed'); BASE_CANCEL(stage, job); raise
        if not predecessor_complete(predecessor, previous):
            controller.summarize(stage, plan, read(predecessor/'status.json')['status'], active='predecessor:'+predecessor.name)
            return
    controller.verify = verify
    controller.command_for = command_for
    controller.run(stage)


def preflight(stage):
    plan = verify(stage); check_predecessor(plan)
    p = read(stage/'protocols/GeoFWI3D/protocol.json')
    manifest = p['manifest']
    if (p['train_cases'] != manifest['splits']['training_subsets']['5pct'][:2]
            or p['evaluation_cases'] != manifest['splits']['validation']):
        raise ValueError('GeoFWI3D fold0 cohort differs')
    if not Path(plan['foundation_python']).exists() or digest(plan['base_staging']) != p['base_sha256']:
        raise ValueError('Foundation runtime or pretrained checkpoint differs')
    for case in p['train_cases']+p['evaluation_cases']:
        for kind in ('image', 'label'):
            row = manifest['cases'][case]
            if digest(Path(p['data'])/row[kind]) != row[kind+'_sha256']:
                raise ValueError('GeoFWI3D image or annotation changed: '+case)
    hashes = read(stage/'source_version.json')['upstream_provenance']['python_source_sha256']
    if not hashes:
        raise ValueError('Missing upstream provenance')
    for name, sha in hashes.items():
        if digest(Path(p['upstream'])/name) != sha:
            raise ValueError('Upstream source changed: '+name)
    for module in ('test_foundation_fast', 'test_foundation_msd', 'test_sam3_two_volume'):
        args = [plan['foundation_python'], '-m', 'scripts.cross_dataset_campaign.'+module]
        if module == 'test_sam3_two_volume':
            args += ['--stage', str(stage)]
        subprocess.run(args, check=True)
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise ValueError('Insufficient personal quota')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        checks=['Nested two-volume support; unchanged 96-volume validation; fold0 seed0 no augmentation',
                'Working trainer byte identity and complete recipe comparison',
                'All 98 source images and annotations, base checkpoint and upstream source hashes',
                'Actual two-volume geometry, eight calibration pairs and 400 training pairs',
                'Optimizer resume and frozen MSD trainer tests; ordered gated handoff tests'], quota=usage,
        gpu_gate='Original finite native/bands/edge losses and retained backward probe before optimizer updates'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'preflight', 'submit', 'run', 'status'])
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()); return
    stage = args.stage.resolve()
    if args.command == 'preflight': preflight(stage)
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'run': run(stage)
    else: print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Queued, not started')


if __name__ == '__main__':
    main()
