"""Resume only the 22 unrun five-volume SAM3 variants in one allocation."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.cross_dataset_campaign.sam3_five_volume_updates_job import tasks as original_tasks

ROOT = Path(__file__).resolve().parents[2]
NAME = 'sam3_geofwi_5vol_completion_20261018'
MODULE = 'scripts.cross_dataset_campaign.sam3_five_volume_completion_job'
PARENT = 'sam3_geofwi_5vol_update_sweep_20261016'
PARENT_SHA = 'b65bf8fa536d8536e667ab3f48278a3234b70504c13301b813e6540ba4eb6e5a'
BASE_SUBMIT = controller.submit


def tasks():
    return [task for task in original_tasks() if not (task['updates'] == 10 and task['arm'] in ('dice', 'dice_bands'))]


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    parent = ROOT/'experiments'/PARENT
    if digest(parent/'manifest.json') != PARENT_SHA:
        raise ValueError('Parent experiment changed')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Parent file changed: '+name)
    results = ROOT/'reports'/PARENT
    comparison = read(results/'comparison.json')
    if (comparison['completed'] != 2 or comparison['total'] != 24
            or {r['id'] for r in comparison['runs']} != {t['id'] for t in original_tasks() if t not in tasks()}
            or any(r['cleanup_status'] != 'complete' for r in comparison['runs'])):
        raise ValueError('Original two runs are not fully archived and cleaned')
    stage.mkdir(parents=True)
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT/'scripts/cross_dataset_campaign/sam3_five_volume_completion_job.py',
                 stage/'source/scripts/cross_dataset_campaign/sam3_five_volume_completion_job.py')
    shutil.copytree(parent/'protocols', stage/'protocols')
    for name in ('source_version.json', 'original_five_volume_protocol.json'):
        shutil.copy2(parent/name, stage/name)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=len(tasks()),
                authorization='User explicitly authorized queueing the 22 variants left unrun by job 703589.',
                exploration_followup='Only the 22 missing five-volume SAM3 variants',
                archive_wait_timeout_seconds=3600)
    plan['storage']['archive'] = str(ROOT/'reports'/NAME)
    plan['storage']['estimated_peak_logical_gib'] = 4
    plan['storage']['mirrored_peak_gib'] = 8
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    batch = (parent/'run.sbatch').read_text()
    batch = batch.replace('hippo-sam3-5vol-updates', 'hippo-sam3-5vol-complete')
    batch = batch.replace('scripts.cross_dataset_campaign.sam3_five_volume_updates_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*'))
                                  if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    manifest = read(stage/'manifest.json')
    for name, sha in manifest.items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen completion source changed: '+name)
    if digest(stage/'parent_manifest.json') != PARENT_SHA:
        raise ValueError('Parent provenance changed')
    parent_manifest = read(stage/'parent_manifest.json')
    for name, sha in parent_manifest.items():
        if (name.startswith(('source/', 'protocols/')) or name in ('source_version.json', 'original_five_volume_protocol.json')) and digest(stage/name) != sha:
            raise ValueError('Scientific source or cohort changed: '+name)
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 22
            or plan['max_job_hours'] != 24 or plan['parallelism']['workers'] != 2
            or plan['archive_wait_timeout_seconds'] != 3600
            or not plan['single_allocation'] or plan['storage']['minimum_free_gib'] < 10):
        raise ValueError('Incorrect completion contract')
    for task in tasks():
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        if (len(p['train_cases']) != 5 or len(p['evaluation_cases']) != 96
                or p['updates'] != task['updates'] or p['learning_rate_horizon_updates'] != 200):
            raise ValueError('Cohort or stopping point changed')
    return plan


def preflight(stage):
    plan = verify(stage)
    p = read(stage/'original_five_volume_protocol.json')
    if digest(plan['base_staging']) != p['base_sha256']:
        raise ValueError('Base SAM3 checkpoint changed')
    for case in p['train_cases']+p['evaluation_cases']:
        for kind in ('image', 'label'):
            item = p['manifest']['cases'][case]
            if digest(Path(p['data'])/item[kind]) != item[kind+'_sha256']:
                raise ValueError('Image/annotation changed: '+case)
    for name, sha in read(stage/'source_version.json')['upstream_provenance']['python_source_sha256'].items():
        if digest(Path(p['upstream'])/name) != sha:
            raise ValueError('SAM3 upstream source changed')
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    smoke = subprocess.run(['sbatch', '--test-only', str(stage/'run.sbatch'), str(stage)],
                           text=True, capture_output=True, check=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']+plan['storage']['mirrored_peak_gib']+plan['storage']['reserve_gib']:
        raise ValueError('Insufficient BeeGFS quota headroom')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
         quota=usage, runs=22, support=p['train_cases'], validation_volumes=96,
         scheduler_test=smoke.stdout+smoke.stderr,
         checks=['The two prior runs are archive-verified and retired',
                 'Exactly 22 missing variants; original scientific source and cohorts unchanged',
                 'All source data, upstream files, and base checkpoint hash verified',
                 'Two GPU workers, 24-hour allocation, one queued job behind current allocation',
                 'One-hour archive acknowledgment window with persistent local collector']))


def submit(stage):
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    for row in controller.queue():
        if row[0] in own:
            return read(stage/'submissions'/f'{row[0]}.json')
    controller.verify = verify
    return BASE_SUBMIT(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'preflight', 'submit', 'run', 'status'])
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare())
        return
    stage = args.stage.resolve()
    if args.command == 'verify':
        print(verify(stage)['name'])
    elif args.command == 'preflight':
        preflight(stage)
        print('SAM3 completion preflight passed')
    elif args.command == 'submit':
        print(submit(stage))
    elif args.command == 'status':
        print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        from scripts.cross_dataset_campaign import sam3_four_volume_parallel as parallel
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__':
    main()
