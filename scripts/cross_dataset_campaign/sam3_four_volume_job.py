"""Four-volume SAM3 constraint-strength comparison with unchanged training code."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'sam3_geofwi_4vol_strength_20261015'
MODULE = 'scripts.cross_dataset_campaign.sam3_four_volume_job'
PARENT = 'foundation_two_sample_2gpu_20261015'
PARENT_SHA = '9c40972644c2bee57f2dd597e6b8c8201e6b3f9fa484d4c7ec5ef6b683e46b54'
FIVE = 'foundation_adni_geofwi_bands_original10_20261015_msd_native'
FIVE_SHA = '7badba27bdbe00ae18f63dc553be5a256d9546dde3b2ac53b5d0a00b4def2edf'
TARGETS = (.1, .2, .4)
ARMS = ('dice_bands', 'dice_bands_edge')
SELECTION = 'First four volumes of the original frozen five-volume support; no validation-based selection'
BASE_SUBMIT = controller.submit


def tasks():
    specs = [('dice', .1)] + [(arm, target) for target in TARGETS for arm in ARMS]
    return [dict(id=f'GeoFWI3D_SAM3_4vol_{arm}_g{round(target*100)}_f0s0',
                 kind='foundation', dataset='GeoFWI3D', model='SAM3', fold=0, seed=0,
                 arm=arm, target=target, p95_cap=5*target, protocol=f'g{round(target*100)}', lane=i % 2)
            for i, (arm, target) in enumerate(specs)]


def recipe(parent, original_five, target):
    result = deepcopy(parent)
    result.update(name=NAME, train_cases=original_five['train_cases'][:4],
                  support_selection=SELECTION, gradient_ratio_target=target,
                  gradient_ratio_p95_cap=5*target)
    return result


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    parent = ROOT/'reports'/PARENT/'frozen'
    five = ROOT/'reports'/FIVE/'frozen'
    if digest(parent/'manifest.json') != PARENT_SHA or digest(five/'manifest.json') != FIVE_SHA:
        raise ValueError('Working predecessor changed')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Parent source changed: '+name)
    stage.mkdir(parents=True)
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('sam3_four_volume_job.py', 'sam3_four_volume_parallel.py'):
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/name, stage/'source/scripts/cross_dataset_campaign'/name)
    for name in ('source_version.json', 'original_recipe.json'):
        shutil.copy2(parent/name, stage/name)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copy2(five/'manifest.json', stage/'five_volume_manifest.json')
    shutil.copy2(parent/'protocols/GeoFWI3D/protocol.json', stage/'parent_protocol.json')
    shutil.copy2(five/'protocols/GeoFWI3D/protocol.json', stage/'original_five_volume_protocol.json')
    for target in TARGETS:
        write(stage/'protocols'/f'g{round(target*100)}'/'protocol.json',
              recipe(read(stage/'parent_protocol.json'), read(stage/'original_five_volume_protocol.json'), target))
    plan = deepcopy(read(parent/'plan.json'))
    for key in ('predecessor', 'supersedes', 'archive_resume_module', 'archive_dependency_stages'):
        plan.pop(key, None)
    plan.update(name=NAME, tasks=tasks(), num_runs=7, single_allocation=True,
        authorization='User approved four GeoFWI3D training volumes, baseline plus bands and bands+edge at10/20/40% targets, proportionally increased gradient ceilings, unchanged200 updates/fold0/seed0/no augmentation, no new ADNI runs.',
        exploration_followup='Only the seven frozen four-volume SAM3 comparisons',
        comparison_policy='Matched four-volume baseline and10% controls separate support-size and constraint-strength effects; identical initialization and sampling across arms; final update checkpoint primary.',
        calibration_policy='Sequential LoRA gradient ratios against the native SAM3 objective, not pure Dice. Target/p95 pairs10%/50%,20%/100%,40%/200%; record actual achieved ratios and gradient alignment.',
        parallelism=dict(workers=2, lanes=[0,1], gpus_per_worker=1, cpus_per_worker=4,
                         gpu_type='4g.40gb', allocation_cpu_count=8, allocation_memory_gib=80),
        analysis=dict(primary='Final mean validation Dice versus matched four-volume baseline',
                      diagnostics=['training Dice','foreground FN/FP','per-volume paired Dice changes',
                                   'realized calibration median/p95 ratios','native/auxiliary gradient cosine',
                                   'gradient clipping fraction','frozen-base and adapter hashes']))
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), minimum_free_gib=10,
                          estimated_peak_logical_gib=4, mirrored_peak_gib=8, reserve_gib=2)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    batch = (parent/'run.sbatch').read_text()
    batch = batch.replace('hippo-foundation-2gpu', 'hippo-sam3-4vol-strength')
    batch = batch.replace('scripts.cross_dataset_campaign.foundation_two_sample_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen four-volume experiment changed: '+name)
    if digest(stage/'parent_manifest.json') != PARENT_SHA or digest(stage/'five_volume_manifest.json') != FIVE_SHA:
        raise ValueError('Parent provenance changed')
    parent = read(stage/'parent_manifest.json')
    for name, sha in parent.items():
        if name.startswith('source/') and digest(stage/name) != sha:
            raise ValueError('Working scientific training source changed: '+name)
    if digest(stage/'parent_protocol.json') != parent['protocols/GeoFWI3D/protocol.json']:
        raise ValueError('Two-volume reference changed')
    five = read(stage/'five_volume_manifest.json')
    if digest(stage/'original_five_volume_protocol.json') != five['protocols/GeoFWI3D/protocol.json']:
        raise ValueError('Original support selection changed')
    for target in TARGETS:
        p = read(stage/'protocols'/f'g{round(target*100)}'/'protocol.json')
        if p != recipe(read(stage/'parent_protocol.json'), read(stage/'original_five_volume_protocol.json'), target):
            raise ValueError('Unrequested scientific change')
        if (len(set(p['train_cases'])) != 4 or len(set(p['evaluation_cases'])) != 96
                or set(p['train_cases']) & set(p['evaluation_cases'])
                or (p['fold'],p['seed'],p['updates'],p['augmentation']) != (0,0,200,'none')
                or p['calibration_slabs_per_volume'] != 4):
            raise ValueError('Wrong cohort or training scope')
    plan = read(stage/'plan.json')
    if (plan['tasks'] != tasks() or plan['num_runs'] != 7 or plan['name'] != NAME
            or plan['parallelism']['workers'] != 2 or plan['parallelism']['lanes'] != [0,1]
            or not plan.get('single_allocation') or 'archive_resume_module' in plan
            or plan['storage']['minimum_free_gib'] < 10 or plan['archive_wait_timeout_seconds'] != 300):
        raise ValueError('Incorrect execution contract')
    return plan


def preflight(stage):
    plan = verify(stage)
    p = read(stage/'protocols/g10/protocol.json')
    m = p['manifest']
    if p['train_cases'] != m['splits']['training_subsets']['5pct'][:4] or p['evaluation_cases'] != m['splits']['validation']:
        raise ValueError('Frozen training/validation split changed')
    if digest(plan['base_staging']) != p['base_sha256']:
        raise ValueError('Base checkpoint changed')
    for case in p['train_cases']+p['evaluation_cases']:
        for kind in ('image','label'):
            if digest(Path(p['data'])/m['cases'][case][kind]) != m['cases'][case][kind+'_sha256']:
                raise ValueError('Image/annotation changed: '+case)
    hashes = read(stage/'source_version.json')['upstream_provenance']['python_source_sha256']
    if not hashes:
        raise ValueError('Missing upstream source provenance')
    for name, sha in hashes.items():
        if digest(Path(p['upstream'])/name) != sha:
            raise ValueError('SAM3 upstream source changed')
    import torch
    from scripts.cross_dataset_campaign.foundation import Volumes, SupportSlabs
    torch.set_num_threads(2)
    reference_schedule = None
    for task in tasks():
        protocol = {**read(stage/'protocols'/task['protocol']/'protocol.json'), 'arm':task['arm']}
        slabs = SupportSlabs(Volumes(protocol), protocol, None)
        schedule = slabs.schedule(protocol)
        if len(slabs.inventory) != 4 or len(schedule) != 400 or len(slabs.calibration_schedule(4)) != 16:
            raise ValueError('Actual support/sampling inventory differs')
        if set(case for case, _ in schedule) != set(p['train_cases']):
            raise ValueError('Missing training volume')
        if reference_schedule is not None and schedule != reference_schedule:
            raise ValueError('Unmatched sampling across conditions')
        reference_schedule = schedule
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    smoke = subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)],text=True,capture_output=True,check=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise ValueError('Insufficient personal storage')
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),quota=usage,
        support=p['train_cases'],training_pairs=400,calibration_pairs=16,validation_volumes=96,
        checks=['All scientific training source byte-identical to successful SAM3 experiment',
                'Actual four-volume support and matched400-pair schedules across seven conditions',
                'All100 images/annotations, pretrained checkpoint and upstream source hashes',
                'Only support size and target/p95 strength change; no ADNI work',
                'Two isolated GPU workers; per-run verified durable archive before cleanup'],
        scheduler_test=smoke.stdout+smoke.stderr,
        gpu_gate='Original finite native/bands/edge losses and retained backward check before training'))


def submit(stage, continuation=None):
    if continuation is not None:
        raise ValueError('One allocation; interrupted recovery is preserved')
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    for row in controller.queue():
        if row[0] in own:
            return read(stage/'submissions'/f'{row[0]}.json')
    return BASE_SUBMIT(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','verify','preflight','submit','run','status'])
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare()); return
    stage = args.stage.resolve()
    controller.verify = verify
    controller.submit = submit
    if args.command == 'verify': print(verify(stage)['name'])
    elif args.command == 'preflight': preflight(stage); print('Four-volume SAM3 preflight passed')
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'status': print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        from scripts.cross_dataset_campaign import sam3_four_volume_parallel as parallel
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__': main()
