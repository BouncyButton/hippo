"""Five-volume SAM3 update and gentle-edge sweep in one bounded allocation."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'sam3_geofwi_5vol_update_sweep_20261016'
MODULE = 'scripts.cross_dataset_campaign.sam3_five_volume_updates_job'
PARENT = 'sam3_geofwi_4vol_strength_20261015'
PARENT_SHA = 'ff0f3ea6731a5c4d9b1031bef57fd5beea331bbf878b626d508c46d49be7a8a3'
FIVE = 'foundation_adni_geofwi_bands_original10_20261015_msd_native'
FIVE_SHA = '7badba27bdbe00ae18f63dc553be5a256d9546dde3b2ac53b5d0a00b4def2edf'
UPDATES = (10, 25, 50, 100, 200)
BASE_SUBMIT = controller.submit


def specifications():
    rows = []
    for updates in UPDATES:
        rows.extend([(updates, 'dice', .2, None), (updates, 'dice_bands', .2, None)])
        if updates in (50, 100):
            rows.extend((updates, 'dice_bands', bands, None) for bands in (.1, .4))
            rows.extend((updates, 'dice_bands_edge', .2, edge) for edge in (.005, .01, .02, .05))
        if updates == 200:
            rows.extend((updates, 'dice_bands_edge', .2, edge) for edge in (.01, .02))
    return rows


def protocol_name(updates, bands, edge):
    return f'u{updates}_b{round(bands*100)}_e{round(edge*1000) if edge is not None else 0}'


def tasks():
    result = []
    load = [0., 0.]
    for updates, arm, bands, edge in specifications():
        lane = min(range(2), key=lambda i: load[i])
        load[lane] += 35 + .12*updates + (12 if edge is not None else 0)
        label = protocol_name(updates, bands, edge)
        result.append(dict(id=f'GeoFWI3D_SAM3_5vol_{arm}_{label}_f0s0', kind='foundation',
                           dataset='GeoFWI3D', model='SAM3', fold=0, seed=0, arm=arm,
                           updates=updates, bands_target=bands, edge_target=edge,
                           protocol=label, lane=lane))
    return result


def recipe(original_five, updates, bands, edge):
    result = deepcopy(original_five)
    result.update(name=NAME, train_cases=original_five['train_cases'],
                  support_selection='Original frozen five-volume training support',
                  updates=updates, learning_rate_horizon_updates=200,
                  gradient_ratio_target=bands, gradient_ratio_p95_cap=5*bands,
                  gradient_ratio_targets={'bands': bands, 'edge': edge or .01},
                  gradient_ratio_p95_caps={'bands': 5*bands, 'edge': 5*(edge or .01)},
                  sampling=f'{updates} updates, two adjacent-slice pairs per update; '
                           'shared seed and first-200-update learning-rate schedule')
    return result


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    parent = ROOT/'reports'/PARENT/'frozen'
    five = ROOT/'reports'/FIVE/'frozen'
    if digest(parent/'manifest.json') != PARENT_SHA or digest(five/'manifest.json') != FIVE_SHA:
        raise ValueError('Frozen predecessors changed')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Parent source changed: '+name)
    stage.mkdir(parents=True)
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('foundation.py', 'storage.py', 'sam3_five_volume_updates_job.py'):
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/name,
                     stage/'source/scripts/cross_dataset_campaign'/name)
    for name in ('source_version.json', 'original_recipe.json'):
        shutil.copy2(parent/name, stage/name)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copy2(five/'manifest.json', stage/'five_volume_manifest.json')
    shutil.copy2(five/'protocols/GeoFWI3D/protocol.json', stage/'original_five_volume_protocol.json')
    original = read(stage/'original_five_volume_protocol.json')
    for updates, arm, bands, edge in specifications():
        write(stage/'protocols'/protocol_name(updates, bands, edge)/'protocol.json',
              recipe(original, updates, bands, edge))
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=len(tasks()), single_allocation=True,
        authorization='User requested one maximum-24-hour SAM3 job with maximum GPU workers, '
                      'five original training volumes, usual validation cohort, variable update counts, '
                      'and gentler bands+edge variants that seek consistent improvement.',
        exploration_followup='No automatic continuation beyond this allocation',
        comparison_policy='At each stopping update, compare bands and gentle-edge variants to '
                          'the native control and matched bands-only run on the same 96 volumes. '
                          'Report mean Dice, paired case wins/losses, lower-tail changes and FN/FP.',
        calibration_policy='Bands target 10/20/40%; edge target 0.5/1/2/5% with its own p95 cap. '
                           'Targets are LoRA gradient ratios against the native SAM3 loss, not Dice multipliers.',
        parallelism=dict(workers=2, lanes=[0,1], gpus_per_worker=1, cpus_per_worker=4,
                         gpu_type='4g.40gb', allocation_cpu_count=8, allocation_memory_gib=80),
        max_job_hours=24, graceful_stop_seconds=85200, archive_wait_timeout_seconds=300,
        analysis=dict(primary='Validation Dice versus matched native and bands-only controls',
                      diagnostics=['training Dice', 'foreground FN/FP', 'paired per-volume Dice changes',
                                   'fraction of validation volumes improved', 'lower-tail paired changes',
                                   'realized calibration ratios and gradient clipping']))
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), minimum_free_gib=10,
                           estimated_peak_logical_gib=4, mirrored_peak_gib=8, reserve_gib=2)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    batch = (parent/'run.sbatch').read_text()
    batch = batch.replace('hippo-sam3-4vol-strength', 'hippo-sam3-5vol-updates')
    batch = batch.replace('scripts.cross_dataset_campaign.sam3_four_volume_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen five-volume experiment changed: '+name)
    if digest(stage/'parent_manifest.json') != PARENT_SHA or digest(stage/'five_volume_manifest.json') != FIVE_SHA:
        raise ValueError('Parent provenance changed')
    parent = read(stage/'parent_manifest.json')
    for name, sha in parent.items():
        if name.startswith('source/') and name not in (
            'source/scripts/cross_dataset_campaign/foundation.py',
            'source/scripts/cross_dataset_campaign/storage.py') and digest(stage/name) != sha:
            raise ValueError('Predecessor source changed: '+name)
    five = read(stage/'five_volume_manifest.json')
    if digest(stage/'original_five_volume_protocol.json') != five['protocols/GeoFWI3D/protocol.json']:
        raise ValueError('Original five-volume support changed')
    original = read(stage/'original_five_volume_protocol.json')
    if len(original['train_cases']) != 5 or len(original['evaluation_cases']) != 96:
        raise ValueError('Wrong original cohort')
    for updates, arm, bands, edge in specifications():
        p = read(stage/'protocols'/protocol_name(updates, bands, edge)/'protocol.json')
        if p != recipe(original, updates, bands, edge):
            raise ValueError('Unrequested scientific change')
        if (len(set(p['train_cases'])) != 5 or len(set(p['evaluation_cases'])) != 96
                or set(p['train_cases']) & set(p['evaluation_cases'])
                or (p['fold'], p['seed'], p['augmentation']) != (0, 0, 'none')
                or p['learning_rate_horizon_updates'] != 200
                or p['calibration_slabs_per_volume'] != 4):
            raise ValueError('Wrong cohort or update schedule')
    plan = read(stage/'plan.json')
    if (plan['tasks'] != tasks() or plan['num_runs'] != 24 or plan['name'] != NAME
            or plan['parallelism']['workers'] != 2 or plan['parallelism']['lanes'] != [0,1]
            or not plan['single_allocation'] or plan['max_job_hours'] != 24
            or plan['storage']['minimum_free_gib'] < 10):
        raise ValueError('Incorrect execution contract')
    return plan


def preflight(stage):
    plan = verify(stage)
    p = read(stage/'original_five_volume_protocol.json')
    m = p['manifest']
    if p['train_cases'] != m['splits']['training_subsets']['5pct'][:5] or p['evaluation_cases'] != m['splits']['validation']:
        raise ValueError('Frozen training/validation split changed')
    if digest(plan['base_staging']) != p['base_sha256']:
        raise ValueError('Base checkpoint changed')
    for case in p['train_cases']+p['evaluation_cases']:
        for kind in ('image', 'label'):
            if digest(Path(p['data'])/m['cases'][case][kind]) != m['cases'][case][kind+'_sha256']:
                raise ValueError('Image/annotation changed: '+case)
    hashes = read(stage/'source_version.json')['upstream_provenance']['python_source_sha256']
    for name, sha in hashes.items():
        if digest(Path(p['upstream'])/name) != sha:
            raise ValueError('SAM3 upstream source changed')
    import torch
    from scripts.cross_dataset_campaign.foundation import Volumes, SupportSlabs
    torch.set_num_threads(2)
    slabs = SupportSlabs(Volumes({**p, 'arm':'dice_bands_edge'}), {**p, 'arm':'dice_bands_edge'}, None)
    if len(slabs.inventory) != 5 or len(slabs.calibration_schedule(4)) != 20:
        raise ValueError('Five-volume calibration support differs')
    for updates in UPDATES:
        protocol = read(stage/'protocols'/protocol_name(updates, .2, None)/'protocol.json')
        schedule = slabs.schedule(protocol)
        if len(schedule) != 2*updates or {case for case, _ in schedule} != set(p['train_cases']):
            raise ValueError('Invalid sampling schedule')
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    smoke = subprocess.run(['sbatch', '--test-only', str(stage/'run.sbatch'), str(stage)],
                           text=True, capture_output=True, check=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise ValueError('Insufficient personal storage')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        quota=usage, support=p['train_cases'], validation_volumes=96, runs=len(tasks()),
        checks=['Frozen five-volume support and 96-volume validation cohort',
                'All source data and pretrained weights hash verified',
                'Shared fixed-200 learning-rate horizon across stopping points',
                'Two isolated GPU workers, the maximum allowed by the stud QoS',
                'Per-run verified off-BeeGFS archive before checkpoint cleanup'],
        scheduler_test=smoke.stdout+smoke.stderr))


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
        print('Five-volume SAM3 preflight passed')
    elif args.command == 'submit':
        print(submit(stage))
    elif args.command == 'status':
        print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        from scripts.cross_dataset_campaign import sam3_four_volume_parallel as parallel
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__':
    main()
