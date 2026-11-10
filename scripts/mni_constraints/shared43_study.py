"""Finite 14-condition, three-fold, three-seed extension of the 43T MNI recipes."""
from copy import deepcopy
import math
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import ARMS, REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.capped_depth_delay.schedule import training_flops
from scripts.mni_constraints import campaign, headroom_bands
from scripts.unetr_subfields.contracts import ROOT, subject

NAME = 'mni_n10_constraints14_shared43t_3fold3seed_20261107'
MODULE = 'scripts.mni_constraints.shared43_study'
PARENT = 'mni_n10_four_models_shared43t_bands_f0s0_20261107'
CAP = 43006868505381
MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')
TARGETS = dict(zip(MODELS, (.6, .1, .05, .1)))


def matrix():
    # Exercise complete objectives first, then fill the matched controls and remaining arms.
    arms = [ARMS[-1], *ARMS[:-1]]
    return [dict(id=f'MNI_{m}_{a}_f{f}s{s}', kind='conventional', dataset='MNI',
                 model=m, arm=a, fold=f, seed=s, protocol=m+'_'+a,
                 augmentation='xy_d4_v1' if m in MODELS[1:3] else 'none',
                 train_subjects=10, target=TARGETS[m] if a != 'dice' else None)
            for a in arms for s in range(3) for f in range(3) for m in MODELS]


def is_reused(task):
    return task['fold'] == task['seed'] == 0 and task['arm'] in ('dice', 'dice_bands')


def prepare(_target=None):
    import numpy as np
    stage = ROOT/'experiments'/NAME
    if (stage/'manifest.json').exists():
        return verify(stage)
    parent = ROOT/'experiments'/PARENT
    for relative, sha in read(parent/'manifest.json').items():
        assert digest(parent/relative) == sha, relative
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('shared43_study.py', 'collect_shared43_study.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name, stage/'source/scripts/mni_constraints'/name)
    source_splits = ROOT/'datasets/Dataset102_MNI/splits_final.json'
    depth_settings = read(ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json')['geometry']
    splits = []
    provenance = []
    for fold, full in enumerate(read(source_splits)[:3]):
        order = np.random.RandomState(0).permutation(sorted({subject(c) for c in full['train']})).tolist()
        splits.append(dict(train=sorted(c for c in full['train'] if subject(c) in set(order[:10])), val=full['val']))
        provenance.append(dict(fold=fold, subject_order=order, selected_subjects=order[:10]))
    write(stage/'split_provenance.json', dict(source_sha256=digest(source_splits), folds=provenance,
        selection='First10 subjects of RandomState(0) permutation of each historical fold training pool; fixed across model seeds. Both hippocampi stay together.'))
    reused = []
    for model in MODELS:
        origin_root = parent/('reused_protocols' if model == 'nnunet' else 'protocols')
        old_band = read(origin_root/(model+'_n10_original_bands')/'protocol.json')
        alias = old_band['trainer_alias']
        band_schedule = old_band['models'][alias]['objectives']['dice_bands']
        profile = read(ROOT/'reports/mni_constraints14_cap8_20261107/profiles'/f'{model}.json')
        assert profile['objective_sha256'] == digest(stage/'source/scripts/mni_constraints/objectives.py')
        assert profile['costs']['dice']['update_flops'] == band_schedule['baseline_flops_per_update']
        for arm, cost in profile['costs'].items():
            if arm == 'dice':
                continue
            values = {r['composed_flops'] for r in profile['profiles'] if r['calibration'] and r['arm'] == arm}
            assert len(values) == 1
            cost['calibration_flops'] = 20*values.pop()+8192
        profile['calibration_cases'] = 20
        profile['calibration_accounting'] = 'Measured per-case graph times20 training crops, plus8192 host-selection allowance; outside optimization cap.'
        adjusted = deepcopy(profile['costs'])
        for arm, cost in adjusted.items():
            cost['update_flops'] += int('bands' in arm)+int('edge' in arm)
        ramp = band_schedule['constraint_warmup_updates']
        timing = dict(prefix_updates=band_schedule['prefix_updates'], constraint_warmup_updates=ramp,
                      sdf_prefix_updates=band_schedule['prefix_updates']+1)
        for arm in ARMS:
            condition = 'baseline' if arm == 'dice' else 'original_bands'
            origin = origin_root/(model+'_n10_'+condition)
            old = read(origin/'protocol.json')
            assert read(origin/'splits_10patients.json')[0] == splits[0]
            folder = stage/'protocols'/(model+'_'+arm)
            folder.mkdir(parents=True)
            shutil.copy2(origin/'nnunet_architecture.json', folder/'nnunet_architecture.json')
            write(folder/'splits_10patients.json', splits)
            write(folder/'profile.json', profile)
            s = (deepcopy(old['models'][alias]['objectives'][arm]) if arm in ('dice', 'dice_bands')
                 else schedule(adjusted, arm, timing, 20, CAP))
            p = deepcopy(old)
            horizon = 200 if model == 'nnunet' else s['max_steps']
            warmup = 0 if model == 'nnunet' else 24 if model == 'unetr_vit_b16' else math.ceil(.05*horizon)
            p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'), folds=[0,1,2], seeds=[0,1,2],
                budget_flops=CAP, constraints=arm.split('_')[1:], requested_arm_order=[arm],
                models={alias:dict(parameters=profile['parameters'], objectives={arm:s})},
                splits_sha256=digest(folder/'splits_10patients.json'), profile_sha256=digest(folder/'profile.json'),
                constraint_warmup_updates=s['constraint_warmup_updates'], lr_horizon_updates=horizon,
                lr_schedule_details=dict(policy=p['lr_policy'], horizon_updates=horizon, warmup_updates=warmup,
                    decay_start_index=horizon//2 if p['lr_policy']=='hold_half' else warmup),
                calibration_target_ratio=TARGETS[model], calibration_p95_cap=.75,
                baseline_only=arm=='dice', adaptive_variant=condition, paired_baseline_predictions=False,
                train_count=20, train_subjects=10, validation_count=10,
                depth_geometry=depth_settings,
                budget_origin='Fixed shared43006868505381 optimization FLOPs from the accepted121-update nnUNet stopping point.',
                learning_rate_policy='Preserve nnUNet200-update LR horizon/100-update hold; other models retain their recipe with per-arm cap-matched horizon.',
                timing_policy='Preserve original bands prefix/ramp (nnUNet20/20; others10% baseline updates). SDF begins nearest5/9 of optimization cap with its own same-length ramp.',
                constraint_recipe='Original balanced union bands, signed face edge consistency, three-subfield SDF, and union-geometry weighted CE; historical14-condition matrix.',
                calibration_policy='All20 training crops only, sequential bands/edge/sdf/depth calibration against base plus prior terms; model-specific target and75% p95 guard. Separate calibration FLOPs.',
                augmentation_policy=p['augmentation'],
                primary_metric='Final-budget per-volume foreground macro Dice; average all nine fold/seed cells equally.',
                decision_rule='Fixed folds0/1/2 and seeds0/1/2; no score-dependent replacement, stopping or recipe changes.',
                spatial_diagnostics='Unchanged quarter-budget error coverage and bands gradient probes; probes do not measure the full combined constraint gradient.',
                checkpoint_selection='Final-budget checkpoint primary; best validation secondary; no early stopping.',
                parent_protocol_sha256=digest(origin/'protocol.json'), parent_stage=PARENT)
            for key in ('split_provenance_sha256', 'trajectory_reference', 'trajectory_reference_sha256'):
                p.pop(key, None)
            write(folder/'protocol.json', p)
            if arm not in ('dice', 'dice_bands'):
                continue
            task = next(t for t in matrix() if t['model']==model and t['arm']==arm and is_reused(t))
            legacy = f'MNI_{model}_n10_{condition}_f0s0'
            archive = ROOT/'reports'/PARENT/('frozen/reused_comparisons' if model=='nnunet' else 'runs')/legacy
            verify_archive(archive, read(archive/'archive_manifest.json'))
            assert read(archive/'cleanup.json')['status'] == read(archive/'completion.json')['status'] == 'complete'
            config = read(archive/'config.json')
            assert config['train_cases']==splits[0]['train'] and config['validation_cases']==splits[0]['val']
            assert config['schedule']==s and config['budget_flops']==CAP and config['lr_horizon_updates']==horizon
            assert config['initial_lr']==p['initial_lrs'][alias] and config['augmentation']==p['augmentation']
            assert config['dataset_sha256']==digest(stage/'data/MNI.pkl')
            assert config['protocol_sha256']==digest(origin/'protocol.json')
            assert config['script_sha256']==digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
            for key in ('training_loss','lr_policy','initial_lrs','augmentation','calibration_p95_cap','calibration_target_ratio'):
                if arm=='dice' and key.startswith('calibration_'):continue
                assert old[key]==p[key], (model,arm,key)
            shutil.copytree(archive, stage/'reused_runs'/task['id'])
            shutil.copytree(origin, stage/'reuse_protocols'/task['protocol'])
            reused.append(dict(task=task, source_archive=str(archive),
                archive_manifest_sha256=digest(archive/'archive_manifest.json'),
                source_protocol_sha256=digest(origin/'protocol.json')))
    plan = dict(name=NAME, tasks=[t for t in matrix() if not is_reused(t)], models=list(MODELS),
        conditions=list(ARMS), logical_conditions=list(ARMS), logical_cells=504, logical_runs=504,
        num_runs=496, reused_runs=8, budget_flops=CAP, folds=[0,1,2], seeds=[0,1,2],
        training_subject_counts=[10], calibration_targets=TARGETS, gpu_workers=2,
        excluded_nodes=['gnode04'], augmentation='model_specific',
        conventional_python='/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python',
        graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=12, archive=str(ROOT/'reports'/NAME)),
        authorization='Extend the accepted shared-cap recipes to14 conditions,3 folds and3 seeds on all4models; twoGPUworkers; maxone running/one queued allocation. Finite continuation only; archive, verify and clean everyrun.',
        selection_caveat='Development validation folds used in earlier recipe exploration; this is not an untouched test set.')
    write(stage/'plan.json', plan)
    write(stage/'reused_runs.json', reused)
    write(stage/'control.json', dict(pause=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Training runtime unchanged. Extend fixed splits/matrix; full measured objectives at shared43T; preserve nnUNet LR trajectory; hash-verified reuse of8exact cells.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.sample_size', MODULE).replace('hippo-mni-samples','hippo-mni-14'))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name)==sha, name
    plan = read(stage/'plan.json')
    assert plan['tasks']==[t for t in matrix() if not is_reused(t)]
    assert len(plan['tasks'])==len({t['id'] for t in plan['tasks']})==496
    reuse=read(stage/'reused_runs.json')
    assert len(reuse)==8 and {x['task']['id'] for x in reuse}=={t['id'] for t in matrix() if is_reused(t)}
    reference_splits=None
    for folder in (stage/'protocols').iterdir():
        p=read(folder/'protocol.json');arm=p['requested_arm_order'][0]
        s=p['models'][p['trainer_alias']]['objectives'][arm]
        assert p['budget_flops']==CAP and p['folds']==p['seeds']==[0,1,2]
        assert training_flops(s['max_steps'],s)<=CAP<training_flops(s['max_steps']+1,s)
        assert p['lr_horizon_updates']==(200 if p['model_identity']=='nnunet' else s['max_steps'])
        assert p['calibration_target_ratio']==TARGETS[p['model_identity']] and p['calibration_p95_cap']==.75
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        splits=read(folder/'splits_10patients.json')
        if reference_splits is None:reference_splits=splits
        assert splits==reference_splits and len(splits)==3
        for split in splits:
            assert len(split['train'])==20 and len(split['val'])==10
            for key,count in [('train',10),('val',5)]:
                assert len(set(split[key]))==2*count and len({subject(c) for c in split[key]})==count
                assert all(sum(subject(c)==v for c in split[key])==2 for v in {subject(c) for c in split[key]})
            assert not {subject(c) for c in split['train']}&{subject(c) for c in split['val']}
        if arm!='dice':
            assert s['prefix_updates']<s['max_steps'] and s['sdf_prefix_updates']<s['max_steps']
            profile=read(folder/'profile.json')
            costs={r['composed_flops'] for r in profile['profiles'] if r['calibration'] and r['arm']==arm}
            assert len(costs)==1 and s['calibration_flops']==20*costs.pop()+8192
    for item in reuse:
        folder=stage/'reused_runs'/item['task']['id']
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==item['archive_manifest_sha256']
    return plan


def preflight(stage):
    verify(stage)
    from scripts.mni_constraints.verify import check_stage
    result=check_stage(stage)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),**result))
    return read(stage/'preflight.json')


def main():
    campaign.MODULE=MODULE
    campaign.prepare=prepare
    campaign.verify=verify
    campaign.preflight=preflight
    campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':
    main()
