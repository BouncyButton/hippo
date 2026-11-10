"""Materialize the long-study draft without selecting pilot-dependent settings.

This command never submits jobs. Missing CPU profiles leave explicit blockers;
rerunning after profiling fills those schedules without touching exploration.
"""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, SEISMIC_CAP, REMOTE, NAME, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, REFERENCE, ADNI, REPORT, schedule, freeze_source

FULL_NAME = 'cross_dataset_full_20261014'
STAGE = ROOT/'experiments'/FULL_NAME
PROFILES = ROOT/'reports'/FULL_NAME/'profiles'
PILOT = ROOT/'experiments'/NAME


def splits():
    msd = read(REFERENCE/'nnunet/splits_10patients.json')
    provenance = read(ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013/split_provenance.json')
    by_case = provenance['case_to_subject']
    adni, support, subject_provenance = [], [], []
    for fold, full in enumerate(read(ROOT/'datasets/Dataset103_ADNI/splits_final.json')[:3]):
        people = sorted({by_case[c] for c in full['train']})
        validation = sorted({by_case[c] for c in full['val']})
        if set(people) & set(validation):
            raise ValueError('ADNI subject leakage')
        chosen = sorted(np.random.RandomState(0).permutation(people)[:int(.05*len(people))].tolist())
        train = sorted(c for c in full['train'] if by_case[c] in chosen)
        row = dict(train=train, val=sorted(full['val']))
        selected = [sorted(c for c in train if by_case[c] == person)[i % 2] for i, person in enumerate(chosen)]
        if len(train) != 10 or len(selected) != 5 or len(row['val']) != 54:
            raise ValueError('ADNI support count differs')
        adni.append(row); support.append(selected)
        subject_provenance.append(dict(fold=fold, subset_seed=0, train_subjects=chosen,
            validation_subjects=validation, conventional_cases=train, foundation_cases=selected,
            selection='Same saved-fold subject sampling rule as the fold-0 pilot; independent of model seed'))
    if adni[0] != read(ADNI/'nnunet/splits_10patients.json')[0]:
        raise ValueError('ADNI fold 0 no longer matches exploration')
    seismic_manifests = [read(ROOT/f'reports/geofwi3d_replication_20261011/cluster/folds/fold_{f}/manifest.json') for f in range(3)]
    seismic = [dict(train=m['splits']['training_subsets']['5pct'], val=m['splits']['validation']) for m in seismic_manifests]
    for rows, count in ((msd,10),(adni,10),(seismic,19)):
        for row in rows:
            if len(set(row['train'])) != count or set(row['train']) & set(row['val']):
                raise ValueError('Invalid full-study split')
    write(STAGE/'splits.json',dict(MSD=msd, ADNI=adni, GeoFWI3D=seismic))
    write(STAGE/'adni_subject_provenance.json',subject_provenance)
    return dict(MSD=msd, ADNI=adni, GeoFWI3D=seismic), support, seismic_manifests


def profile_path(dataset, model):
    if dataset == 'GeoFWI3D':
        return PROFILES/f'geofwi3d_{model}.json'
    if model == 'nnunet':
        return REPORT/'profiles/msd_nnunet_dice.json' if dataset == 'MSD' else PROFILES/'adni_nnunet_dice.json'
    return (ROOT/f'reports/capped_depth_delay_31976_20261006/profiles/{model}.json'
            if dataset == 'MSD' else ADNI/model/'profile.json')


def conventional(dataset, model, fold, target, all_splits, missing):
    origin = (REFERENCE if dataset == 'MSD' else ADNI)/model
    p = deepcopy(read(origin/'protocol.json'))
    tag = f'{dataset}_{model}_f{fold}_target{round(target*100)}'
    folder = STAGE/'protocols'/tag
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(origin/'nnunet_architecture.json',folder/'nnunet_architecture.json')
    write(folder/'splits_10patients.json',all_splits[dataset])
    cap = SEISMIC_CAP if dataset == 'GeoFWI3D' else CAP
    count = 19 if dataset == 'GeoFWI3D' else 10
    p.update(dataset=dataset,augmentation='none',folds=[fold],seeds=[0,1,2],train_count=count,
        training_loss='Dice-only for all models; nnUNet retains native deep-supervision weights',
        calibration_target_ratio=target,calibration_p95_cap=.75 if target == .6 else .5,
        budget_flops=cap,budget_gflops=cap/1e9,validation_target_flops=[cap*i//20 for i in range(1,21)],
        training_source_sha256={},requested_arm_order=list(ARMS),
        checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
        classes=['background','anterior','posterior'] if dataset == 'MSD' else ['background','hippocampus' if dataset == 'ADNI' else 'salt'])
    if dataset == 'GeoFWI3D':
        descriptor=dict(dataset_root=str(REMOTE/'Dataset203_GeoFWI3D'),
            manifest=str(REMOTE/f'geofwi3d_3fold_3seed_5pct_500updates_20261011_01/folds/fold_{fold}/manifest.json'))
        write(folder/'dataset.json',descriptor)
        p.update(data=str(REMOTE/FULL_NAME/'protocols'/tag/'dataset.json'),pkl_sha256=digest(folder/'dataset.json'),
            num_classes=2,spatial_size=[96,96,96],budget_policy='152 nnUNet updates: eight complete passes over 19 volumes',
            preprocessing='Native 96-cubed normalized seismic volume, no spatial resampling')
    else:
        p['data']=str(REMOTE/('hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl' if dataset == 'MSD' else 'hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'))
    if dataset != 'MSD':
        cases={c for row in all_splits[dataset] for c in row['train']+row['val']}
        p['native_geometry']={c:dict(model_affine=np.eye(4).tolist()) for c in sorted(cases)}
    profile_file = profile_path(dataset,model)
    if profile_file.exists():
        profile = read(profile_file)
        alias = p['trainer_alias']; old = p['models'][alias]['objectives']
        p['models'][alias]['parameters']=profile.get('parameters',p['models'][alias]['parameters'])
        p['models'][alias]['objectives']={arm:schedule(profile['costs'],arm,old[arm],count,cap) for arm in ARMS}
        write(folder/'profile.json',profile)
        p['profile_sha256']=digest(folder/'profile.json')
        p['profile_status']='measured'
    else:
        missing.add(str(profile_file))
        p['models'][p['trainer_alias']]['objectives']={}
        p['profile_status']='pending_measurement'
        p['profile_sha256']=None
    p['splits_sha256']=digest(folder/'splits_10patients.json')
    p['nnunet_architecture_sha256']=digest(folder/'nnunet_architecture.json')
    write(folder/'protocol.json',p)
    return tag


def foundation(dataset,fold,seed,all_splits,support,seismic_manifests):
    p=deepcopy(read(PILOT/'protocols'/f'{"ADNI" if dataset == "MSD" else dataset}_foundation_fold0_seed0/protocol.json'))
    p.update(dataset=dataset,fold=fold,seed=seed,augmentation='none')
    if dataset == 'MSD':
        reference=read(ROOT/f'docs/experiments/medsam3_multiclass_replication_20260930/protocols/fold{fold}_seed{seed}.json')
        p.update(reference)
        p.update(data=str(REMOTE/'hippo/datasets/Dataset101_MSD'),prompts=reference['class_prompts'],
            initialization='medical_pretrained_LoRA',support_selection='Exact existing MSD three-class replication support',
            geometry_units='canonical source voxels; depth/SDF extension requires geometry review before release')
        p.pop('dataset_sha256',None);p.pop('support_subjects',None)
        p['requested_arms']=[a for a in ARMS if a not in ('dice','dice_bands','dice_bands_edge')]
    elif dataset == 'ADNI':
        p.update(train_cases=support[fold],evaluation_cases=all_splits['ADNI'][fold]['val'])
        provenance=read(STAGE/'adni_subject_provenance.json')[fold]
        p['support_subjects']=provenance['train_subjects'];p['requested_arms']=list(ARMS)
    else:
        p.update(train_cases=all_splits[dataset][fold]['train'][:5],evaluation_cases=all_splits[dataset][fold]['val'],
            manifest=seismic_manifests[fold],requested_arms=list(ARMS))
    if len(p['train_cases']) != 5 or set(p['train_cases']) & set(p['evaluation_cases']):
        raise ValueError('Foundation support mismatch')
    tag=f'{dataset}_foundation_f{fold}s{seed}'
    write(STAGE/'protocols'/tag/'protocol.json',p)
    return tag


def main():
    STAGE.mkdir(parents=True,exist_ok=True)
    if (STAGE/'submission.json').exists():
        raise ValueError('Cannot modify a submitted long study')
    all_splits,support,manifests=splits()
    missing=set();lookup={}
    for dataset in all_splits:
        for model in MODELS:
            for fold in range(3):
                targets=(.1,.4,.6) if model == 'nnunet' else (.1,)
                lookup[(dataset,model,fold)]={str(target):conventional(dataset,model,fold,target,all_splits,missing) for target in targets}
    plan=deepcopy(read(PILOT/'full_study_plan.json'))
    runtime=read(PILOT/'plan.json')
    for key in ('foundation_python','conventional_python','base_staging','base_sha256',
                'max_job_hours','graceful_stop_seconds'):
        plan[key]=runtime[key]
    plan['storage']=dict(minimum_free_gib=10,estimated_peak_logical_gib=4,
        mirrored_peak_gib=8,reserve_gib=2,archive=str(ROOT/'reports'/FULL_NAME),
        policy='Per-experiment verified local archive before bulky artifact retirement')
    for task in plan['tasks']:
        task['id']=f"{task['dataset']}_{task['model']}_{task['arm']}_f{task['fold']}s{task['seed']}"
        if task['kind']=='conventional':
            alternatives=lookup[(task['dataset'],task['model'],task['fold'])]
            task['protocol_candidates']=alternatives
            task['protocol']=None if task['model']=='nnunet' else alternatives['0.1']
        else:
            task['protocol']=foundation(task['dataset'],task['fold'],task['seed'],all_splits,support,manifests)
    if len({t['id'] for t in plan['tasks']}) != 1863:
        raise ValueError('Duplicate full-study identities')
    plan.update(name=FULL_NAME,status='materialized_draft_pending_exploration',submission_enabled=False,
        required_preparation=sorted(missing),
        release_checks=['Select nnUNet target from completed exploration',
            'Review ADNI/seismic foundation adaptation and constraint benefit',
            'Review seismic fitting at the eight-pass cap',
            'Review MSD foundation SDF/depth geometry and compatibility with earlier replication',
            'Freeze selected protocols, verify cluster inputs, and authorize full submission'],
        budget_flops_by_dataset={'MSD':CAP,'ADNI':CAP,'GeoFWI3D':SEISMIC_CAP})
    write(STAGE/'plan.json',plan)
    batch=(ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text()
    batch=batch.replace('hippo-exploration-noaug','hippo-full-study-noaug')
    batch=batch.replace('scripts.cross_dataset_campaign.controller run','scripts.cross_dataset_campaign.full_controller run')
    (STAGE/'run.sbatch').write_text(batch)
    freeze_source(STAGE)
    write(STAGE/'control.json',dict(pause=True,full_submission_enabled=False))
    write(STAGE/'manifest.json',{str(p.relative_to(STAGE)):digest(p) for p in sorted(STAGE.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json','control.json')})
    write(ROOT/'reports'/FULL_NAME/'preparation.json',dict(stage=str(STAGE),submission_enabled=False,
        cells=1863,protocols=len(list((STAGE/'protocols').glob('*/protocol.json'))),
        missing_profiles=sorted(missing),manifest_sha256=digest(STAGE/'manifest.json')))
    print('Prepared 1,863 cells; missing profiles:',len(missing),'; submission disabled',flush=True)


if __name__=='__main__':main()
