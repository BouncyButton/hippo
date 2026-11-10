"""Freeze the exploration and a separately gated 1,863-cell full-study plan."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, SEISMIC_CAP, REMOTE, NAME, read, write, digest

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'reports/capped_replication_31976_20261007/verified_metadata_20261010/frozen/source/experiments/capped_replication_31976_20261007'
ADNI=ROOT/'experiments/adni_constraints_noaug_5pct_fold0_seed0_20261013/protocols'
REPORT=ROOT/'reports'/NAME


def schedule(costs, arm, old, count, budget=CAP):
    baseline=costs['dice']['update_flops']
    prefix=old['prefix_updates'];ramp=old['constraint_warmup_updates']
    full=costs[arm]['update_flops']+int('depth' in arm)+2*int('sdf' in arm)
    early=costs[arm.replace('_sdf','')]['update_flops']+int('depth' in arm)
    late=prefix
    delayed='sdf' in arm and old['sdf_prefix_updates']>old['prefix_updates']
    if delayed:
        candidate=prefix+max(0,budget*5//9-prefix*baseline)//early
        late=min((candidate,candidate+1),key=lambda n:(abs(prefix*baseline+(n-prefix)*early-budget*5//9),n))
    elif 'sdf' in arm:
        early=full
    if arm=='dice':
        prefix=late=0;early=full=baseline
    def consumed(n):
        return min(n,prefix)*baseline+max(0,min(n,late)-prefix)*early+max(0,n-late)*full
    maximum=0
    while consumed(maximum+1)<=budget:
        maximum+=1
    return dict(prefix_updates=prefix,sdf_prefix_updates=late,baseline_flops_per_update=baseline,
        early_flops_per_update=early,flops_per_update=full,calibration_flops=costs[arm]['calibration_flops'],
        constraint_warmup_updates=ramp,sdf_warmup_updates=ramp,budget_scope='optimization_only',
        max_steps=maximum,consumed_training_flops=consumed(maximum),consumed_flops=consumed(maximum),
        consumed_total_flops=consumed(maximum)+costs[arm]['calibration_flops'],unused_flops=budget-consumed(maximum),
        patient_pass_epochs=maximum/count)


def conventional_protocol(stage,dataset,identity,target=.1):
    origin=(REFERENCE if dataset=='MSD' else ADNI)/identity
    parent=read(origin/'protocol.json')
    protocol=deepcopy(parent)
    tag=f'{dataset}_{identity}_target{int(target*100)}'
    folder=stage/'protocols'/tag;folder.mkdir(parents=True,exist_ok=True)
    for filename in ('splits_10patients.json','nnunet_architecture.json'):
        shutil.copyfile(origin/filename,folder/filename)
    alias=protocol['trainer_alias']
    protocol.update(dataset=dataset,augmentation='none',seeds=[0,1,2],folds=[0,1,2],train_count=10,
        training_loss='Dice-only for all models; nnUNet retains native deep-supervision weights',
        calibration_target_ratio=target,calibration_p95_cap=.75 if target==.6 else .5,
        checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
        training_source_sha256={},restart_policy='Resume full optimizer, RNG and sampling state',
        classes=['background','anterior','posterior'] if dataset=='MSD' else ['background','hippocampus'])
    if dataset=='MSD':
        profile=read(REPORT/'profiles/msd_nnunet_dice.json') if identity=='nnunet' else read(ROOT/f'reports/capped_depth_delay_31976_20261006/profiles/{identity}.json')
        protocol['data']=str(REMOTE/'hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    else:
        profile=read(origin/'profile.json')
        protocol['data']=str(REMOTE/'hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl')
    if dataset=='GeoFWI3D':
        profile=read(REPORT/f'profiles/{identity}.json')
        if profile.get('loss_recipe')!='dice_only':
            raise ValueError('Seismic profile must use the confirmed Dice-only recipe')
        manifest=read(ROOT/'reports/geofwi3d_replication_20261011/cluster/folds/fold_0/manifest.json')
        split=dict(train=manifest['splits']['training_subsets']['5pct'],val=manifest['splits']['validation'])
        write(folder/'splits_10patients.json',[split])
        descriptor=dict(dataset_root=str(REMOTE/'Dataset203_GeoFWI3D'),
            manifest=str(REMOTE/'geofwi3d_3fold_3seed_5pct_500updates_20261011_01/folds/fold_0/manifest.json'))
        write(folder/'dataset.json',descriptor)
        protocol.update(data=str(REMOTE/NAME/'protocols'/tag/'dataset.json'),
            pkl_sha256=digest(folder/'dataset.json'),train_count=19,num_classes=2,spatial_size=[96,96,96],
            budget_flops=SEISMIC_CAP,budget_gflops=SEISMIC_CAP/1e9,
            validation_target_flops=[SEISMIC_CAP*i//20 for i in range(1,21)],
            budget_policy='Common seismic cap set by 152 Dice-only nnUNet updates: eight passes over 19 volumes',
            classes=['background','salt'],native_geometry={c:dict(model_affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for c in split['train']+split['val']},
            requested_arm_order=['dice'])
        # Only baseline pilots are authorized at this stage. Full constraints
        # require separately measured 96-cube loss and calibration graphs.
        cost=profile['costs']['dice']['update_flops']
        protocol['models'][alias]['parameters']=profile['parameters']
        protocol['models'][alias]['objectives']={'dice':schedule(profile['costs'],'dice',dict(prefix_updates=0,sdf_prefix_updates=0,constraint_warmup_updates=1),19,SEISMIC_CAP)}
    else:
        old=protocol['models'][alias]['objectives']
        protocol['models'][alias]['objectives']={arm:schedule(profile['costs'],arm,old[arm],10) for arm in ARMS}
    protocol['splits_sha256']=digest(folder/'splits_10patients.json')
    protocol['nnunet_architecture_sha256']=digest(folder/'nnunet_architecture.json')
    write(folder/'profile.json',profile)
    protocol['profile_sha256']=digest(folder/'profile.json')
    write(folder/'protocol.json',protocol)
    return tag


def foundation_protocol(stage,dataset):
    reference=read(ROOT/'docs/experiments/medsam3_multiclass_replication_20260930/protocols/fold0_seed17.json')
    p={k:reference[k] for k in ('base_sha256','updates','accumulation','warmup_updates','learning_rate','final_learning_rate',
         'calibration_slabs_per_volume','constraint_warmup_updates','gradient_ratio_target','gradient_ratio_p95_cap','calibration_seed')}
    runtime=REMOTE/'medsam3_runtime_20260928'
    p.update(dataset=dataset,fold=0,seed=0,augmentation='none',classes=['background','hippocampus' if dataset=='ADNI' else 'salt'],
        upstream=str(runtime/'MedSAM3'),lora_weights=str(runtime/'weights/best_lora_weights.pt'),
        lora_sha256=digest(ROOT/'MedSam3_weights/best_lora_weights.pt'),
        prompts={'1':'hippocampus' if dataset=='ADNI' else 'salt body'},
        initialization='medical_pretrained_LoRA' if dataset=='ADNI' else 'SAM3_with_fresh_zero_output_LoRA',
        sampling='200 updates, two uniformly sampled adjacent-slice pairs per update; identical schedule across arms',
        inference='Whole-volume reconstruction from image and fixed text prompt only',
        geometry_units='source array voxels; no anatomical orientation assumed')
    if dataset=='ADNI':
        provenance=read(ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013/split_provenance.json')
        split=read(ADNI/'nnunet/splits_10patients.json')[0]
        selected=[]
        for i,subject in enumerate(provenance['train_subjects']):
            names=sorted(c for c in split['train'] if provenance['case_to_subject'][c]==subject)
            selected.append(names[i%2])
        if len(set(selected))!=5: raise ValueError('Expected five distinct ADNI crops')
        p.update(train_cases=selected,evaluation_cases=split['val'],support_selection='One crop from each saved 5% training subject; alternating L/R in sorted subject order; independent of model seed',
            data=str(REMOTE/'hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'),
            dataset_sha256=digest(ROOT/'datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'),
            support_subjects=provenance['train_subjects'])
    else:
        manifest=read(ROOT/'reports/geofwi3d_replication_20261011/cluster/folds/fold_0/manifest.json')
        p.update(train_cases=manifest['splits']['training_subsets']['5pct'][:5],evaluation_cases=manifest['splits']['validation'],
            data=str(REMOTE/'Dataset203_GeoFWI3D'),manifest=manifest,
            support_selection='First five volumes in the previously frozen 19-volume training subset')
    tag=dataset+'_foundation_fold0_seed0'
    write(stage/'protocols'/tag/'protocol.json',p)
    return tag


def freeze_source(stage):
    paths=['scripts/cross_dataset_campaign','scripts/adni_baselines','scripts/adni_constraints',
        'scripts/capped_depth_delay','scripts/depth_supervision/geometry.py','scripts/campaign90.py',
        'scripts/optimization90.py','scripts/compute_constraint_objectives.py','scripts/nnunet_multi_objective.py',
        'scripts/nnunet_band_objective.py','scripts/train_compute_matched_baselines.py',
        'scripts/profile_intermediate_baselines.py','scripts/profile_unet_constraint_budget.py',
        'scripts/geofwi3d_data.py','scripts/train_medsam3_few_shot.py','scripts/train_medsam3_constraint_few_shot.py',
        'scripts/train_medsam3_multiclass.py','scripts/medsam3_supervised_constraints.py','scripts/medsam3_calibration_cache.py',
        'evaluation/medsam3_zero_shot.py','baselines/simple_segmentation.py','baselines/swin_unetr/swin_unetr.py',
        'baselines/nnUNet/nnunetv2','thesis/new_constraints','reports/compute_budget_5pct_fold0_seed0_20261003/profile_reference.py',
        'reports/compute_budget_5pct_fold0_seed0_20261003/profile_matched_models.py']
    source=stage/'source'
    for name in paths:
        src,dst=ROOT/name,source/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir():
            shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pt','*.nii.gz'))
        else:
            shutil.copyfile(src,dst)
    write(source/'source_manifest.json',{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file() and p.name!='source_manifest.json'})


def full_plan(stage):
    tasks=[]
    for dataset in ('MSD','ADNI','GeoFWI3D'):
        for fold in (0,1,2):
            for seed in (0,1,2):
                for model in MODELS:
                    for arm in ARMS:
                        tasks.append(dict(kind='conventional',dataset=dataset,model=model,fold=fold,seed=seed,arm=arm))
    existing={'dice','dice_bands','dice_bands_edge'}
    for dataset,folds,seeds in [('MSD',(0,1,3),(17,83,191)),('ADNI',(0,1,2),(0,1,2)),('GeoFWI3D',(0,1,2),(0,1,2))]:
        for fold in folds:
            for seed in seeds:
                for arm in ARMS:
                    if dataset=='MSD' and arm in existing:continue
                    tasks.append(dict(kind='foundation',dataset=dataset,model='SAM3' if dataset=='GeoFWI3D' else 'MedSAM3',fold=fold,seed=seed,arm=arm))
    if len(tasks)!=1863: raise ValueError('Full matrix incomplete')
    write(stage/'full_study_plan.json',dict(status='prepared_not_authorized_for_submission',submission_enabled=False,
        augmentation='none',conventional_baseline='Dice-only; nnUNet deep supervision retained',
        budget_flops_by_dataset={'MSD':CAP,'ADNI':CAP,'GeoFWI3D':SEISMIC_CAP},
        required_decisions=['MSD nnUNet calibration target after exploration','Foundation pilot viability and constraint benefit','Seismic cap adequacy'],
        required_preparation=['Binary 64-cube nnUNet Dice-only constraint profiles','96-cube seismic constraint profiles',
            'Materialize all saved folds and per-seed foundation protocols after exploration decisions'],
        conventional_runs=1512,foundation_new_runs=351,total_new_runs_before_pilot_reuse=1863,tasks=tasks,
        reuse_policy='Reuse only exactly matching complete archived pilot cells; no augmented or Dice+CE nnUNet reuse'))


def main():
    stage=ROOT/'experiments'/NAME
    stage.mkdir(parents=True,exist_ok=True)
    if (stage/'submission.json').exists():raise ValueError('Never alter a submitted stage')
    tasks=[]
    for target in (.1,.4,.6):
        tag=conventional_protocol(stage,'MSD','nnunet',target)
        for arm in (ARMS if target==.1 else ARMS[1:]):
            tasks.append(dict(id=f'MSD_nnunet_{arm}_target{int(target*100)}_f0s0',kind='conventional',dataset='MSD',model='nnunet',fold=0,seed=0,arm=arm,protocol=tag,target=target))
    for dataset in ('ADNI','GeoFWI3D'):
        tag=foundation_protocol(stage,dataset)
        for arm in ('dice','dice_bands','dice_bands_edge'):
            tasks.append(dict(id=f'{dataset}_foundation_{arm}_f0s0',kind='foundation',dataset=dataset,fold=0,seed=0,arm=arm,protocol=tag))
    for identity in MODELS:
        tag=conventional_protocol(stage,'GeoFWI3D',identity)
        tasks.append(dict(id=f'GeoFWI3D_{identity}_dice_f0s0',kind='conventional',dataset='GeoFWI3D',model=identity,fold=0,seed=0,arm='dice',protocol=tag,target=.1))
    if len(tasks)!=50:raise ValueError('Expected 50 exploration cells')
    write(stage/'plan.json',dict(name=NAME,tasks=tasks,num_runs=50,augmentation='none',
        budget_flops_by_dataset={'MSD':CAP,'ADNI':CAP,'GeoFWI3D':SEISMIC_CAP},
        conventional_python=str(REMOTE/'.conda/envs/hippocampus/bin/python'),
        foundation_python=str(REMOTE/'medsam3_runtime_20260928/venv/bin/python'),
        storage=dict(minimum_free_gib=8,estimated_peak_logical_gib=3,mirrored_peak_gib=6,reserve_gib=2,
            archive=str(REPORT),policy='Archive and verify each run outside BeeGFS before deleting its generated bulky artifacts'),
        base_staging='/tmp/hippo-cross-dataset-3160552/sam3.pt',base_sha256=read(stage/'protocols/ADNI_foundation_fold0_seed0/protocol.json')['base_sha256'],
        max_job_hours=24,graceful_stop_seconds=23*3600+40*60))
    write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=False))
    write(stage/'source_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_worktree=True,source_of_truth='Frozen source files and their SHA256 manifest include uncommitted implementation'))
    full_plan(stage);freeze_source(stage)
    shutil.copyfile(ROOT/'scripts/cross_dataset_campaign/run.sbatch',stage/'run.sbatch')
    write(stage/'manifest.json',{str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name not in ('manifest.json','control.json')})
    REPORT.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(stage=str(stage),exploration_runs=len(tasks),full_study_runs=1863)))


if __name__=='__main__':main()
