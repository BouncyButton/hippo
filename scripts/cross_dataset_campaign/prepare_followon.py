"""Freeze the user-authorized ADNI/seismic/MedSAM3 follow-on: 126 conditions."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, SEISMIC_CAP, REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, schedule, freeze_source

NAME='adni_seismic_ce60_followon_20261014'
MODULE='scripts.cross_dataset_campaign.followon_job'
PREDECESSOR='msd_four_models_ce_target60_20261014'


def prepare():
    stage=ROOT/'experiments'/NAME
    if (stage/'submission.json').exists():raise ValueError('Do not change a submitted follow-on')
    lookup={}
    for dataset in ('ADNI','GeoFWI3D'):
        for model in MODELS:
            source=ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'{dataset}_{model}_f0_target10'
            p=deepcopy(read(source/'protocol.json'))
            tag=f'{dataset}_{model}';folder=stage/'protocols'/tag;folder.mkdir(parents=True,exist_ok=True)
            for name in ('splits_10patients.json','nnunet_architecture.json'):
                shutil.copyfile(source/name,folder/name)
            profile_path=ROOT/'reports'/NAME/'profiles'/f'{tag}.json';profile=read(profile_path)
            if profile['status']!='profiled' or profile['loss_recipe']!='dice_ce' or profile['dataset']!=dataset:
                raise ValueError('Verified Dice+CE profile required')
            if (profile['objective_sha256']!=digest(ROOT/'scripts/cross_dataset_campaign/binary_ce_constraints.py')
                    or profile['shared_objective_sha256']!=digest(ROOT/'scripts/cross_dataset_campaign/msd_ce_constraints.py')):
                raise ValueError('Profile loss graph differs')
            shutil.copyfile(profile_path,folder/'profile.json')
            shutil.copyfile(ROOT/profile['reference_profile'],folder/'reference_profile.json')
            if digest(folder/'reference_profile.json')!=profile['reference_profile_sha256']:
                raise ValueError('Reference full-model profile changed')
            cap=CAP if dataset=='ADNI' else SEISMIC_CAP
            count=10 if dataset=='ADNI' else 19
            alias=p['trainer_alias'];old=p['models'][alias]['objectives']
            p.update(name=NAME,dataset=dataset,augmentation='none',weight_ce=1,folds=[0],seeds=[0],
                training_loss='Established per-model Dice+CE, coefficients one, plus original binary auxiliaries',
                calibration_base='Dice+CE',calibration_target_ratio=.6,calibration_p95_cap=.75,
                dice_include_background=model!='nnunet',requested_arm_order=list(ARMS),constraints=True,
                budget_flops=cap,budget_gflops=cap/1e9,validation_target_flops=[cap*i//20 for i in range(1,21)],
                checkpoint_selection='Final budget checkpoint is primary; best validation is secondary',
                baseline_run_id=f'{dataset}_{model}_ce_dice_target60_f0s0',training_source_sha256={},
                profile_sha256=digest(folder/'profile.json'),profile_status='measured')
            if dataset=='GeoFWI3D':
                shutil.copyfile(source/'dataset.json',folder/'dataset.json')
                p.update(data=str(REMOTE/NAME/'protocols'/tag/'dataset.json'),pkl_sha256=digest(folder/'dataset.json'),
                    budget_policy='Keep the previously agreed fixed seismic cap; Dice+CE updates determined by measured cost')
            p['models']={alias:dict(parameters=profile['parameters'],objectives={
                arm:schedule(profile['costs'],arm,old[arm],count,cap) for arm in ARMS})}
            write(folder/'protocol.json',p);lookup[(dataset,model)]=tag
    foundation=deepcopy(read(ROOT/'experiments/cross_dataset_exploration_20261014/protocols/ADNI_foundation_fold0_seed0/protocol.json'))
    foundation.update(name=NAME,fold=0,seed=0,augmentation='none',requested_arms=list(ARMS),
        gradient_ratio_target=.6,gradient_ratio_p95_cap=.75,
        baseline_loss='Unchanged established MedSAM3 native fine-tuning loss',
        user_choice='Preserve established MSD native baseline and 200-update recipe; constraints at 60%',
        primary_checkpoint='final_update',updates=200)
    if len(foundation['train_cases'])!=5 or len(foundation['evaluation_cases'])!=54:
        raise ValueError('Expected five support crops and 54 validation crops')
    write(stage/'protocols/ADNI_MedSAM3/protocol.json',foundation)
    tasks=[]
    for arm in ARMS:
        for dataset in ('ADNI','GeoFWI3D'):
            for model in MODELS:
                tasks.append(dict(id=f'{dataset}_{model}_ce_{arm}_target60_f0s0',kind='conventional',
                    dataset=dataset,model=model,fold=0,seed=0,arm=arm,target=.6,protocol=lookup[(dataset,model)]))
        tasks.append(dict(id=f'ADNI_MedSAM3_{arm}_target60_f0s0',kind='foundation',dataset='ADNI',
            model='MedSAM3',fold=0,seed=0,arm=arm,target=.6,protocol='ADNI_MedSAM3'))
    plan=deepcopy(read(ROOT/'experiments'/PREDECESSOR/'plan.json'))
    old_stage=ROOT/'experiments'/PREDECESSOR
    plan.update(name=NAME,num_runs=126,tasks=tasks,authorization='User explicitly requests one queued follow-on after the exact MSD sweep: ADNI and seismic four-model/all-14-condition Dice+CE target60 sweeps, plus all 14 ADNI MedSAM3 native-baseline target60 conditions, five crops and 200 updates.',
        budget_flops_by_dataset={'ADNI':CAP,'GeoFWI3D':SEISMIC_CAP},
        archive_resume_module=MODULE,archive_dependency_stages=[str(REMOTE/PREDECESSOR)],
        predecessor=dict(name=PREDECESSOR,stage=str(REMOTE/PREDECESSOR),
            manifest_sha256=digest(old_stage/'manifest.json'),module='scripts.cross_dataset_campaign.msd_ce_constraints_job'),
        exploration_followup='The original full replicated sweep remains held; this 126-run follow-on is explicitly authorized',
        comparison_policy='Compare each constraint to its matched unaugmented baseline within model/dataset; fixed final checkpoint primary')
    plan['storage'].update(archive=str(ROOT/'reports'/NAME),estimated_peak_logical_gib=5,
        mirrored_peak_gib=10,reserve_gib=2,minimum_free_gib=12)
    write(stage/'plan.json',plan)
    write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=False))
    write(stage/'source_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_worktree=True,source_of_truth='Frozen source and SHA256 manifest',request=plan['authorization']))
    freeze_source(stage)
    batch=(ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text().replace(
        'hippo-exploration-noaug','hippo-adni-seismic-ce60').replace(
        'scripts.cross_dataset_campaign.controller run',MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    print(stage)


if __name__=='__main__':prepare()
