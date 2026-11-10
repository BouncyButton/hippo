"""Freeze and gate the ADNI-only mixed-loss sweep after completed MSD archives."""
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign import msd_mixed_losses_job as msd
from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

NAME = 'adni_mixed_losses_nnunet60_others10_20261014'
MODULE = 'scripts.cross_dataset_campaign.adni_mixed_losses_job'
BASE_SUBMIT = controller.submit
BASE_COMMAND = controller.command_for


def task_id(model, arm):
    native = model=='nnunet'
    return f'ADNI_{model}_{"dicece" if native else "diceonly"}_{arm}_target{60 if native else 10}_f0s0'


def verify_split(stage):
    provenance = read(stage/'split_provenance.json')
    if (provenance['requested_fraction']!=.05 or provenance['subset_seed']!=0
            or len(provenance['train_subjects'])!=5 or len(provenance['validation_subjects'])!=27
            or len(provenance['full_train_subjects'])!=108
            or set(provenance['full_train_subjects'])&set(provenance['validation_subjects'])):
        raise ValueError('Expected the saved subject-level fold0 5% ADNI split')
    splits = []
    for model in MODELS:
        split = read(stage/'protocols'/model/'splits_10patients.json')[0]
        if (len(set(split['train']))!=10 or len(set(split['val']))!=54
                or set(split['train'])&set(split['val'])):
            raise ValueError('Expected ten training and 54 validation crops')
        for key,subject_key in (('train','train_subjects'),('val','validation_subjects')):
            subjects = [provenance['case_to_subject'][name] for name in split[key]]
            if set(subjects)!=set(provenance[subject_key]) or any(subjects.count(s)!=2 for s in set(subjects)):
                raise ValueError('Subject split or bilateral crops differ')
        splits.append(split)
    if any(split!=splits[0] for split in splits):raise ValueError('Model splits differ')
    return provenance


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():raise ValueError('Never overwrite an existing campaign')
    predecessor = ROOT/'reports'/msd.NAME/'frozen'
    msd.verify(predecessor)
    originals = ROOT/'experiments/adni_constraints_noaug_5pct_fold0_seed0_20261013/protocols'
    source_hashes = read(originals/'nnunet/protocol.json')['training_source_sha256']
    for name,sha in source_hashes.items():
        if digest(ROOT/name)!=sha:raise ValueError('Original ADNI scientific source changed: '+name)
    stage.mkdir(parents=True)
    shutil.copyfile(ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013/split_provenance.json',stage/'split_provenance.json')
    tasks = []
    for model in MODELS:
        native = model=='nnunet';target,ceiling = (.6,.75) if native else (.1,.5)
        origin = ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'ADNI_{model}_f0_target10'
        folder = stage/'protocols'/model;folder.mkdir(parents=True)
        p = deepcopy(read(origin/'protocol.json'))
        for name in ('splits_10patients.json','nnunet_architecture.json'):
            shutil.copyfile(origin/name,folder/name)
        profile_origin = (ROOT/'reports/adni_seismic_ce60_followon_20261014/profiles/ADNI_nnunet.json'
            if native else origin/'profile.json')
        profile = read(profile_origin)
        if native:
            if (profile['status']!='profiled' or profile['loss_recipe']!='dice_ce'
                    or profile['dataset']!='ADNI' or profile['model']!='nnunet'):
                raise ValueError('Expected the measured native ADNI Dice+CE profile')
            for key,name in (('objective_sha256','binary_ce_constraints.py'),('shared_objective_sha256','msd_ce_constraints.py')):
                if profile[key]!=digest(ROOT/'scripts/cross_dataset_campaign'/name):raise ValueError('CE profile graph changed')
            shutil.copyfile(ROOT/profile['reference_profile'],folder/'reference_profile.json')
            if digest(folder/'reference_profile.json')!=profile['reference_profile_sha256']:raise ValueError('CE reference profile changed')
        else:
            if profile.get('loss_recipe','dice_only')!='dice_only':raise ValueError('Expected established Dice profile')
            for name,sha in profile['adapter_sources_sha256'].items():
                if digest(ROOT/name)!=sha:raise ValueError('Binary profile adapter changed')
        shutil.copyfile(profile_origin,folder/'profile.json')
        alias = p['trainer_alias'];old = p['models'][alias]['objectives']
        p['models'] = {alias:dict(parameters=profile['parameters'],objectives={
            arm:schedule(profile['costs'],arm,old[arm],10,CAP) for arm in ARMS})}
        p.update(name=NAME,dataset='ADNI',augmentation='none',folds=[0],seeds=[0],weight_ce=int(native),
            requested_fraction=.05,train_count=10,validation_count=54,requested_arm_order=list(ARMS),
            calibration_base='Dice+CE' if native else 'Dice',calibration_target_ratio=target,calibration_p95_cap=ceiling,
            training_loss='Native foreground Dice+CE with original deep supervision' if native else 'Established MONAI Dice including background',
            budget_flops=CAP,budget_gflops=CAP/1e9,validation_target_flops=[CAP*i//20 for i in range(1,21)],
            checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
            baseline_run_id=task_id(model,'dice'),profile_sha256=digest(folder/'profile.json'),
            split_provenance_sha256=digest(stage/'split_provenance.json'),training_source_sha256={},
            runtime_acceleration=dict(enabled=False,reason='MSD-specific acceleration is not assumed validated for ADNI'))
        write(folder/'protocol.json',p)
    for arm in ARMS:
        for model in MODELS:
            native = model=='nnunet'
            tasks.append(dict(id=task_id(model,arm),kind='conventional',dataset='ADNI',model=model,
                fold=0,seed=0,arm=arm,target=.6 if native else .1,baseline_loss='Dice+CE' if native else 'Dice',protocol=model))
    plan = deepcopy(read(predecessor/'plan.json'))
    plan.update(name=NAME,num_runs=56,tasks=tasks,budget_flops_by_dataset={'ADNI':CAP},
        archive_resume_module=MODULE,archive_dependency_stages=[],
        predecessor=dict(name=msd.NAME,stage=str(REMOTE/msd.NAME),manifest_sha256=digest(predecessor/'manifest.json')),
        authorization='User requests the same ADNI analysis after the current MSD campaign: all14 conditions for four models, no augmentation, saved5% split, fold0 seed0; native nnUNet Dice+CE target60%, other models Dice-only target10%.',
        exploration_followup='This new ADNI-only campaign is authorized; the old ADNI/seismic/MedSAM3 and1863-run campaigns remain held',
        comparison_policy='Fresh matched baseline per model; report within-model gains, Dice, FN/FP, curves and calibration. ADNI has one foreground class, so A/P swaps are not applicable.')
    plan['storage'].update(archive=str(ROOT/'reports'/NAME),estimated_peak_logical_gib=5,
        mirrored_peak_gib=10,reserve_gib=2,minimum_free_gib=12)
    write(stage/'plan.json',plan)
    write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=False))
    write(stage/'source_version.json',dict(original_adni_source_hashes=source_hashes,
        predecessor_manifest_sha256=plan['predecessor']['manifest_sha256'],
        profile_policy='Existing measured binary profiles; native Dice+CE for nnUNet and Dice-only for the others. Numerical equivalence checked before submission.'))
    freeze_source(stage)
    for name,sha in source_hashes.items():
        target = stage/'source'/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        if digest(target)!=sha:raise ValueError('Frozen source differs from profile lineage: '+name)
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f)
        for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json'})
    batch = (predecessor/'run.sbatch').read_text().replace('hippo-msd-mixed-60-10','hippo-adni-mixed-60-10').replace(msd.MODULE+' run',MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    verify(stage)
    print(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen ADNI source changed: '+name)
    plan = read(stage/'plan.json')
    if (plan['name']!=NAME or plan['num_runs']!=56 or plan['augmentation']!='none'
            or [(t['model'],t['arm']) for t in plan['tasks']]!=[(m,a) for a in ARMS for m in MODELS]
            or plan['budget_flops_by_dataset']!={'ADNI':CAP}):
        raise ValueError('Expected exactly four ADNI models by14 conditions')
    for name,sha in read(stage/'source_version.json')['original_adni_source_hashes'].items():
        if digest(stage/'source'/name)!=sha:raise ValueError('ADNI profile source lineage differs')
    provenance = verify_split(stage)
    for model in MODELS:
        folder = stage/'protocols'/model;p = read(folder/'protocol.json')
        native = model=='nnunet';target,ceiling = (.6,.75) if native else (.1,.5)
        if (p['dataset']!='ADNI' or p['augmentation']!='none' or p['weight_ce']!=int(native)
                or p['calibration_base']!=('Dice+CE' if native else 'Dice')
                or p['calibration_target_ratio']!=target or p['calibration_p95_cap']!=ceiling
                or p['budget_flops']!=CAP or p['folds']!=[0] or p['seeds']!=[0]
                or p['pkl_sha256']!=provenance['dataset_sha256'] or p['requested_fraction']!=.05
                or p['runtime_acceleration']['enabled'] or p['baseline_run_id']!=task_id(model,'dice')):
            raise ValueError('Unauthorized ADNI scientific settings')
        for name,key in (('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')):
            if digest(folder/name)!=p[key]:raise ValueError('Protocol artifact changed: '+name)
        if p['split_provenance_sha256']!=digest(stage/'split_provenance.json'):raise ValueError('Subject split provenance differs')
        profile = read(folder/'profile.json')
        if native:
            if (profile['objective_sha256']!=digest(stage/'source/scripts/cross_dataset_campaign/binary_ce_constraints.py')
                    or profile['shared_objective_sha256']!=digest(stage/'source/scripts/cross_dataset_campaign/msd_ce_constraints.py')
                    or profile['reference_profile_sha256']!=digest(folder/'reference_profile.json')):
                raise ValueError('Native profile lineage differs')
        else:
            for name,sha in profile['adapter_sources_sha256'].items():
                if digest(stage/'source'/name)!=sha:raise ValueError('Dice profile adapter changed')
        for task in (t for t in plan['tasks'] if t['model']==model):
            if ((task['dataset'],task['fold'],task['seed'],task['target'],task['id'])!=('ADNI',0,0,target,task_id(model,task['arm']))
                    or task['baseline_loss']!=p['calibration_base']):raise ValueError('Unauthorized ADNI task')
            s = p['models'][p['trainer_alias']]['objectives'][task['arm']]
            if s!=schedule(profile['costs'],task['arm'],s,10,CAP):raise ValueError('FLOP schedule differs')
            if s['max_steps']<=s['sdf_prefix_updates']:raise ValueError('No updates remain for constraints')
    return plan


def predecessor_complete(plan):
    dependency = plan['predecessor'];stage = Path(dependency['stage'])
    if dependency['name']!=msd.NAME or digest(stage/'manifest.json')!=dependency['manifest_sha256']:
        raise ValueError('MSD prerequisite differs')
    previous = msd.verify(stage)
    status = read(stage/'status.json') if (stage/'status.json').exists() else {}
    if read(stage/'control.json')['pause'] or status.get('status')!='complete' or status.get('completed')!=56:return False
    for task in previous['tasks']:
        output = stage/'runs'/task['id']
        ack = read(output/'archive_acknowledgment.json')
        if (read(output/'completion.json')['status']!='complete' or read(output/'cleanup.json')['status']!='complete'
                or ack['status']!='verified' or not ack['source_verified']
                or ack['manifest_sha256']!=digest(output/'archive_manifest.json')
                or ack['source_manifest_sha256']!=dependency['manifest_sha256']):
            raise ValueError('MSD archive/cleanup prerequisite is incomplete')
    return True


def submit(stage, continuation=None):
    plan = verify(stage)
    if not predecessor_complete(plan):raise RuntimeError('Wait for all56 MSD runs, verified archives and cleanup')
    with (stage/'start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if continuation is None:
            own = {p.stem for p in (stage/'submissions').glob('*.json')}
            active = [row for row in controller.queue() if row[0] in own]
            if active:return read(stage/'submissions'/f'{active[0][0]}.json')
            if (stage/'status.json').exists() and read(stage/'status.json').get('status')=='complete':
                raise RuntimeError('ADNI campaign already complete')
        previous = controller.verify;controller.verify = verify
        try:return BASE_SUBMIT(stage,continuation)
        finally:controller.verify = previous


def command_for(stage, plan, task, base_path):
    command = BASE_COMMAND(stage,plan,task,base_path)
    command[2] = 'scripts.cross_dataset_campaign.adni_mixed_losses'
    return command


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare();return
    controller.verify=verify;controller.command_for=command_for;controller.submit=submit
    if len(sys.argv)>1 and sys.argv[1]=='preflight':
        import argparse
        from scripts.cross_dataset_campaign.preflight_adni_mixed_losses import check
        parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('--stage',type=Path,required=True)
        stage=parser.parse_args().stage
        checks=check(stage,verify(stage))
        write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=checks))
        print('ADNI mixed-loss56-run preflight passed',flush=True);return
    if len(sys.argv)>1 and sys.argv[1]=='run':
        stage=Path(sys.argv[sys.argv.index('--stage')+1])
        if not predecessor_complete(verify(stage)):raise RuntimeError('MSD prerequisite not complete')
    controller.main()


if __name__=='__main__':main()
