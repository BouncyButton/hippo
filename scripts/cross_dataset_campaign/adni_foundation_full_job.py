"""Frozen 126-run, two-sample MedSAM3 ADNI constraint replication."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.adni_foundation_folds_job import select_support
from scripts.cross_dataset_campaign.common import REMOTE, ARMS, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT

NAME='adni_medsam3_2sample_3fold_3seed_14arms_20261018'
MODULE='scripts.cross_dataset_campaign.adni_foundation_full_job'
PARENT='adni_medsam3_2sample_folds13_seed0_20261018'
PARENT_SHA='7052e09eacabdfed80fa806eb7ef0908659edd7851a0bd506a12bcac62a58088'
FOLDS=(0,1,3)
SEEDS=(0,1,2)
ADDITIONS=('adni_foundation_full_job.py','adni_foundation_full_pool.py','adni_foundation_full_runtime.py',
           'adni_foundation_full_worker.py','manage_adni_foundation_full.py','test_adni_foundation_full.py')


def tasks():
    return [dict(id=f'ADNI_MedSAM3_2sample_{arm}_f{fold}s{seed}',kind='foundation',dataset='ADNI',
        model='MedSAM3',fold=fold,seed=seed,arm=arm,target=.1,protocol=f'ADNI_f{fold}_s{seed}')
        for arm in ARMS for fold in FOLDS for seed in SEEDS]


def protocol_for(original,selection,seed):
    p=deepcopy(original)
    p.update(name=NAME,fold=selection['fold'],seed=seed,train_cases=selection['train_cases'],
        support_subjects=selection['support_subjects'],evaluation_cases=selection['evaluation_cases'],
        requested_arms=list(ARMS),support_selection='Same saved-fold RandomState(0) five-subject subset rule as the two-sample pilot; first two sorted subjects and alternating L/R; fixed across model seeds')
    return p


def prepare():
    parent=ROOT/'reports'/PARENT/'frozen';stage=ROOT/'experiments'/NAME
    if stage.exists():verify(stage);return stage
    if digest(parent/'manifest.json')!=PARENT_SHA:raise ValueError('Wrong completed pilot source')
    for name,sha in read(parent/'manifest.json').items():
        if digest(parent/name)!=sha:raise ValueError('Parent source changed: '+name)
    old=read(parent/'parent_protocol.json');audit=read(parent/'split_audit.json')
    stage.mkdir(parents=True)
    shutil.copytree(parent/'source',stage/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ADDITIONS:shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/name,stage/'source/scripts/cross_dataset_campaign'/name)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    for name in ('parent_protocol.json','split_audit.json','saved_folds.json','split_provenance.json','source_version.json'):
        shutil.copy2(parent/name,stage/name)
    for selection in audit['selections']:
        for seed in SEEDS:write(stage/'protocols'/f"ADNI_f{selection['fold']}_s{seed}"/'protocol.json',protocol_for(old,selection,seed))
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=126,folds=list(FOLDS),seeds=list(SEEDS),
        requested_arms=list(ARMS),archive_resume_module=MODULE,reference_campaign=PARENT,reference_manifest_sha256=PARENT_SHA,
        gpu_preflight_protocols=[f'ADNI_f{f}_s0' for f in FOLDS],
        authorization='User requests a full three-fold three-seed MedSAM3 ADNI fine-tuning sweep with the established constraint list, continuing the two-sample study.',
        exploration_followup='126 fresh runs only; 14 established conditions across folds 0/1/3 and seeds 0/1/2',
        comparison_policy='Nine fresh matched baselines first. Same two training crops within each fold across seeds and arms. Original native loss, pretrained initialization, optimizer, learning-rate schedule, precision, sampling and 200-update budget; no augmentation.',
        geometry_policy='Existing ADNI source-voxel geometry: SDF clipped at 8 source voxels; normalized PCA-plane depth CE using source X as reference. No anatomical-axis or millimeter claim without affine metadata.',
        parallelism=dict(workers=2,gpus_per_worker=1,cpus_per_worker=4,gpu_type='4g.40gb',allocation_cpu_count=8,allocation_memory_gib=80,account_max_gpus=2,
            policy='Two isolated experiments; all nine baselines archived/cleaned before auxiliary conditions; per-run archive gate'))
    plan['storage'].update(archive=str(ROOT/'reports'/NAME),minimum_free_gib=12,
        estimated_peak_logical_gib=5,mirrored_peak_gib=10,reserve_gib=2)
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=True))
    batch=(parent/'run.sbatch').read_text().replace('hippo-adni-medsam3-f13','hippo-adni-medsam3-full')
    batch=batch.replace('scripts.cross_dataset_campaign.adni_foundation_folds_job run',MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json','control.json')})
    verify(stage)
    write(ROOT/'reports'/NAME/'request_summary.json',dict(name=NAME,folds=list(FOLDS),seeds=list(SEEDS),conditions=list(ARMS),
        runs=126,fresh_baselines=9,training_crops=2,training_subjects=2,validation_crops=54,validation_subjects=27,
        updates=200,augmentation='none',workers=2,calibration_target=.1,calibration_p95_cap=.5,
        support_note=audit['note'],geometry_policy=plan['geometry_policy'],scope='All results fresh; historical pilots are references only'))
    return stage


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen campaign changed: '+name)
    if digest(stage/'parent_manifest.json')!=PARENT_SHA:raise ValueError('Parent provenance differs')
    parent=read(stage/'parent_manifest.json')
    for name,sha in parent.items():
        if name.startswith('source/') and digest(stage/name)!=sha:raise ValueError('Original training source changed: '+name)
    if digest(stage/'parent_protocol.json')!=parent['parent_protocol.json']:raise ValueError('Original recipe changed')
    if digest(stage/'saved_folds.json')!=parent['saved_folds.json'] or digest(stage/'split_audit.json')!=parent['split_audit.json']:
        raise ValueError('Saved training/validation cohorts changed')
    old=read(stage/'parent_protocol.json');audit=read(stage/'split_audit.json')['selections']
    if [s['fold'] for s in audit]!=list(FOLDS):raise ValueError('Wrong folds')
    for selection in audit:
        for seed in SEEDS:
            p=read(stage/'protocols'/f"ADNI_f{selection['fold']}_s{seed}"/'protocol.json')
            if p!=protocol_for(old,selection,seed):raise ValueError('Fine-tuning recipe differs')
    plan=read(stage/'plan.json')
    if (plan['name']!=NAME or plan['tasks']!=tasks() or plan['num_runs']!=126 or plan['archive_resume_module']!=MODULE
            or plan['archive_wait_timeout_seconds']!=300 or plan['max_job_hours']!=24
            or plan['parallelism']['workers']!=2 or plan['folds']!=list(FOLDS) or plan['seeds']!=list(SEEDS)):
        raise ValueError('Scope/resource contract changed')
    return plan


def submit(stage,continuation=None):
    verify(stage)
    own={p.stem for p in (stage/'submissions').glob('*.json')}
    active=[r for r in controller.queue() if r[0] in own]
    if continuation is None and active:return read(stage/'submissions'/f'{active[0][0]}.json')
    previous=controller.verify;controller.verify=verify
    try:return controller.submit(stage,continuation)
    finally:controller.verify=previous


def run(stage):
    from scripts.cross_dataset_campaign import adni_foundation_full_pool as pool
    plan=verify(stage);job=os.environ['SLURM_JOB_ID']
    def successor():
        path=stage/'successor.json'
        if path.exists():
            r=read(path)
            if (r['dependency']==job and r['manifest_sha256']==digest(stage/'manifest.json')
                    and any(x[0]==r['job_id'] and x[1]=='PENDING' for x in controller.queue())):return r
        return submit(stage,job)
    pool.run(stage,plan,successor,lambda:controller.cancel_successor(stage,job))


def preflight(stage):
    plan=verify(stage);old=read(stage/'parent_protocol.json')
    for path,sha in ((plan['base_staging'],plan['base_sha256']),(old['data'],old['dataset_sha256']),
                     (old['lora_weights'],old['lora_sha256'])):
        if digest(path)!=sha:raise ValueError('Data/weights differ: '+path)
    for name,sha in read(stage/'source_version.json')['upstream_provenance']['python_source_sha256'].items():
        if digest(Path(old['upstream'])/name)!=sha:raise ValueError('Upstream changed: '+name)
    subprocess.run([plan['foundation_python'],'-m','scripts.cross_dataset_campaign.test_adni_foundation_full','--stage',str(stage)],check=True)
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    qos=subprocess.check_output(['sacctmgr','-n','-P','show','qos','stud','format=Name,MaxTRESPerUser%80,MaxTRESPJ%80,MaxJobsPU,MaxSubmitPU'],text=True)
    limits=dict(x.split('=') for x in next(x.split('|')[1] for x in qos.splitlines() if x.startswith('stud|')).split(','))
    if int(limits['gres/gpu'])!=2 or int(limits['cpu'])<8:raise ValueError('Reassess account limits')
    scheduler=subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)],capture_output=True,text=True,check=True)
    usage=controller.quota()
    if usage['free_gib']<plan['storage']['minimum_free_gib']:raise ValueError('Insufficient headroom')
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),quota=usage,qos=qos,
        scheduler_test=scheduler.stdout+scheduler.stderr,checked_at=time.time(),gpu_gate='All-loss finite backward/calibration must pass on both distinct supports before any optimizer updates',
        checks=['Original trainer byte-identical','126 exact cells with nine fresh baselines','Subject split, geometry and gradient tests',
                'Two-worker pool, archive gate, resume, pause and failure tests','Data/weights/upstream hashes','Scheduler and quota']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify','preflight','submit','run','status'])
    parser.add_argument('--stage',type=Path);args=parser.parse_args();stage=args.stage or ROOT/'experiments'/NAME
    if args.command=='prepare':print(prepare())
    elif args.command=='verify':print(verify(stage)['name'])
    elif args.command=='preflight':preflight(stage)
    elif args.command=='submit':print(submit(stage))
    elif args.command=='run':run(stage)
    else:print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')


if __name__=='__main__':main()
