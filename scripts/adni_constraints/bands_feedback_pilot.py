"""Two bands-strength tests with frozen ADNI data, training code and MSD timing."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

NAME = 'adni_nnunet_bands_feedback_20261013'
PARENT = 'adni_constraints_noaug_5pct_fold0_seed0_20261013'
REMOTE = Path('/mnt/beegfsstudents/home/3160552')
CAP = 31976450667090
VARIANTS = {'bands_target20': .2, 'bands_target40': .4}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def quota():
    value = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
    m = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',value)
    if not m:
        raise ValueError('Cannot verify quota')
    return float(m[2])-float(m[1]), value


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:
            raise ValueError('Pilot file changed: '+name)
    plan=read(stage/'plan.json')
    parent=Path(plan['parent_stage'])
    assert digest(parent/'manifest.json')==plan['parent_manifest_sha256']
    for name,sha in read(parent/'manifest.json').items():
        assert digest(parent/name)==sha, name
    assert digest(Path(plan['dataset']))==plan['dataset_sha256']
    return plan


def prepare(stage):
    root=Path.cwd()
    parent=root/'experiments'/PARENT
    if (stage/'manifest.json').exists():
        raise FileExistsError('Pilot is already frozen')
    stage.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,stage/'pilot.py')
    prior=read(parent/'protocols/nnunet/protocol.json')
    records=root/'reports'/PARENT/'comparison_20261013/results/nnunet'
    calibration=read(records/'dice_bands/fold0/seed0/run/calibration.json')
    references={}
    for objective in ['dice','dice_bands']:
        run=records/objective/'fold0/seed0/run'
        references[objective]=dict(final_dice=read(run/'validation_latest.json')['summary']['hippocampus_hard_dice'],
            completion_sha256=digest(run/'completion.json'),validation_sha256=digest(run/'validation_latest.json'))
    original_schedule=prior['models']['nnunet']['objectives']['dice_bands']
    schedules=[]
    for variant,target in VARIANTS.items():
        directory=stage/'conditions'/variant/'protocols/nnunet'
        directory.mkdir(parents=True)
        for filename in ['splits_10patients.json','nnunet_architecture.json','profile.json']:
            shutil.copyfile(parent/'protocols/nnunet'/filename,directory/filename)
        protocol={**prior,'name':NAME+'_'+variant,'requested_arm_order':['dice_bands'],
            'models':{'nnunet':{**prior['models']['nnunet'],'objectives':{'dice_bands':original_schedule}}},
            'band_target_ratio':target,'band_p95_cap':.5,'pilot_controller_sha256':digest(stage/'pilot.py'),
            'calibration_policy':f'At update 9, ten unaugmented training crops; bands median gradient target {target}, p95 ceiling 0.5. Original full-resolution native logit calibration. No validation calibration.',
            'pilot_change':'Only the bands calibration target changes from 0.1; preserve original 9-update prefix and 9-update ramp.'}
        write(directory/'protocol.json',protocol)
        rows=calibration['cases'];ratios=[r['bands_over_dice'] for r in rows]
        # Saved statistics are the same median/p95 reductions as calibration.
        stats=calibration['statistics']['bands']
        expected=min(target/stats['median_raw_ratio'],.5/stats['p95_raw_ratio'])
        schedules.append(dict(variant=variant,target=target,expected_band_weight=expected,
            model='nnunet',alias='nnunet',objective='dice_bands',updates=original_schedule['max_steps'],
            optimization_gflops=original_schedule['consumed_training_flops']/1e9))
    remote=REMOTE/NAME
    script=f'''#!/usr/bin/env bash
#SBATCH --job-name=hippo-adni-bands-feedback
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
export PYTHONPATH="{REMOTE/PARENT/'source'}:{REMOTE/PARENT/'source/scripts'}:{REMOTE/PARENT/'source/baselines/nnUNet'}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/hippo-adni-bands-mpl-${{SLURM_JOB_ID}}
/home/3160552/.conda/envs/hippocampus/bin/python "{remote/'pilot.py'}" --mode run --stage "{remote}"
'''
    (stage/'run.sbatch').write_text(script)
    plan=dict(name=NAME,parent_stage=str(REMOTE/PARENT),parent_manifest_sha256=digest(parent/'manifest.json'),
        dataset=str(REMOTE/'hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'),dataset_sha256=prior['pkl_sha256'],
        runs=schedules,num_runs=2,fold=0,seed=0,augmentation='none',budget_flops_per_run=CAP,
        runtime_versions=read(parent/'plan.json')['runtime_versions'],references=references,
        expected_anchor_model_sha256=calibration['anchor_model_sha256'],baseline_runs_reused=True,
        calibration_in_cap=False,diagnostic_compute='Separate CPU FP32 read-only probes; excluded from training cap.',
        rationale='Test whether stronger bands feedback improves the endpoint after the baseline catches up. Timing already matches corrected MSD.',
        retention='Node-local optimizer writes; retain verified best model and all metadata for new conditions.')
    write(stage/'plan.json',plan)
    manifest={str(p.relative_to(stage)):digest(p) for p in stage.rglob('*') if p.is_file() and 'diagnostic_checkpoints' not in p.parts}
    write(stage/'manifest.json',manifest)
    write(root/'reports'/NAME/'prepared_plan.json',plan)
    print(json.dumps(plan,indent=2))


def train(arguments):
    from scripts.adni_constraints import train as entry
    from scripts.capped_depth_delay import core
    from compute_constraint_objectives import calibrated_weight
    configure=entry.configure
    original_calibrate=core.calibrate
    def pilot_configure(protocol,arm):
        assert arm=='dice_bands' and protocol['augmentation']=='none'
        assert protocol['band_target_ratio'] in VARIANTS.values()
        assert digest(Path(__file__))==protocol['pilot_controller_sha256']
        configure(protocol,arm)
        def select(rows,objective):
            assert objective=='dice_bands'
            weight,statistics=calibrated_weight([r['bands_over_dice'] for r in rows],
                target=protocol['band_target_ratio'],cap=protocol['band_p95_cap'])
            return {'bands':weight},{'bands':statistics}
        def calibrate(*a,**kw):
            result=original_calibrate(*a,**kw)
            result['target_ratio']=protocol['band_target_ratio']
            result['p95_cap']=protocol['band_p95_cap']
            return result
        core.select_weights=select
        core.calibrate=calibrate
    entry.configure=pilot_configure
    sys.argv=[sys.argv[0],*arguments]
    entry.main()


def run(stage):
    plan=verify(stage)
    from scripts.adni_constraints.worker import verify_run
    import torch
    import monai
    assert torch.__version__==plan['runtime_versions']['torch'] and monai.__version__==plan['runtime_versions']['monai']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    root=Path(str(stage)+'_results');root.mkdir(exist_ok=True)
    scratch=Path('/tmp')/f'{NAME}-{os.getuid()}-{os.environ["SLURM_JOB_ID"]}'
    scratch.mkdir(mode=0o700,exist_ok=False)
    assert shutil.disk_usage(scratch).free>4*1024**3
    write(root/'launch.json',dict(job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),augmentation='none',scratch=str(scratch)))
    results=[]
    for spec in plan['runs']:
        variant=spec['variant'];target=root/variant
        assert not target.exists(),'Refuse to overwrite or replay any completed trial'
        assert quota()[0]>=1.5
        output=scratch/variant
        print(json.dumps(dict(starting=spec)),flush=True)
        subprocess.run([sys.executable,str(stage/'pilot.py'),'--mode','train','--model','nnunet',
            '--objective','dice_bands','--fold','0','--seed','0','--pkl',plan['dataset'],
            '--protocol-dir',str(stage/'conditions'/variant/'protocols/nnunet'),
            '--output',str(output),'--threads','4','--resume'],check=True)
        row=verify_run(stage/'conditions'/variant,output,spec)
        calibration=read(output/'calibration.json')
        assert calibration['anchor_successful_updates']==9 and calibration['anchor_model_sha256']==plan['expected_anchor_model_sha256']
        assert abs(calibration['weights']['bands']-spec['expected_band_weight'])<1e-12
        assert calibration['target_ratio']==spec['target'] and calibration['statistics']['bands']['weighted_p95_ratio']<=.5+1e-12
        row.update(variant=variant,target=spec['target'],band_weight=calibration['weights']['bands'],
            gain_vs_baseline_pp=100*(row['final_dice']-plan['references']['dice']['final_dice']),
            gain_vs_original_bands_pp=100*(row['final_dice']-plan['references']['dice_bands']['final_dice']))
        pending=root/(variant+'.export')
        shutil.copytree(output,pending)
        for item in output.iterdir():
            if item.is_file():assert digest(item)==digest(pending/item.name)
        pending.rename(target)
        shutil.rmtree(output)
        results.append(row)
        write(root/'summary.json',dict(status='complete' if len(results)==2 else 'in_progress',runs=results,references=plan['references'],augmentation='none',budget_flops=CAP))
        print(json.dumps(dict(completed=row)),flush=True)
    write(stage/'completion.json',dict(status='complete',runs=2,summary_sha256=digest(root/'summary.json')))


def submit(stage):
    assert os.getuid()==1395 and stage.name==NAME
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=stage/'submission.json'
        if receipt.exists():print(receipt.read_text());return
        plan=verify(stage)
        preflight=read(stage/'preflight.json')
        assert preflight['status']=='passed' and preflight['manifest_sha256']==digest(stage/'manifest.json')
        assert quota()[0]>=3
        jobs=[r.split('|') for r in subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines() if r.strip()]
        assert len(jobs)<=1 and all(r[1]=='RUNNING' for r in jobs),jobs
        command=['sbatch','--parsable']
        if jobs:command+=['--dependency=afterany:'+jobs[0][0]]
        command.append(str(stage/'run.sbatch'))
        job=subprocess.check_output(command,cwd=stage,text=True).strip().split(';')[0]
        assert job.isdigit()
        record=dict(job_id=int(job),runs=plan['runs'],augmentation='none',fold=0,seed=0,budget_gflops_per_run=CAP/1e9,
                    dependency=jobs[0][0] if jobs else None,submitted_at_unix=time.time(),manifest_sha256=digest(stage/'manifest.json'))
        write(receipt,record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['prepare','train','run','submit'],required=True)
    parser.add_argument('--stage',type=Path)
    args,remaining=parser.parse_known_args()
    if args.mode=='train':train(remaining)
    else:
        assert args.stage is not None and not remaining
        {'prepare':prepare,'run':run,'submit':submit}[args.mode](args.stage)
