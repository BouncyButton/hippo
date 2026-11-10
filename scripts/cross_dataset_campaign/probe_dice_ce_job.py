"""Prepare, validate and run the corrected 90-update MSD diagnostic."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, NAME
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

UPDATES=90
PROBE='msd_nnunet_native_dice_ce_90updates_20261014'
TASK_ID='MSD_nnunet_native_dice_ce_90updates_f0s0'
PER_UPDATE=355293896301
ORIGINAL_COMMAND=controller.command_for


def prepare():
    stage=ROOT/'experiments'/PROBE
    if (stage/'submission.json').exists():
        raise ValueError('Do not alter a submitted diagnostic')
    folder=stage/'protocols'/'MSD_nnunet_native_dice_ce'
    folder.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'experiments'/NAME
    origin=parent/'protocols'/'MSD_nnunet_target10'
    for name in ('splits_10patients.json','nnunet_architecture.json'):
        shutil.copyfile(origin/name,folder/name)
    p=deepcopy(read(origin/'protocol.json'))
    p.update(name=PROBE,diagnostic_updates=UPDATES,augmentation='none',folds=[0],seeds=[0],
        requested_arm_order=['dice'],constraints=False,
        training_loss='Native nnUNet foreground-only Dice + CE, equal weights; unchanged deep supervision',
        dice_include_background=False,diagnostic='Exactly 90 optimizer updates: nine passes over ten training volumes',
        budget_flops=UPDATES*PER_UPDATE,budget_gflops=UPDATES*PER_UPDATE/1e9,
        validation_milestones=20,validation_target_flops=[i*UPDATES*PER_UPDATE//20 for i in range(1,21)],
        checkpoint_selection='Final update-90 checkpoint is primary; validation at the original 20 FLOP milestones',
        training_source_sha256={},parent_exploration_manifest_sha256=digest(parent/'manifest.json'))
    costs={'dice':dict(update_flops=PER_UPDATE,calibration_flops=0)}
    p['models']['nnunet']['objectives']={'dice':schedule(costs,'dice',dict(
        prefix_updates=0,sdf_prefix_updates=0,constraint_warmup_updates=1),10,UPDATES*PER_UPDATE)}
    reference=ROOT/'reports/four_models_six_optimization_31976_20261006/runs_metadata/nnunet/dice/fold0/seed0/run/config.json'
    write(folder/'profile.json',dict(model='nnunet',loss_recipe='native_Dice_plus_CE',costs=costs,
        parameters=p['models']['nnunet']['parameters'],
        provenance='Existing audited native MSD nnUNet update graph, unchanged architecture/loss/optimizer; augmentation excluded from arithmetic',
        reference_config_sha256=digest(reference)))
    p['profile_sha256']=digest(folder/'profile.json')
    write(folder/'protocol.json',p)
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=PROBE,num_runs=1,tasks=[dict(id=TASK_ID,kind='conventional',dataset='MSD',
        model='nnunet',fold=0,seed=0,arm='dice',protocol=folder.name)],
        diagnostic_updates=UPDATES,budget_flops_by_dataset={'MSD':UPDATES*PER_UPDATE})
    plan['storage']['archive']=str(ROOT/'reports'/PROBE)
    write(stage/'plan.json',plan)
    write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        source_of_truth='Frozen source snapshot with explicit native Dice+CE diagnostic wrapper',
        request='Corrected request: keep exploration stopped and run MSD Dice+CE for 90 optimizer updates'))
    freeze_source(stage)
    batch=(ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text()
    batch=batch.replace('hippo-exploration-noaug','hippo-msd-ce90-noaug')
    batch=batch.replace('scripts.cross_dataset_campaign.controller run','scripts.cross_dataset_campaign.probe_dice_ce_job run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    print(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:
            raise ValueError('Frozen diagnostic changed: '+name)
    plan=read(stage/'plan.json')
    if plan['name']!=PROBE or len(plan['tasks'])!=1 or plan['augmentation']!='none':
        raise ValueError('Unexpected diagnostic plan')
    task=plan['tasks'][0];p=read(stage/'protocols'/task['protocol']/'protocol.json')
    s=p['models']['nnunet']['objectives']['dice']
    if p['diagnostic_updates']!=UPDATES or s['max_steps']!=UPDATES or p['budget_flops']!=UPDATES*PER_UPDATE:
        raise ValueError('Diagnostic must stop after exactly 90 updates')
    if p['augmentation']!='none' or p['dice_include_background'] is not False or p['constraints']:
        raise ValueError('Native, unaugmented, unconstrained recipe required')
    return plan


def command_for(stage,plan,task,base_path):
    command=ORIGINAL_COMMAND(stage,plan,task,base_path)
    command[2]='scripts.cross_dataset_campaign.probe_dice_ce'
    return command


def preflight(stage):
    plan=verify(stage);folder=stage/'protocols'/plan['tasks'][0]['protocol'];p=read(folder/'protocol.json')
    for name,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
        if digest(folder/name)!=p[key]:raise ValueError('Protocol file differs')
    if digest(p['data'])!=p['pkl_sha256']:raise ValueError('MSD dataset differs')
    split=read(folder/'splits_10patients.json')[0]
    if len(split['train'])!=10 or len(split['val'])!=52 or set(split['train'])&set(split['val']):
        raise ValueError('Unexpected data split')
    import torch
    import numpy as np
    from scripts.cross_dataset_campaign.conventional import ORIGINAL_LOSS_OPTIMIZER
    torch.set_num_threads(2);torch.manual_seed(0)
    criterion,optimizer=ORIGINAL_LOSS_OPTIMIZER('nnunet',torch.nn.Linear(1,1),p)
    if type(criterion.loss).__name__!='DC_and_CE_loss' or criterion.loss.dc.do_bg:
        raise ValueError('Native foreground Dice+CE was not restored')
    if criterion.loss.weight_ce!=1 or criterion.loss.weight_dice!=1:
        raise ValueError('Native objective weights differ')
    if not np.allclose(criterion.weight_factors,np.array([1.,.5,.25,0.])/1.75):
        raise ValueError('Deep supervision changed')
    if optimizer.param_groups[0]['lr']!=.01 or optimizer.param_groups[0]['momentum']!=.95 or p['lr_horizon_updates']!=174:
        raise ValueError('Historical optimizer/schedule differs')
    outputs=[torch.randn(1,3,n,n,n,requires_grad=True) for n in (8,4,2,1)]
    labels=[torch.randint(0,3,(1,1,n,n,n)) for n in (8,4,2,1)]
    loss=criterion(outputs,labels);loss.backward()
    if not torch.isfinite(loss) or any(not torch.isfinite(x.grad).all() for x in outputs[:3]):
        raise ValueError('Nonfinite native loss/gradients')
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),
        checks=['Frozen source/data/split identities','Exactly 90 updates, no augmentation/constraints',
                'Native foreground Dice+CE equal weights, original deep supervision, finite CPU backward',
                'Unchanged SGD and 174-update LR horizon','Shell syntax'],native_loss_test=float(loss.detach())))
    print('Native Dice+CE 90-update preflight passed',flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare();return
    controller.verify=verify;controller.command_for=command_for
    if len(sys.argv)>1 and sys.argv[1]=='preflight':
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('--stage',type=Path,required=True)
        args=parser.parse_args();preflight(args.stage);return
    controller.main()


if __name__=='__main__':main()
