"""Six authorized unaugmented 3D UNet Dice+CE constraint comparisons."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest, CAP
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

PROBE = 'msd_unet_ce_constraints_20261014'
ARMS = ('dice_bands', 'dice_bands_edge_depth')
TARGETS = (.1, .4, .6)
P95_CAPS = {.1: .5, .4: .5, .6: .75}
AUTHORIZATION = 'User requests only 3D UNet first: bands and original-fourteen winner bands+edge+depth, each 10/40/60%, all added to Dice+CE.'
JOB_NAME = 'hippo-unet-ce-constraints'
JOB_MODULE = 'scripts.cross_dataset_campaign.probe_unet_constraints_job'
ORIGINAL_COMMAND = controller.command_for


def prepare():
    stage = ROOT/'experiments'/PROBE
    if (stage/'submission.json').exists():
        raise ValueError('Never modify a submitted pilot')
    parent = ROOT/'experiments/msd_other_models_loss_pairs_20261014'
    baseline = parent/'protocols/basic_unet16_dice_ce'
    baseline_run = ROOT/'reports/msd_other_models_loss_pairs_20261014/runs/MSD_basic_unet16_dice_ce_f0s0'
    from scripts.cross_dataset_campaign.storage import verify_archive
    verify_archive(baseline_run, read(baseline_run/'archive_manifest.json'))
    old = read(ROOT/'experiments/cross_dataset_full_20261014/protocols/MSD_basic_unet16_f0_target10/protocol.json')
    profile_path = ROOT/'reports'/PROBE/'profile.json'
    profile = read(profile_path)
    if profile['status'] != 'profiled' or profile['calibration_base'] != 'Dice+CE':
        raise ValueError('Exact constraint/CE profiles required')
    selection = read(ROOT/'reports'/PROBE/'historical_selection.json')
    if selection['ranking'][0]['arm'] != ARMS[1] or selection['choice'] != ARMS[1]:
        raise ValueError('Use the user-selected best of the original fourteen conditions')
    write(stage/'historical_selection.json', selection)
    for target in TARGETS:
        folder = stage/'protocols'/f'basic_unet16_target{int(target*100)}'
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('splits_10patients.json','nnunet_architecture.json'):
            shutil.copyfile(baseline/name, folder/name)
        shutil.copyfile(profile_path, folder/'profile.json')
        p = deepcopy(read(baseline/'protocol.json'))
        p.pop('diagnostic_updates',None); p.pop('diagnostic',None)
        p.update(name=PROBE, augmentation='none', weight_ce=1, folds=[0], seeds=[0],
            constraints=True, requested_arm_order=list(ARMS), calibration_target_ratio=target,
            calibration_p95_cap=P95_CAPS[target], calibration_base='Dice+CE',
            training_loss='Unchanged MONAI Dice + mean CE, coefficients one, plus calibrated original constraints',
            checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
            validation_milestones=20, validation_target_flops=[CAP*i//20 for i in range(1,21)],
            training_source_sha256={}, profile_sha256=digest(folder/'profile.json'),
            baseline_reference=dict(campaign=parent.name,run=baseline_run.name,
                protocol_sha256=digest(baseline/'protocol.json'),
                initialization_sha256=read(baseline_run/'initialization.json')['model_state_sha256'],
                validation_sha256=digest(baseline_run/'validation_latest.json'),
                validation_dice=read(baseline_run/'validation_latest.json')['summary']['macro_hard_dice']))
        p['models'] = {'unet':dict(parameters=profile['parameters'],objectives={
            arm:schedule(profile['costs'],arm,old['models']['unet']['objectives'][arm],10,CAP) for arm in ARMS})}
        write(folder/'protocol.json',p)
    tasks = [dict(id=f'MSD_basic_unet16_ce_{arm}_target{int(target*100)}_f0s0',kind='conventional',
        dataset='MSD',model='basic_unet16',fold=0,seed=0,arm=arm,target=target,
        protocol=f'basic_unet16_target{int(target*100)}') for arm in ARMS for target in TARGETS]
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=PROBE,num_runs=len(tasks),tasks=tasks,
        authorization=AUTHORIZATION,
        exploration_followup='Other-model pilots, exploration and full sweep remain unqueued',
        baseline_reference_score=0.8117938473325637)
    plan['storage'].update(archive=str(ROOT/'reports'/PROBE),estimated_peak_logical_gib=1,
        mirrored_peak_gib=2,reserve_gib=2,minimum_free_gib=4)
    write(stage/'plan.json',plan)
    write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=False))
    write(stage/'source_version.json',dict(request=plan['authorization'],
        source_of_truth='Frozen source files and SHA256 manifest include uncommitted changes',
        baseline_manifest_sha256=digest(parent/'manifest.json')))
    freeze_source(stage)
    batch=(ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text()
    batch=batch.replace('hippo-exploration-noaug',JOB_NAME).replace(
        'scripts.cross_dataset_campaign.controller run',JOB_MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    print(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen pilot changed: '+name)
    plan=read(stage/'plan.json')
    if plan['name'] != PROBE or plan['augmentation'] != 'none' or len(plan['tasks']) != len(ARMS)*len(TARGETS):
        raise ValueError('Unexpected unaugmented 3D UNet run matrix')
    combinations=[]
    for task in plan['tasks']:
        p=read(stage/'protocols'/task['protocol']/'protocol.json')
        profile=read(stage/'protocols'/task['protocol']/'profile.json')
        if (task['model'],task['fold'],task['seed'],p['model_identity']) != ('basic_unet16',0,0,'basic_unet16'):
            raise ValueError('Only the authorized 3D UNet fold0 seed0 pilot is allowed')
        if (p['weight_ce'] != 1 or p['augmentation'] != 'none' or p['budget_flops'] != CAP
                or p['calibration_base'] != 'Dice+CE' or p['calibration_target_ratio'] != task['target']
                or p['calibration_p95_cap'] != P95_CAPS[task['target']]):
            raise ValueError('Incorrect baseline, cap, augmentation or calibration')
        s=p['models']['unet']['objectives'][task['arm']]
        if s != schedule(profile['costs'],task['arm'],s,10,CAP):
            raise ValueError('Schedule differs from measured FLOPs')
        if s['prefix_updates'] != 50 or s['constraint_warmup_updates'] != 50:
            raise ValueError('Historical prefix/ramp changed')
        combinations.append((task['arm'],task['target']))
    if combinations != [(a,t) for a in ARMS for t in TARGETS]:
        raise ValueError('Unexpected task order or conditions')
    return plan


def command_for(stage,plan,task,base_path):
    command=ORIGINAL_COMMAND(stage,plan,task,base_path)
    command[2]='scripts.cross_dataset_campaign.probe_unet_constraints'
    return command


def preflight(stage):
    plan=verify(stage)
    import numpy as np
    import torch
    from scripts.cross_dataset_campaign import probe_unet_constraints as obj
    from scripts.cross_dataset_campaign.probe_ce_bands import calibration_case as band_calibration
    from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
    from scripts.cross_dataset_campaign.conventional import select_at_target
    torch.set_num_threads(2);torch.manual_seed(0)
    folder=stage/'protocols'/plan['tasks'][0]['protocol'];p=read(folder/'protocol.json')
    if digest(p['data']) != p['pkl_sha256']:
        raise ValueError('Dataset hash changed')
    for target in TARGETS:
        folder=stage/'protocols'/f'basic_unet16_target{int(target*100)}'
        protocol=read(folder/'protocol.json')
        for name,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
            if digest(folder/name) != protocol[key]:raise ValueError('Artifact changed: '+name)
        split=read(folder/'splits_10patients.json')[0]
        if len(split['train']) != 10 or len(split['val']) != 52 or set(split['train']) & set(split['val']):
            raise ValueError('Expected identical 10/52 split')
        if protocol['initial_lrs']['unet'] != .0003 or protocol['lr_horizon_updates'] != 1095:
            raise ValueError('Historical LR recipe changed')
    axis=torch.arange(64);x,y,z=torch.meshgrid(axis,axis,axis,indexing='ij')
    foreground=((x-32)/18)**2+((y-32)/12)**2+((z-32)/10)**2<1
    labels=torch.where(foreground,torch.where(x<32,1,2),0)[None,None]
    logits=torch.randn(1,3,64,64,64,requires_grad=True)
    affine=np.eye(4);affine[:3,:3]=np.array([[0,1,0],[1,0,0],[0,0,1]])
    depth=obj.original.geometry_weights(labels,affine,p['depth_geometry'])
    base=paired_loss(logits,labels,'dice',{},0.,weight_ce=1)
    if not torch.equal(base,obj.training_loss(logits,labels,'dice',{},0.)):
        raise ValueError('Dice+CE baseline changed')
    criterion,optimizer=obj.experiment.ORIGINAL_LOSS_OPTIMIZER('unet',torch.nn.Linear(1,1),p)
    if not criterion.include_background or type(optimizer).__name__ != 'AdamW':
        raise ValueError('Established Dice/optimizer changed')
    direct_grad={'dice':torch.autograd.grad(base,logits,retain_graph=True)[0]}
    terms=dict(bands=obj.original.OuterBoundaryBandLoss()(logits,labels).loss,
        edge=obj.original.BoundaryEdgeConsistencyLoss()(logits,labels),
        depth=obj.original.depth_loss(logits,labels,depth))
    for key,loss in terms.items():
        direct_grad[key]=torch.autograd.grad(loss,logits,retain_graph=True)[0]
    for arm in ARMS:
        used=[k for k in terms if k in arm];weights={k:.13 for k in used}
        loss=obj.training_loss(logits,labels,arm,weights,.4,depth=depth)
        expected=base+sum(.4*weights[k]*terms[k] for k in used)
        if not torch.allclose(loss,expected,atol=1e-6):
            raise ValueError('Auxiliaries are not added to Dice+CE')
        gradient,=torch.autograd.grad(loss,logits,retain_graph=True)
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite gradient')
        row=obj.calibration_case(logits,labels,arm,depth=depth)
        if not np.isclose(row['dice_rms'],float(direct_grad['dice'].square().mean().sqrt()),rtol=1e-6):
            raise ValueError('Calibration denominator is not Dice+CE')
        if arm == 'dice_bands':
            if row != band_calibration(logits,labels,arm):raise ValueError('Bands calibration changed')
        else:
            for a,ga in direct_grad.items():
                for b,gb in direct_grad.items():
                    if not np.isclose(row['_depth_pair_means'][a+':'+b],float((ga*gb).mean()),rtol=1e-6,atol=1e-20):
                        raise ValueError('Incorrect calibration Gram matrix')
        for target in TARGETS:
            cap=P95_CAPS[target]
            selected,stats=select_at_target([deepcopy(row) for _ in range(10)],arm,target,cap)
            joint=direct_grad['dice'].clone()
            for name in used:
                measured=float((selected[name]*direct_grad[name]).square().mean().sqrt()/joint.square().mean().sqrt())
                if not np.isclose(measured,target,rtol=1e-5):raise ValueError('Sequential joint-gradient target differs')
                if stats[name]['weighted_p95_ratio'] > cap+1e-8:raise ValueError('p95 cap exceeded')
                joint=joint+selected[name]*direct_grad[name]
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=[
        f'{len(plan["tasks"])} authorized 3D UNet conditions; original-fourteen winner verified',
        'Frozen data/source and 10/52 split, fold0 seed0, no augmentation',
        'Both constraints added on top of unchanged Dice+CE',
        'Correct gradients and sequential calibration against joint Dice+CE baseline',
        f'Targets {TARGETS} and p95 ceilings {P95_CAPS}',
        'Measured FLOP schedules and original 50-update prefix/ramp',
        'Original AdamW and 1095-update LR horizon', 'Shell syntax']))
    print('3D UNet Dice+CE constraints preflight passed',flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        prepare();return
    controller.verify=verify
    controller.command_for=command_for
    if len(sys.argv)>1 and sys.argv[1]=='preflight':
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('--stage',type=Path,required=True)
        preflight(parser.parse_args().stage);return
    controller.main()


if __name__=='__main__':
    main()
