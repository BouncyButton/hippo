"""Finite four-model pilots for a lower, shared MNI optimization budget."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, install_signals
from scripts.cross_dataset_campaign import controller
from scripts.unetr_subfields import cap_extension as batch
from scripts.mni_constraints import campaign

ROOT = campaign.ROOT
MODULE = 'scripts.mni_constraints.budget_pilot'
HISTORICAL_CAP = 31976450667090
PARENT = 'mni_constraint_target_search_f0s0_20261107'


def tasks(recipe='winning', models=None):
    return [dict(id=f'MNI_{model}_f0s0', model=model, dataset='MNI', kind='conventional',
                 protocol=model, arm='dice', fold=0, seed=0,
                 augmentation='xy_d4_three_quarters_v1' if recipe=='headroom_v3' and model=='unetr_vit_b16' else
                    'xy_d4_half_v1' if recipe=='headroom_v2' and model=='unetr_vit_b16' else 'xy_d4_v1' if model=='unetr_vit_b16' or
                    (recipe in ('headroom_v1','headroom_v2','headroom_v3') and model=='swinunetr') or
                    (recipe=='headroom_v1' and model=='basic_unet16') else 'none')
            for model in batch.MODELS if models is None or model in models]


def prepare(name, multiplier, recipe_name='winning', models=None, lr_overrides=None):
    stage = ROOT/'experiments'/name
    if (stage/'manifest.json').exists():
        return verify(stage)
    parent = ROOT/'experiments'/PARENT
    for part in ('source','data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__), stage/'source/scripts/mni_constraints/budget_pilot.py')
    trainer = stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    source = trainer.read_text()
    marker = '        score = result["summary"]["macro_hard_dice"]'
    assert source.count(marker)==1
    source = source.replace(marker, '''        training_result = base.evaluate(model, train, device, amp)
        base.atomic_json(args.output / "training_latest.json", training_result)
        training_summary = {"training_" + key: value for key, value in training_result["summary"].items()}
''' + marker)
    marker = '               **result["summary"]}'
    assert source.count(marker)==1
    source = source.replace(marker, '               **result["summary"], **training_summary}')
    trainer.write_text(source)
    if recipe_name in ('headroom_v1','headroom_v2','headroom_v3'):
        storage=stage/'source/scripts/cross_dataset_campaign/storage.py'
        source=storage.read_text()
        marker="and config['model_identity']=='unetr_vit_b16'"
        assert source.count(marker)==1
        storage.write_text(source.replace(marker,"and config['model_identity'] in ('unetr_vit_b16','swinunetr','basic_unet16')"))
    if recipe_name in ('headroom_v2','headroom_v3'):
        source=trainer.read_text()
        marker='        if protocol["augmentation"] == "xy_d4_v1":'
        assert source.count(marker)==1
        trainer.write_text(source.replace(marker,'        if protocol["augmentation"] == "xy_d4_v1" or (protocol["augmentation"] == "xy_d4_half_v1" and charged % 2 == 0) or (protocol["augmentation"] == "xy_d4_three_quarters_v1" and charged % 4 != 3):'))
        adapter=stage/'source/scripts/unetr_subfields/cap_extension.py'
        source=adapter.read_text();marker="protocol['augmentation']=='xy_d4_v1'"
        assert source.count(marker)==1
        adapter.write_text(source.replace(marker,"protocol['augmentation'] in ('xy_d4_v1','xy_d4_half_v1','xy_d4_three_quarters_v1')"))
        source=storage.read_text()
        marker="config['augmentation']=='xy_d4_v1' and task.get('augmentation')=='xy_d4_v1'"
        assert source.count(marker)==1
        storage.write_text(source.replace(marker,"config['augmentation'] in ('xy_d4_v1','xy_d4_half_v1','xy_d4_three_quarters_v1') and task.get('augmentation')==config['augmentation']"))
    recipes = read(campaign.RECIPES)
    models=list(batch.MODELS) if models is None else list(models)
    lr_overrides={} if lr_overrides is None else lr_overrides
    assert models and len(set(models))==len(models) and set(models)<=set(batch.MODELS)
    assert set(lr_overrides)<=set(models) and all(math.isfinite(v) and v>0 for v in lr_overrides.values())
    cap = round(multiplier*HISTORICAL_CAP)
    for model, recipe in recipes.items():
        if model not in models:continue
        origin = ROOT/'experiments'/recipe['stage']/'protocols/model'
        folder = stage/'protocols'/model
        shutil.copytree(origin, folder)
        p = deepcopy(recipe['protocol'])
        if recipe_name in ('headroom_v1','headroom_v2','headroom_v3'):
            p['augmentation']=next(t['augmentation'] for t in tasks(recipe_name) if t['model']==model)
            p['headroom_recipe']=recipe_name
            if model=='unetr_vit_b16':p['lr_policy']='hold_half'
            p['preprocessing']='Native nonzero z-score and symmetric padding; pairedxyD4 augmentation where declared; no interpolation or label removal.'
        profile = read(folder/'profile.json')
        count = cap//profile['update_flops']
        alias = p['trainer_alias']
        if model in lr_overrides:p['initial_lrs']={alias:lr_overrides[model]}
        p.update(name=name, data=str(REMOTE/name/'data/MNI.pkl'), budget_flops=cap,
                 models={alias:dict(parameters=profile['parameters'], objectives={'dice':batch.schedule(profile['update_flops'],cap)})},
                 folds=[0], seeds=[0], requested_arm_order=['dice'], constraints=[],
                 lr_horizon_updates=count, constraint_warmup_updates=1,
                 validation_target_flops=[cap*i//20 for i in range(1,21)],
                 calibration_target_ratio=.1, calibration_p95_cap=.5,
                 budget_origin=f'Lower shared cap pilot: {multiplier} times historical cap, selected using training Dice.',
                 checkpoint_selection='Final-budget checkpoint primary; validation does not determine the cap.',
                 base_protocol_sha256=digest(origin/'protocol.json'),
                 lr_schedule_details=dict(policy=p['lr_policy'],horizon_updates=count,
                     warmup_updates=0 if model=='nnunet' else 24 if model=='unetr_vit_b16' else math.ceil(.05*count),
                     decay_start_index=count//2 if p['lr_policy']=='hold_half' else 24 if model=='unetr_vit_b16' else math.ceil(.05*count)),
                 training_dice_monitor='Unaugmented evaluation of all10 training crops at every5% optimization-FLOP milestone; evaluation compute excluded from optimization cap.')
        write(folder/'protocol.json',p)
    plan = dict(name=name, tasks=tasks(recipe_name,models), models=models, lr_overrides=lr_overrides,
        recipe_name=recipe_name, num_runs=len(models), multiplier=multiplier, budget_flops=cap,
        historical_cap=HISTORICAL_CAP, gpu_workers=2, excluded_nodes=['gnode04'], augmentation='model_specific',
        conventional_python='/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python',
        graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/name)),
        criterion=dict(training_dice_lower=.85,training_dice_upper=.90,selection_metric='Training macro hard Dice only; validation reported, not used to tune cap'),
        authorization='User requested a lower shared cap for all4 models with85–90% training Dice headroom. Two GPU workers; at most one running and one queued job. This finite pilot does not restart the constraint-target search.',
        limitation='One common cap may not put every model in85–90%; measure and report compatibility rather than invent model-specific limits.')
    write(stage/'plan.json',plan)
    write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        lr_overrides=lr_overrides,
        changes=('Lower common cap and matched LR horizons; existing losses, optimizers and data retained; peak LR overrides are explicitly listed. Add training Dice evaluation at validation milestones. No training-Dice early stopping. '+
                 ('Headroom_v1 adds pairedD4 to Swin/UNet and uses hold-half LR for Geo UNETR.' if recipe_name=='headroom_v1' else
                  'Headroom_v2 retains pairedD4 for Swin, halves its frequency for Geo UNETR (even update indices only), keeps hold-half LR for Geo UNETR; smallUNet returns to no augmentation.' if recipe_name=='headroom_v2' else
                  'Headroom_v3 applies pairedD4 on three of every four Geo UNETR updates (indices modulo4 other than3); hold-half LR, all losses and FLOP costs unchanged. Other models retain v2 augmentation choices.' if recipe_name=='headroom_v3' else 'Existing augmentations retained.'))))
    sbatch=(parent/'run.sbatch').read_text().replace('hippo-mni-target-search','hippo-mni-budget').replace('scripts.mni_constraints.target_search',MODULE)
    (stage/'run.sbatch').write_text(sbatch)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json')
    assert plan['tasks']==tasks(plan.get('recipe_name','winning'),plan.get('models')) and plan['gpu_workers']==2
    assert len(plan['tasks'])==plan['num_runs']
    assert plan['budget_flops']<campaign.CAP
    for task in plan['tasks']:
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        cost=read(folder/'profile.json')['update_flops']
        s=p['models'][p['trainer_alias']]['objectives']['dice']
        assert p['budget_flops']==plan['budget_flops'] and s==batch.schedule(cost,plan['budget_flops'])
        assert p['lr_horizon_updates']==s['max_steps'] and p['folds']==p['seeds']==[0]
        assert p['constraints']==[] and p['augmentation']==task['augmentation']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name)==p[key],name
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    return plan


def preflight(stage):
    import torch
    torch.set_num_threads(4)
    from scripts.unetr_subfields import core
    from types import SimpleNamespace
    plan=verify(stage)
    augmentation_checks=[]
    if plan.get('recipe_name') in ('headroom_v1','headroom_v2','headroom_v3'):
        from scripts.unetr_subfields.geo_recipe import apply_transform
        # A voxel ID image paired with categorical labels catches any image/label
        # mismatch, including flips which preserve aggregate class counts.
        image=torch.arange(64**3).reshape(1,1,64,64,64).float()
        label=(image.long()%4)
        for index in range(8):
            a,b=apply_transform(image,label,index)
            assert torch.equal(a.long()%4,b)
            assert torch.equal(torch.bincount(b.flatten(),minlength=4),torch.bincount(label.flatten(),minlength=4))
            augmentation_checks.append(index)
        if plan['recipe_name'] in ('headroom_v2','headroom_v3'):
            source=(stage/'source/scripts/cross_dataset_campaign/capped_core.py').read_text()
            assert 'protocol["augmentation"] == "xy_d4_half_v1" and charged % 2 == 0' in source
            assert sum(i%2==0 for i in range(20))==10
            if plan['recipe_name']=='headroom_v3':
                assert 'protocol["augmentation"] == "xy_d4_three_quarters_v1" and charged % 4 != 3' in source
                assert sum(i%4!=3 for i in range(20))==15
    rows=[]
    for task in plan['tasks']:
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        identity=task['model'];alias=p['trainer_alias'];n=p['lr_horizon_updates'];initial=p['initial_lrs'][alias]
        lrs=[batch.lr_at(i,n,initial,identity,p['lr_policy']) for i in range(n+10)]
        assert all(0<=x<=initial*1.000001 for x in lrs)
        assert lrs[-1]<=initial*.010001
        batch.bind(SimpleNamespace(protocol_dir=folder,model=alias,objective='dice',fold=0))
        if identity=='unetr_vit_b16' and p.get('headroom_recipe') in ('headroom_v2','headroom_v3'):
            core.trainer.check_no_augmentation(p)
        model=core.base.build_model(alias,read(folder/'nnunet_architecture.json')).to('meta')
        assert sum(x.numel() for x in model.parameters())==p['models'][alias]['parameters']
        output=model(torch.zeros(1,1,64,64,64,device='meta'))
        shapes=[list(x.shape) for x in output] if isinstance(output,(list,tuple)) else [list(output.shape)]
        profile=read(folder/'profile.json')
        expected_shapes=([[1,4,64,64,64]] if identity=='unetr_vit_b16' else profile['output_shapes'])
        assert shapes==expected_shapes
        rows.append(dict(model=identity,updates=n,patient_passes=n/10,cap=p['budget_flops'],last_lr=lrs[n-1],loss=p['training_loss']))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=rows,
                paired_augmentation_transforms_checked=augmentation_checks,optimizer_updates=0)
    write(stage/'preflight.json',result)
    return result


def command_for(stage,plan,task,base_path=None):
    folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
    return [plan['conventional_python'],'-u','-m',MODULE,'train','--model',p['trainer_alias'],
            '--objective','dice','--fold','0','--seed','0','--pkl',str(stage/'data/MNI.pkl'),
            '--protocol-dir',str(folder),'--output',str(stage/'runs'/task['id']),'--resume']


def train(args):
    p=read(args.protocol_dir/'protocol.json')
    if p.get('headroom_recipe') in ('headroom_v1','headroom_v2','headroom_v3'):
        from scripts.unetr_subfields import core
        def declared_augmentation(protocol):
            identity=protocol['model_identity']
            expected=next(t['augmentation'] for t in tasks(protocol['headroom_recipe']) if t['model']==identity)
            assert protocol['augmentation']==expected
        core.trainer.check_no_augmentation=declared_augmentation
    campaign.train(args)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--name',required=True);p.add_argument('--multiplier',type=float,required=True)
    p.add_argument('--recipe',choices=['winning','headroom_v1','headroom_v2','headroom_v3'],default='winning')
    p.add_argument('--models',nargs='+',choices=batch.MODELS)
    p.add_argument('--lr-overrides',type=json.loads,default={})
    for command in ('verify','preflight','submit','run','run-one'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
        if command=='run-one':p.add_argument('--task',required=True)
    p=sub.add_parser('train');p.add_argument('--model',required=True);p.add_argument('--objective',required=True)
    for key in ('fold','seed'):p.add_argument('--'+key,type=int,required=True)
    for key in ('pkl','protocol-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();install_signals()
    controller.verify=verify;controller.command_for=command_for
    batch.verify=verify;batch.command_for=command_for;batch.RUNNER_MODULE=MODULE
    if args.command=='prepare':print(json.dumps(prepare(args.name,args.multiplier,args.recipe,args.models,args.lr_overrides)))
    elif args.command=='verify':print(json.dumps(verify(args.stage)))
    elif args.command=='preflight':print(json.dumps(preflight(args.stage)))
    elif args.command=='submit':print(json.dumps(controller.submit(args.stage)))
    elif args.command=='run':batch.run(args.stage)
    elif args.command=='run-one':batch.run_one(args)
    else:train(args)


if __name__=='__main__':main()
