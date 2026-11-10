"""Two-GPU MNI baseline batches at an explicitly chosen common FLOP cap."""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, stopping, install_signals
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import quota, prepare as prepare_archive, retire
from scripts.unetr_subfields.contracts import ROOT, CAP, DATASETS, subject

MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')
RUNNER_MODULE = 'scripts.unetr_subfields.cap_extension'


def lr_at(step, maximum, initial, identity, policy):
    """A bounded, non-cycling schedule; long budget must not revive cosine LR."""
    if policy == 'constant': return initial
    if identity == 'nnunet':
        hold = maximum//2 if policy == 'hold_half' else 0
        progress = max(0., min(1., (step-hold)/max(1, maximum-hold)))
        return initial * (1-progress)**.9
    warmup = 24 if identity == 'unetr_vit_b16' else max(1, math.ceil(.05*maximum))
    if step < warmup: return initial*(step+1)/warmup
    decay_start=max(warmup,maximum//2) if policy=='hold_half' else warmup
    if step<decay_start:return initial
    progress = min(1., max(0., (step-decay_start)/max(1, maximum-decay_start-1)))
    return initial*(.01+.99*(1+math.cos(math.pi*progress))/2)


def schedule(cost, cap):
    updates=cap//cost
    if type(cost) is not int or type(cap) is not int or updates < 1: raise ValueError('Invalid budget')
    return dict(prefix_updates=0,sdf_prefix_updates=0,constraint_warmup_updates=1,sdf_warmup_updates=1,
        baseline_flops_per_update=cost,early_flops_per_update=cost,flops_per_update=cost,calibration_flops=0,
        budget_scope='optimization_only',max_steps=updates,consumed_training_flops=updates*cost,
        consumed_flops=updates*cost,consumed_total_flops=updates*cost,unused_flops=cap-updates*cost,patient_pass_epochs=updates/10)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha: raise ValueError('Frozen artifact changed: '+name)
    plan=read(stage/'plan.json');p=read(stage/'protocols/model/protocol.json');profile=read(stage/'protocols/model/profile.json')
    alias=p['trainer_alias'];arm=p['requested_arm_order'][0]
    assert p['dataset']=='MNI' and p['folds']==[0,1,2] and p['seeds']==[0,1,2] and p['train_count']==10
    assert p['classes']==DATASETS['MNI']['classes'] and p['spatial_size']==[64,64,64]
    assert p['budget_flops']==plan['budget_flops'] and p['models'][alias]['objectives'][arm]==schedule(profile['update_flops'],p['budget_flops'])
    assert p['lr_horizon_updates']==p['models'][alias]['objectives'][arm]['max_steps']
    assert p['model_identity']==plan['model'] and not p['constraints']
    for name,key in [('profile.json','profile_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('splits_10patients.json','splits_sha256')]:
        assert digest(stage/'protocols/model'/name)==p[key],name
    assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    for split in read(stage/'protocols/model/splits_10patients.json'):
        train,val=split['train'],split['val']
        assert len(train)==len(val)==10 and len({subject(c) for c in train})==5
        assert not {subject(c) for c in train}&{subject(c) for c in val}
    assert plan['tasks']==task_list(plan['model'],plan['pilot'],plan.get('pilot_fold',0),plan.get('pilot_all_seeds',False))
    return plan


def task_list(identity,pilot=False,pilot_fold=0,pilot_all_seeds=False):
    pairs=[(pilot_fold,seed) for seed in range(3 if pilot_all_seeds else 2)] if pilot else [(fold,seed) for seed in range(3) for fold in range(3)]
    return [dict(id=f'MNI_{identity}_f{fold}s{seed}',kind='conventional',dataset='MNI',model=identity,
                 fold=fold,seed=seed,arm='dice_ce' if identity=='unetr_vit_b16' else 'dice',protocol='model',
                 augmentation='xy_d4_v1' if identity=='unetr_vit_b16' else 'none') for fold,seed in pairs]


def prepare(args):
    stage=ROOT/'experiments'/args.name;parent=ROOT/'experiments/mni_matched_lr_dice_ce_20261106'
    if (stage/'manifest.json').exists(): return verify(stage)
    stage.mkdir(parents=True,exist_ok=True)
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    geo=args.model=='unetr_vit_b16'
    reference=(ROOT/'experiments/unetr_subfields_geo_recipe_seed_search_20261106/protocols/MNI' if geo else
               ROOT/'experiments/mni_other_baselines_20261106/protocols'/args.model)
    folder=stage/'protocols/model';shutil.copytree(reference,folder,dirs_exist_ok=True)
    uniform=getattr(args,'swin_loss','original')=='uniform_allclass'
    allclass=getattr(args,'swin_loss','original') in ('balanced_allclass','uniform_allclass')
    balanced=getattr(args,'swin_loss','original') in ('balanced_squared','balanced_allclass','uniform_allclass')
    squared=getattr(args,'swin_loss','original') in ('foreground_squared','balanced_squared','balanced_allclass','uniform_allclass')
    logarithmic=getattr(args,'swin_loss','original')=='log_squared'
    if logarithmic:
        assert args.model=='swinunetr'
        shutil.copy2(ROOT/'reports/mni_baseline_rescue_20261106/profiles/swin_log_squared.json',folder/'profile.json')
    if squared:
        assert args.model=='swinunetr'
        profile_name='swin_uniform_allclass.json' if uniform else 'swin_balanced_allclass.json' if allclass else 'swin_balanced_squared.json' if balanced else 'swin_foreground_squared.json'
        shutil.copy2(ROOT/'reports/mni_baseline_rescue_20261106/profiles'/profile_name,folder/'profile.json')
    p=read(folder/'protocol.json');profile=read(folder/'profile.json');alias='unet' if args.model in ('unetr_vit_b16','basic_unet16') else args.model
    arm='dice_ce' if geo else 'dice';cap=args.multiplier*CAP;count=cap//profile['update_flops']
    p.update(name=args.name,data=str(REMOTE/args.name/'data/MNI.pkl'),budget_flops=cap,seeds=[0,1,2],folds=[0,1,2],
             trainer_alias=alias,lr_horizon_updates=count,models={alias:dict(parameters=profile['parameters'],objectives={arm:schedule(profile['update_flops'],cap)})},
             validation_target_flops=[cap*i//20 for i in range(1,21)],lr_policy=args.lr_policy,
             learning_rate_policy='Cap-bounded schedule; nnUNet '+args.lr_policy+'; AdamW warmup+cosine; Geo retains24-step warmup.',
             selection_caveat='Development validation used for recipe/cap exploration; not an untouched test estimate.')
    checkpoint_interval=getattr(args,'checkpoint_interval',5)
    if checkpoint_interval not in (5,25):raise ValueError('Unsupported checkpoint cadence')
    p['checkpoint_interval_updates']=checkpoint_interval
    warmup=0 if args.model=='nnunet' or args.lr_policy=='constant' else 24 if geo else math.ceil(.05*count)
    decay_start=count//2 if args.lr_policy=='hold_half' else warmup
    p['learning_rate_policy']=f'{args.lr_policy}; warmup={warmup} updates, decay starts at update index{decay_start}; horizon={count}. nnUNet uses polynomial0.9; AdamW models use bounded cosine to1%peak.'
    p['lr_schedule_details']=dict(policy=args.lr_policy,warmup_updates=warmup,decay_start_index=decay_start,horizon_updates=count)
    if squared:
        p.update(training_loss=profile['recipe']['loss'],loss_recipe='foreground_squared_dice',
                 dice_loss_arguments=profile['loss_arguments'],profile_sha256=digest(folder/'profile.json'))
        for filename in ('foreground_squared_dice.py','profile_foreground_squared.py'):
            shutil.copy2(ROOT/'scripts/unetr_subfields'/filename,stage/'source/scripts/unetr_subfields'/filename)
    if balanced:
        weights=read(ROOT/'reports/mni_baseline_rescue_20261106/swin_training_class_weights.json')
        assert weights['dataset_sha256']==p['pkl_sha256'] and weights['splits_sha256']==p['splits_sha256']
        if uniform:
            weights['formula']='Uniform foreground weights1; training counts retained only for audit.'
            for row in weights['folds'].values():row['weights']=[1.,1.,1.]
        p.update(loss_recipe='uniform_allclass_squared_dice' if uniform else 'balanced_allclass_squared_dice' if allclass else 'balanced_foreground_squared_dice',training_class_weights=weights,
                 output_head_initialization='zero_weights_and_bias_uniform_initial_softmax')
        if allclass:p['background_class_weight']=1.
        for filename in ('balanced_squared_dice.py','profile_balanced_squared.py'):
            shutil.copy2(ROOT/'scripts/unetr_subfields'/filename,stage/'source/scripts/unetr_subfields'/filename)
        if allclass:shutil.copy2(ROOT/'scripts/unetr_subfields/profile_balanced_allclass.py',stage/'source/scripts/unetr_subfields/profile_balanced_allclass.py')
        if uniform:shutil.copy2(ROOT/'scripts/unetr_subfields/profile_uniform_allclass.py',stage/'source/scripts/unetr_subfields/profile_uniform_allclass.py')
    if logarithmic:
        p.update(training_loss=profile['recipe']['loss'],loss_recipe='log_squared_dice',
                 dice_loss_arguments=profile['loss_arguments'],profile_sha256=digest(folder/'profile.json'))
        for filename in ('log_squared_dice.py','profile_log_squared.py'):
            shutil.copy2(ROOT/'scripts/unetr_subfields'/filename,stage/'source/scripts/unetr_subfields'/filename)
    if args.lr is not None:p['initial_lrs']={alias:args.lr}
    write(folder/'protocol.json',p)
    shutil.copy2(ROOT/'scripts/unetr_subfields/cap_extension.py',stage/'source/scripts/unetr_subfields/cap_extension.py')
    if checkpoint_interval!=5:
        trainer=stage/'source/scripts/cross_dataset_campaign/capped_core.py';source=trainer.read_text()
        marker=' or step % 5 == 0:'
        assert source.count(marker)==1
        trainer.write_text(source.replace(marker,f' or step % {checkpoint_interval} == 0:'))
    pilot_fold=getattr(args,'pilot_fold',0);pilot_all_seeds=getattr(args,'pilot_all_seeds',False)
    tasks=task_list(args.model,args.pilot,pilot_fold,pilot_all_seeds)
    plan=dict(name=args.name,model=args.model,pilot=args.pilot,pilot_fold=pilot_fold,pilot_all_seeds=pilot_all_seeds,tasks=tasks,budget_flops=cap,
        historical_cap=CAP,multiplier=args.multiplier,augmentation=p['augmentation'],num_runs=len(tasks),
        conventional_python='/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python',
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module='scripts.unetr_subfields.cap_extension',
        gpu_workers=2,checkpoint_interval_updates=checkpoint_interval,storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/args.name)),
        criterion=dict(minimum_run_dice=.45,minimum_seed_mean=.50),
        authorization='User authorized changing recipes and increasing a shared cap to obtain four MNI baselines at fixed3folds×3seeds. TwoGPU workers, maxone running/onequeued job. Swin Dice-only; UNETR Geo; per-runarchive/cleanup.',
        previous_evidence=args.evidence)
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),reference_protocol_sha256=digest(reference/'protocol.json'),
        checkpoint_interval_updates=checkpoint_interval,
        changes='Larger explicit FLOP cap and bounded LR schedule; exact data preserved. Loss recipe is specified by protocol.training_loss, including an optional foreground-squared Dice-only Swin variant. Two-worker execution. Recovery-save cadence only changes IO; all validation milestones and final checkpoint retained.'))
    sbatch=(parent/'run.sbatch').read_text().replace('hippo-mni-caplr','hippo-mni-extend').replace('gpu:4g.40gb:1','gpu:4g.40gb:2').replace('cpus-per-task=4','cpus-per-task=8').replace('--mem=32G','--mem=48G').replace('scripts.unetr_subfields.matched_lr_job','scripts.unetr_subfields.cap_extension')
    (stage/'run.sbatch').write_text(sbatch)
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f) for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json'})
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*')) if f.is_file() and f.name not in ('manifest.json','control.json')})
    return verify(stage)


def bind(args):
    from functools import partial
    from scripts.unetr_subfields import core, baseline_runtime, geo_recipe
    from scripts.capped_depth_delay import objectives
    p=read(args.protocol_dir/'protocol.json');identity=p['model_identity'];geo=identity=='unetr_vit_b16';spec=DATASETS['MNI']
    base=core.base
    base.build_model=(lambda alias,arch:geo_recipe.build_model(4,spec['spatial_size'])) if geo else (lambda alias,arch:baseline_runtime.build_model(identity,arch))
    base.build_loss_and_optimizer=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER
    base.PreparedPatients=partial(core.PreparedSubfields,spec=spec)
    base.evaluate=partial(core.evaluate,classes=spec['classes'])
    base.learning_rate=lambda alias,step,maximum,initial:lr_at(step,p['lr_horizon_updates'],initial,identity,p['lr_policy'])
    cost=p['models'][args.model]['objectives'][args.objective]['flops_per_update']
    base.validation_steps=lambda maximum,count=20:sorted({min(maximum,max(1,target//cost)) for target in p['validation_target_flops']}|{maximum})
    if p.get('loss_recipe') in ('balanced_foreground_squared_dice','balanced_allclass_squared_dice','uniform_allclass_squared_dice'):
        from scripts.unetr_subfields import balanced_squared_dice
        allclass=p['loss_recipe'] in ('balanced_allclass_squared_dice','uniform_allclass_squared_dice')
        assert identity=='swinunetr' and p['dice_loss_arguments']=={**balanced_squared_dice.LOSS_ARGUMENTS,'include_background':allclass}
        builder=base.build_model
        base.build_model=lambda alias,arch:balanced_squared_dice.initialize_head(builder(alias,arch))
        weights=p['training_class_weights']['folds'][str(args.fold)]['weights']
        if allclass:weights=[p['background_class_weight'],*weights]
        core.trainer.training_loss=partial(balanced_squared_dice.training_loss,class_weights=weights,include_background=allclass)
    elif p.get('loss_recipe')=='log_squared_dice':
        from scripts.unetr_subfields import log_squared_dice
        assert identity=='swinunetr' and p['dice_loss_arguments']==log_squared_dice.LOSS_ARGUMENTS
        core.trainer.training_loss=log_squared_dice.training_loss
    elif p.get('loss_recipe')=='foreground_squared_dice':
        from scripts.unetr_subfields import foreground_squared_dice
        assert identity=='swinunetr' and p['dice_loss_arguments']==foreground_squared_dice.LOSS_ARGUMENTS
        core.trainer.training_loss=foreground_squared_dice.training_loss
    else:core.trainer.training_loss=geo_recipe.training_loss if geo else objectives.training_loss
    if geo:
        def check(protocol):
            assert protocol['augmentation']=='xy_d4_v1' and protocol['loss_arguments']==geo_recipe.LOSS_ARGUMENTS and protocol['dropout_rate']==.1
        core.trainer.check_no_augmentation=check
    return core.trainer


def train(args):
    trainer=bind(args);p=read(args.protocol_dir/'protocol.json')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(args.output/'runtime_recipe.json',dict(model=p['model_identity'],loss=p['training_loss'],augmentation=p['augmentation'],
             initial_lr=p['initial_lrs'][args.model],lr_horizon_updates=p['lr_horizon_updates'],lr_policy=p['lr_policy'],
             protocol_sha256=digest(args.protocol_dir/'protocol.json'),adapter_sha256=digest(Path(__file__)),
             checkpoint_interval_updates=p.get('checkpoint_interval_updates',5),
             initialization=p.get('output_head_initialization','fresh_from_fixed_seed'),primary_checkpoint='final_budget',baseline_only=True))
        if p.get('loss_recipe') in ('balanced_foreground_squared_dice','balanced_allclass_squared_dice','uniform_allclass_squared_dice'):
            write(args.output/'loss_weights.json',dict(**p['training_class_weights']['folds'][str(args.fold)],
                 classes=p['classes'][1:],formula=p['training_class_weights']['formula'],validation_used=False,
                 background_weight=p.get('background_class_weight'),
                 output_head_initialization=p['output_head_initialization']))
        trainer.run(args)


def command_for(stage,plan,task,base_path=None):
    p=read(stage/'protocols/model/protocol.json')
    return [plan['conventional_python'],'-u','-m','scripts.unetr_subfields.cap_extension','train',
            '--model',p['trainer_alias'],'--objective',task['arm'],'--fold',str(task['fold']),'--seed',str(task['seed']),
            '--pkl',str(stage/'data/MNI.pkl'),'--protocol-dir',str(stage/'protocols/model'),
            '--output',str(stage/'runs'/task['id']),'--resume']


def run_one(args):
    stage=args.stage;plan=read(stage/'plan.json');task=next(t for t in plan['tasks'] if t['id']==args.task)
    output=stage/'runs'/task['id'];output.mkdir(parents=True,exist_ok=True)
    if (output/'cleanup.json').exists():return
    if not (output/'completion.json').exists():
        with (output/'train.log').open('a') as log:
            process=subprocess.Popen(command_for(stage,plan,task),cwd=stage/'source',stdout=log,stderr=subprocess.STDOUT)
            sent=False
            while process.poll() is None:
                if stopping() and not sent:process.send_signal(signal.SIGUSR1);sent=True
                time.sleep(2)
            if process.returncode:raise SystemExit(process.returncode if process.returncode>0 else 1)
    prepare_archive(output,task)
    start=time.monotonic()
    while not (output/'archive_acknowledgment.json').exists():
        if stopping() or time.monotonic()-start>plan['archive_wait_timeout_seconds']:raise SystemExit(75)
        time.sleep(5)
    retire(output)


def run(stage):
    plan=verify(stage);job=os.environ['SLURM_JOB_ID'];install_signals()
    os.environ['HIPPO_CONTROL']=str(stage/'control.json');os.environ['HIPPO_DEADLINE']=str(time.time()+plan['graceful_stop_seconds'])
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopping():return
        controller.submit(stage,continuation=job)
        devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
        pending=[t for t in plan['tasks'] if not (stage/'runs'/t['id']/'cleanup.json').exists()]
        active={};failed=None
        try:
            while pending or active:
                for device in devices:
                    if device not in active and pending and not stopping() and not failed:
                        if quota()['free_gib']<plan['storage']['minimum_free_gib']:raise RuntimeError('Insufficient quota')
                        task=pending.pop(0)
                        child=subprocess.Popen([plan['conventional_python'],'-u','-m',RUNNER_MODULE,'run-one','--stage',str(stage),'--task',task['id']],
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':device},cwd=stage/'source')
                        active[device]=(task,child)
                for device,(task,child) in list(active.items()):
                    code=child.poll()
                    if code is not None:
                        del active[device]
                        if code not in (0,75):failed=dict(task=task,exit_code=code)
                        if code==75:os.environ['HIPPO_DEADLINE']='0'
                if stopping() or failed:
                    for task,child in active.values():
                        if child.poll() is None:child.send_signal(signal.SIGUSR1)
                    if not active:break
                controller.summarize(stage,plan,'running',[task['id'] for task,_ in active.values()])
                time.sleep(5)
            if failed:
                write(stage/'failure.json',failed);controller.summarize(stage,plan,'failed');controller.cancel_successor(stage,job);raise RuntimeError(str(failed))
            if stopping():
                paused=read(stage/'control.json')['pause'];controller.summarize(stage,plan,'paused' if paused else 'continuing_at_time_limit')
                if paused:controller.cancel_successor(stage,job)
                return
            controller.summarize(stage,plan,'complete');controller.cancel_successor(stage,job)
        except Exception:
            os.environ['HIPPO_DEADLINE']='0'
            # Signal live children before releasing the allocation; preserve their recovery artifacts.
            for task,child in active.values():
                if child.poll() is None:child.send_signal(signal.SIGUSR1)
            for task,child in active.values():child.wait()
            controller.cancel_successor(stage,job)
            write(stage/'failure.json',dict(traceback=traceback.format_exc()))
            controller.summarize(stage,plan,'failed');raise


def preflight(stage):
    plan=verify(stage)
    import torch,monai
    from scripts.unetr_subfields import core,baseline_runtime,geo_recipe
    torch.set_num_threads(4);assert monai.__version__=='1.6.0' and str(torch.__version__).startswith('2.12.1')
    p=read(stage/'protocols/model/protocol.json');profile=read(stage/'protocols/model/profile.json');identity=p['model_identity']
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');prepared=core.PreparedSubfields(frame,frame.subject_id.tolist(),DATASETS['MNI'])
    model=(geo_recipe.build_model(4,(64,64,64)) if identity=='unetr_vit_b16' else baseline_runtime.build_model(identity,read(stage/'protocols/model/nnunet_architecture.json')))
    if p.get('loss_recipe') in ('balanced_foreground_squared_dice','balanced_allclass_squared_dice','uniform_allclass_squared_dice'):
        from scripts.unetr_subfields import balanced_squared_dice
        balanced_squared_dice.initialize_head(model)
        assert all(torch.count_nonzero(v)==0 for v in model.out.parameters())
        lookup={str(row.subject_id):row for row in frame.itertuples()}
        splits=read(stage/'protocols/model/splits_10patients.json')
        for fold,split in enumerate(splits):
            weights=p['training_class_weights']['folds'][str(fold)]
            assert weights['train_cases']==split['train']
            counts=[sum(int((lookup[c].label_data==label).sum()) for c in split['train']) for label in (1,2,3)]
            assert counts==weights['counts']
            inverse=[1/n for n in counts]
            if p['loss_recipe']=='uniform_allclass_squared_dice':assert weights['weights']==[1.,1.,1.]
            else:assert all(abs(a-3*b/sum(inverse))<1e-12 for a,b in zip(weights['weights'],inverse))
    model=model.to('meta').train()
    assert sum(v.numel() for v in model.parameters())==profile['parameters']
    out=model(torch.zeros((1,1,64,64,64),device='meta'));shapes=[list(v.shape) for v in out] if isinstance(out,(tuple,list)) else [list(out.shape)]
    assert all(shape[1]==4 for shape in shapes) and shapes[0]==[1,4,64,64,64]
    horizon=p['lr_horizon_updates'];initial=p['initial_lrs'][p['trainer_alias']]
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),prepared_cases=len(prepared),
        optimizer_updates=0,model=identity,output_shapes=shapes,horizon=horizon,
        first_lr=lr_at(0,horizon,initial,identity,p['lr_policy']),last_lr=lr_at(horizon-1,horizon,initial,identity,p['lr_policy'])))
    return read(stage/'preflight.json')


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--name',required=True);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--multiplier',type=int,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--lr',type=float);p.add_argument('--lr-policy',choices=['hold_half','cap_matched','constant'],default='hold_half');p.add_argument('--evidence',default='')
    p.add_argument('--checkpoint-interval',type=int,choices=[5,25],default=5)
    p.add_argument('--swin-loss',choices=['original','foreground_squared','balanced_squared','balanced_allclass','uniform_allclass','log_squared'],default='original')
    p.add_argument('--pilot-fold',type=int,choices=[0,1,2],default=0)
    p.add_argument('--pilot-all-seeds',action='store_true')
    for command in ('verify','preflight','submit','run','run-one'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
        if command=='run-one':p.add_argument('--task',required=True)
    p=sub.add_parser('train');p.add_argument('--model',required=True);p.add_argument('--objective',required=True)
    for name in ('fold','seed'):p.add_argument('--'+name,type=int,required=True)
    for name in ('pkl','protocol-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();install_signals();controller.verify=verify;controller.command_for=command_for
    if args.command=='prepare':print(json.dumps(prepare(args)))
    elif args.command=='train':train(args)
    elif args.command=='run-one':run_one(args)
    elif args.command=='run':run(args.stage)
    elif args.command=='preflight':print(json.dumps(preflight(args.stage)))
    elif args.command=='submit':print(json.dumps(controller.submit(args.stage)))
    else:print(json.dumps(verify(args.stage)))


if __name__=='__main__':main()
