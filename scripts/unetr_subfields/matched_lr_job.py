"""Fresh MNI nnUNet/SwinUNETR Dice+CE runs with cap-matched LR horizons."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import math
from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.unetr_subfields.contracts import ROOT,CAP,DATASETS,baseline_schedule

NAME='mni_matched_lr_dice_ce_20261106'
PARENT='mni_other_baselines_20261106'
IDENTITIES=('nnunet','swinunetr')


def tasks():
    return [dict(id=f'MNI_{model}_dice_ce_caplr_f{fold}s{seed}',kind='conventional',dataset='MNI',
        model=model,fold=fold,seed=seed,arm='dice',protocol=model)
        for seed in range(3) for fold in range(3) for model in IDENTITIES]


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen cap-matched campaign changed: '+name)
    plan=read(stage/'plan.json')
    if plan['name']!=NAME or plan['tasks']!=tasks() or plan['num_runs']!=18 or plan['augmentation']!='none':raise ValueError('Wrong rerun scope')
    allowed={'name','data','models','lr_horizon_updates','learning_rate_policy','training_loss','profile_sha256','loss_recipe','swin_loss_arguments','lr_schedule_details'}
    for identity in IDENTITIES:
        folder=stage/'protocols'/identity;p=read(folder/'protocol.json');old=read(stage/'parent_protocols'/identity/'protocol.json');profile=read(folder/'profile.json')
        if {k:v for k,v in p.items() if k not in allowed}!={k:v for k,v in old.items() if k not in allowed}:raise ValueError('Change outside requested schedule/loss scope')
        if p['dataset']!='MNI' or p['augmentation']!='none' or p['constraints'] or p['seeds']!=[0,1,2] or p['folds']!=[0,1,2] or p['budget_flops']!=CAP or p['loss_recipe']!='dice_ce':raise ValueError('Baseline identity changed')
        schedule=baseline_schedule(profile['update_flops'],10)
        if p['models']!={identity:dict(parameters=profile['parameters'],objectives={'dice':schedule})} or p['lr_horizon_updates']!=schedule['max_steps']:raise ValueError('Horizon/update-cost mismatch')
        for name,key in [('profile.json','profile_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('splits_10patients.json','splits_sha256')]:
            if digest(folder/name)!=p[key]:raise ValueError('Protocol artifact differs: '+name)
        if digest(stage/'data/MNI.pkl')!=p['pkl_sha256']:raise ValueError('Prepared data changed')
    return plan


def command_for(stage,plan,task,base_path):
    p=read(stage/'protocols'/task['model']/'protocol.json')
    return [plan['conventional_python'],'-m','scripts.unetr_subfields.matched_lr_job','train',
        '--model',task['model'],'--objective','dice','--fold',str(task['fold']),'--seed',str(task['seed']),
        '--pkl',p['data'],'--protocol-dir',str(stage/'protocols'/task['model']),
        '--output',str(stage/'runs'/task['id']),'--resume']


def prepare():
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    from scripts.unetr_subfields.baseline_job import verify as verify_parent
    verify_parent(parent);stage.mkdir(parents=True,exist_ok=True)
    for name in ('source','data'):
        shutil.copytree(parent/name,stage/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for identity in IDENTITIES:
        for target in ('protocols','parent_protocols'):
            shutil.copytree(parent/'protocols'/identity,stage/target/identity,dirs_exist_ok=True)
        folder=stage/'protocols'/identity;shutil.copy2(ROOT/'reports'/NAME/'profiles'/(identity+'.json'),folder/'profile.json')
        p=read(folder/'protocol.json');profile=read(folder/'profile.json');schedule=baseline_schedule(profile['update_flops'],10);steps=schedule['max_steps']
        policy=(f'Native polynomial decay (power0.9), horizon={steps} updates, no warmup.' if identity=='nnunet' else
                f'Warmup for {math.ceil(.05*steps)} updates, then cosine to 1% of peak by the final update; horizon={steps}.')
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),models={identity:dict(parameters=profile['parameters'],objectives={'dice':schedule})},
            lr_horizon_updates=steps,learning_rate_policy=policy,training_loss=profile['recipe']['loss'],
            profile_sha256=digest(folder/'profile.json'),loss_recipe='dice_ce',
            lr_schedule_details=dict(horizon_updates=steps,warmup_updates=0 if identity=='nnunet' else math.ceil(.05*steps),
                decay='polynomial_power0.9' if identity=='nnunet' else 'cosine',initial_lr=p['initial_lrs'][identity],
                indexing='Zero-based optimizer update; nnUNet polynomial reaches zero at the post-budget boundary.'))
        if identity=='swinunetr':p['swin_loss_arguments']=dict(include_background=True,to_onehot_y=True,softmax=True,squared_pred=False,smooth_nr=1e-5,smooth_dr=1e-5,lambda_dice=1.,lambda_ce=1.,ce_reduction='mean')
        write(folder/'protocol.json',p)
    for name in ('matched_lr_runtime.py','matched_lr_job.py','profile_matched_lr.py'):
        shutil.copy2(ROOT/'scripts/unetr_subfields'/name,stage/'source/scripts/unetr_subfields'/name)
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=18,objective='dice_ce',archive_resume_module='scripts.unetr_subfields.matched_lr_job',
        scope='18 fresh MNI baselines: nnUNet and SwinUNETR, folds0/1/2,seeds0/1/2. LR horizons equal profiled cap updates; add CE to Swin existing Dice. No other recipe change.',
        authorization='User explicitly requested these two models, cap-matched schedules, Dice+CE for Swin, same three MNI folds/seeds.',
        storage={**plan['storage'],'archive':str(ROOT/'reports'/NAME)})
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False,disabled_tasks=[]))
    write(stage/'source_version.json',dict(parent_campaign=PARENT,parent_manifest_sha256=digest(parent/'manifest.json'),change='Cap-matched LR schedules; Swin Dice+CE; fresh weights; identical data, splits and other hyperparameters.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('hippo-mni-base','hippo-mni-caplr').replace('scripts.unetr_subfields.baseline_job run','scripts.unetr_subfields.matched_lr_job run'))
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f) for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json'})
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*')) if f.is_file() and f.name not in ('manifest.json','control.json')})
    return verify(stage)


def preflight(stage):
    verify(stage)
    import torch,monai,pandas as pd
    from scripts.unetr_subfields import matched_lr_runtime as runtime
    torch.set_num_threads(4)
    assert monai.__version__=='1.6.0' and str(torch.__version__).startswith('2.12.1')
    p=read(stage/'protocols/nnunet/protocol.json');frame=pd.read_pickle(p['data'],compression='gzip')
    prepared=runtime.core.PreparedSubfields(frame,frame.subject_id.tolist(),DATASETS['MNI']);checked=[]
    for identity in IDENTITIES:
        folder=stage/'protocols'/identity;p=read(folder/'protocol.json');profile=read(folder/'profile.json');runtime.bind(p)
        model=runtime.build_model(identity,read(folder/'nnunet_architecture.json')).to('meta').train()
        outputs=model(torch.zeros((1,1,64,64,64),device='meta'));shapes=[list(o.shape) for o in outputs] if isinstance(outputs,(tuple,list)) else [list(outputs.shape)]
        assert shapes==profile['output_shapes'] and sum(x.numel() for x in model.parameters())==profile['parameters']
        lr=lambda step:runtime.core.base.learning_rate(identity,step,p['lr_horizon_updates'],p['initial_lrs'][identity])
        checked.append(dict(model=identity,output_shapes=shapes,horizon=p['lr_horizon_updates'],first_lr=lr(0),last_update_lr=lr(p['lr_horizon_updates']-1)))
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,prepared_cases=len(prepared),checks=checked))
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('prepare')
    for command in ('verify','preflight','submit','run'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--model',choices=IDENTITIES,required=True);p.add_argument('--objective',choices=['dice'],required=True)
    for name in ('fold','seed'):p.add_argument('--'+name,type=int,required=True)
    for name in ('pkl','output','protocol-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    if args.command=='train':
        from scripts.unetr_subfields.matched_lr_runtime import train
        train(args);return
    if args.command=='prepare':prepare();print(NAME);return
    controller.verify=verify;controller.command_for=command_for
    if args.command=='verify':print(verify(args.stage)['name'])
    elif args.command=='preflight':print(preflight(args.stage))
    elif args.command=='submit':
        import json
        print(json.dumps(controller.submit(args.stage)))
    else:controller.run(args.stage)

if __name__=='__main__':main()
