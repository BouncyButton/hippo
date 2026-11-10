"""Eight target60 MNI pilots, after retiring explicitly abandoned study runs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE,install_signals,stopping
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import quota
from scripts.mni_constraints import campaign as original
from scripts.unetr_subfields import cap_extension as batch

ROOT=original.ROOT
NAME='mni_bands_edge_target60_f0s0_20261107'
MODULE='scripts.mni_constraints.strong_pilot'
PARENT=original.NAME
ARMS=('dice_bands','dice_bands_edge')


def tasks():
    return [dict(id=f'MNI_{model}_{arm}_f0s0',kind='conventional',dataset='MNI',model=model,
        arm=arm,fold=0,seed=0,protocol=model+'_'+arm,
        augmentation='xy_d4_v1' if model=='unetr_vit_b16' else 'none') for arm in ARMS for model in batch.MODELS]


def prepare():
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    stage.mkdir(parents=True,exist_ok=True)
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('strong_pilot.py','launch_strong.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    for task in tasks():
        folder=stage/'protocols'/task['protocol']
        shutil.copytree(parent/'protocols'/task['protocol'],folder,dirs_exist_ok=True)
        p=read(folder/'protocol.json')
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),folds=[0],seeds=[0],
            calibration_target_ratio=.6,calibration_p95_cap=.75,
            calibration_strength_policy='Historical stronger sequential calibration:60% median gradient ratio with75% p95 limit. Recalibrate against actual winning base loss; do not multiply stored lambdas blindly.',
            parent_protocol_sha256=digest(parent/'protocols'/task['protocol']/'protocol.json'))
        write(folder/'protocol.json',p)
    inventory=read(ROOT/'reports'/PARENT/'stop_inventory.json')
    retirements=[]
    for entry in inventory['retirements']:
        assert not entry['completed'], 'Completed outputs must use the original completed-run archive path'
        task=entry['task'];folder=stage/'retirement_protocols'/task['protocol']
        shutil.copytree(parent/'protocols'/task['protocol'],folder,dirs_exist_ok=True)
        retirements.append(dict(id='abandoned_'+task['id'],original_run=task['id'],protocol=task['protocol'],
            model=task['model'],fold=task['fold'],seed=task['seed'],files=entry['files']))
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=8,logical_conditions=list(ARMS),logical_cells=12,
        reused_baselines=4,folds=[0],seeds=[0],calibration_target_ratio=.6,
        archive_resume_module=MODULE,retirements=retirements,parent_stage=str(REMOTE/PARENT),
        authorization='User explicitly abandoned14-condition study and requested stronger bands and bands+edge pilots for four models at one fold/seed, twoGPU workers; evaluate/archive/clean interrupted runs first.',
        storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/NAME)))
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'baseline_reuse.json',[r for r in read(parent/'baseline_reuse.json') if r['fold']==r['seed']==0])
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Only calibration target/cap changed from10%/50% to60%/75%; restricted to bands,bands+edge atfold0seed0. Training objective, schedules, data and base recipes retained; abandoned evaluation has zero optimizer updates.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('hippo-mni-constraints','hippo-mni-target60').replace(original.MODULE,MODULE))
    original.freeze(stage)
    return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');assert plan['tasks']==tasks() and plan['num_runs']==8
    assert plan['gpu_workers']==2 and plan['excluded_nodes']==['gnode04']
    assert len(read(stage/'baseline_reuse.json'))==4
    for task in tasks():
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        assert p['folds']==p['seeds']==[0] and p['calibration_target_ratio']==.6 and p['calibration_p95_cap']==.75
        assert p['budget_flops']==original.CAP and p['requested_arm_order']==[task['arm']]
        s=p['models'][p['trainer_alias']]['objectives'][task['arm']]
        assert p['lr_horizon_updates']==s['max_steps']
        assert original.training_flops(s['max_steps'],s)<=original.CAP<original.training_flops(s['max_steps']+1,s)
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    return plan


def preflight(stage):
    plan=verify(stage)
    import torch,monai
    from scripts.mni_constraints import objectives as o
    from scripts.unetr_subfields import baseline_runtime,geo_recipe,core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');rows=[]
    for identity in batch.MODELS:
        folder=stage/'protocols'/(identity+'_dice_bands_edge');p=read(folder/'protocol.json')
        cases=read(folder/'splits_10patients.json')[0]['train']
        data=core.PreparedSubfields(frame,cases,DATASETS['MNI']);label=data[0]['label'][None]
        torch.manual_seed(0)
        model=(geo_recipe.build_model(4,(64,64,64)) if identity=='unetr_vit_b16' else baseline_runtime.build_model(identity,read(folder/'nnunet_architecture.json'))).to('meta').train()
        outputs=model(torch.zeros((1,1,64,64,64),device='meta'))
        shapes=[list(v.shape) for v in outputs] if isinstance(outputs,(tuple,list)) else [list(outputs.shape)]
        assert shapes==read(folder/'profile.json')['output_shapes']
        assert sum(v.numel() for v in model.parameters())==p['models'][p['trainer_alias']]['parameters']
        del model,outputs
        criterion=None
        if identity=='nnunet':criterion,_=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER('nnunet',torch.nn.Linear(1,1),p)
        outputs=[torch.randn(shape,requires_grad=True) for shape in shapes]
        if identity!='nnunet':outputs=outputs[0]
        z=outputs[0] if isinstance(outputs,list) else outputs
        for arm in ARMS:
            row=o.calibration_case(outputs,label,arm,criterion=criterion,identity=identity)
            weak,_=o.select_weights([deepcopy(row)],arm,target=.1,cap=.5)
            strong,stats=o.select_weights([deepcopy(row)],arm,target=.6,cap=.75)
            assert strong['bands']>weak['bands']
            assert all(v['weighted_median_ratio']<=.60000001 and v['weighted_p95_ratio']<=.75000001 for v in stats.values())
            loss=o.training_loss(outputs,label,arm,strong,1.,criterion=criterion,identity=identity)
            gradient=torch.autograd.grad(loss,z)[0]
            assert torch.isfinite(loss) and torch.isfinite(gradient).all()
            rows.append(dict(model=identity,arm=arm,weak_weights=weak,strong_weights=strong,statistics=stats))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=rows,
        optimizer_updates=0,torch=torch.__version__,monai=monai.__version__)
    write(stage/'preflight.json',result);return result


def evaluate_abandoned(stage,identifier):
    import torch
    from types import SimpleNamespace
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.unetr_subfields.mni_diagnostic import archive_and_retire
    plan=read(stage/'plan.json');item=next(r for r in plan['retirements'] if r['id']==identifier)
    output=stage/'runs'/identifier
    if (output/'cleanup.json').exists():return
    if not (output/'archive_manifest.json').exists():
        torch.set_num_threads(4);device=torch.device('cuda');assert torch.cuda.is_bf16_supported()
        folder=stage/'retirement_protocols'/item['protocol'];p=read(folder/'protocol.json');cfg=read(output/'config.json')
        assert digest(folder/'protocol.json')==cfg['protocol_sha256']
        for name,info in item['files'].items():assert digest(output/name)==info['sha256'],name
        cp=torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        ledger=read(output/'budget_ledger.json');assert cp['configuration']==cfg
        assert cp['charged_attempts']<=ledger['charged_attempts']
        args=SimpleNamespace(protocol_dir=folder,model=p['trainer_alias'],objective=cfg['objective'],fold=cfg['fold'])
        batch.bind(args)
        model=core.base.build_model(p['trainer_alias'],read(folder/'nnunet_architecture.json')).to(device)
        model.load_state_dict(cp['model'])
        frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');spec=DATASETS['MNI']
        assert digest(stage/'data/MNI.pkl')==cfg['dataset_sha256']
        for name,cases in [('training_interrupted',cfg['train_cases']),('validation_interrupted',cfg['validation_cases'])]:
            result=core.evaluate(model,core.PreparedSubfields(frame,cases,spec),device,True,spec['classes'])
            assert {r['case_name'] for r in result['cases']}==set(cases)
            assert all('confusion' in r and 'foreground_fn' in r for r in result['cases'])
            write(output/(name+'.json'),result)
        best_path=output/'checkpoint_best.pt'
        if best_path.exists():
            best=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(best['model'])
            write(output/'validation_best_poststop.json',core.evaluate(model,core.PreparedSubfields(frame,cfg['validation_cases'],spec),device,True,spec['classes']))
            del best
        write(output/'interrupted_protocol.json',p)
        write(output/'abandonment.json',dict(status='abandoned_by_user',original_run=item['original_run'],
            original_parent=plan['parent_stage'],original_source_manifest_sha256=read(stage/'source_version.json')['parent_manifest_sha256'],
            completed_updates=cp['step'],checkpoint_charged_updates=cp['charged_attempts'],charged_updates=ledger['charged_attempts'],
            consumed_training_flops=ledger['training_flops'],additional_optimizer_updates=0,
            best_checkpoint_available=best_path.exists(),excluded_from_complete_results=True,ap_swaps_applicable=False,
            reason='User stopped14-condition study and requested stronger bands/bands+edge pilots; evaluate final available checkpoint before artifact deletion.'))
        del model,cp;torch.cuda.empty_cache()
    archive_and_retire(output,dict(id=identifier,kind='abandoned',dataset='MNI',model=item['model']))


def command_for(stage,plan,task,base_path=None):
    command=original.command_for(stage,plan,task,base_path)
    command[command.index(original.MODULE)]=MODULE
    return command


def run(stage):
    plan=verify(stage);install_signals()
    os.environ['HIPPO_CONTROL']=str(stage/'control.json')
    os.environ['HIPPO_DEADLINE']=str(time.time()+plan['graceful_stop_seconds'])
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
    pending=[r for r in plan['retirements'] if not (stage/'runs'/r['id']/'cleanup.json').exists()]
    assert len(pending)<=2
    write(stage/'status.json',dict(status='retiring_previous_study',completed=0,total=8,active=[r['id'] for r in pending]))
    children=[]
    for device,item in zip(devices,pending):
        children.append(subprocess.Popen([plan['conventional_python'],'-u','-m',MODULE,'retire','--stage',str(stage),'--task',item['id']],
            cwd=stage/'source',env={**os.environ,'CUDA_VISIBLE_DEVICES':device}))
    codes=[child.wait() for child in children]
    if any(codes):
        write(stage/'status.json',dict(status='failed',phase='retirement',exit_codes=codes));raise RuntimeError(str(codes))
    write(stage/'retirement_complete.json',dict(status='complete',runs=[r['id'] for r in plan['retirements']],
        released_bytes=sum(read(stage/'runs'/r['id']/'cleanup.json')['released_file_bytes'] for r in plan['retirements']),quota=quota()))
    batch.run(stage)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    for command in ('verify','preflight','submit','run','run-one','retire'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
        if command in ('run-one','retire'):p.add_argument('--task',required=True)
    p=sub.add_parser('train');p.add_argument('--model',required=True);p.add_argument('--objective',required=True)
    for key in ('fold','seed'):p.add_argument('--'+key,type=int,required=True)
    for key in ('pkl','protocol-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();install_signals()
    controller.verify=verify;controller.command_for=command_for
    batch.verify=verify;batch.command_for=command_for;batch.RUNNER_MODULE=MODULE
    if args.command=='prepare':print(json.dumps({k:v for k,v in prepare().items() if k not in ('tasks','retirements')}))
    elif args.command=='verify':print(json.dumps(dict(status='passed',num_runs=verify(args.stage)['num_runs'])))
    elif args.command=='preflight':print(json.dumps(preflight(args.stage)))
    elif args.command=='submit':print(json.dumps(controller.submit(args.stage)))
    elif args.command=='run':run(args.stage)
    elif args.command=='run-one':batch.run_one(args)
    elif args.command=='retire':evaluate_abandoned(args.stage,args.task)
    else:original.train(args)


if __name__=='__main__':main()
