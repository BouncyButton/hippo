"""Gated 27-run MNI baseline extension after three successful UNETR seeds."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import statistics
from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.unetr_subfields.contracts import ROOT,CAP,DATASETS,baseline_schedule,subject

NAME='mni_other_baselines_20261106'
PARENT='unetr_subfields_geo_recipe_seed_search_20261106'
IDENTITIES=('nnunet','swinunetr','basic_unet16')
SETTINGS={'nnunet':('nnunet',.01,174),'swinunetr':('swinunetr',.0001,610),'basic_unet16':('unet',.0003,1095)}


def tasks(seeds):
    return [dict(id=f'MNI_{model}_baseline_f{fold}s{seed}',kind='conventional',dataset='MNI',
        model=model,fold=fold,seed=seed,arm='dice',protocol=model)
        for seed in seeds for fold in range(3) for model in IDENTITIES]


def gate_seeds(report):
    from scripts.cross_dataset_campaign.storage import verify_archive
    from scripts.unetr_subfields.seed_search import selection
    rows=[]
    for output in (report/'runs').glob('MNI_unetr_vit_b16_geo_recipe_*'):
        if not (output/'cleanup.json').exists(): continue
        manifest=read(output/'archive_manifest.json');verify_archive(output,manifest)
        done=read(output/'completion.json');task=manifest['task']
        if done['status']!='complete' or done['updates']!=245 or done['consumed_training_flops']>CAP:
            raise ValueError('Invalid UNETR gate result')
        rows.append(dict(dataset='MNI',seed=task['seed'],fold=task['fold'],final_dice=read(output/'validation_latest.json')['summary']['macro_hard_dice']))
    result=selection(rows,['MNI'])
    if not result['target_met']: raise ValueError('Gate not met: three complete UNETR seeds with mean Dice >=0.45 required')
    return result['groups']['MNI']['qualifying_seeds']


def verify_gate(stage):
    gate=read(stage/'gate.json');seeds=gate['seeds']
    if len(seeds)!=3 or len(set(seeds))!=3 or gate['threshold']!=.45: raise ValueError('Invalid gate')
    original=read(stage/'reference_protocol/protocol.json'); splits=read(stage/'reference_protocol/splits_10patients.json')
    means={}
    for seed in seeds:
        scores=[]
        for fold in range(3):
            folder=stage/'gate_evidence'/f'MNI_unetr_vit_b16_geo_recipe_f{fold}s{seed}'
            manifest=read(folder/'archive_manifest.json')
            for name in ('completion.json','config.json','runtime_recipe.json','validation_latest.json'):
                if digest(folder/name)!=manifest['metadata'][name]['sha256']:raise ValueError('Gate evidence differs')
            done,config,recipe=[read(folder/name) for name in ('completion.json','config.json','runtime_recipe.json')]
            if done['status']!='complete' or done['updates']!=245 or done['budget_flops']!=CAP or done['consumed_training_flops']>CAP:
                raise ValueError('Gate completion differs')
            if (config['fold'],config['seed'],config['dataset_sha256'])!=(fold,seed,original['pkl_sha256']):raise ValueError('Gate identity differs')
            if config['train_cases']!=splits[fold]['train'] or config['validation_cases']!=splits[fold]['val']:raise ValueError('Gate/data splits differ')
            if recipe['initial_lr']!=.0006 or recipe['dropout_rate']!=.1 or recipe['augmentation']!='xy_d4_v1':raise ValueError('Gate is not Geo recipe')
            if read(folder/'cleanup.json')['status']!='complete':raise ValueError('Gate cleanup incomplete')
            scores.append(read(folder/'validation_latest.json')['summary']['macro_hard_dice'])
        means[str(seed)]=statistics.mean(scores)
        if means[str(seed)]<.45: raise ValueError('UNETR seed failed gate')
    return means


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen baseline extension changed: '+name)
    means=verify_gate(stage);plan=read(stage/'plan.json');seeds=read(stage/'gate.json')['seeds']
    if plan['tasks']!=tasks(seeds) or plan['num_runs']!=27 or plan['augmentation']!='none':raise ValueError('Expected baseline-only MNI extension')
    reference=read(stage/'reference_protocol/protocol.json')
    for identity in IDENTITIES:
        folder=stage/'protocols'/identity;p=read(folder/'protocol.json');profile=read(folder/'profile.json');alias,lr,horizon=SETTINGS[identity]
        if p['dataset']!='MNI' or p['classes']!=DATASETS['MNI']['classes'] or p['augmentation']!='none' or p['constraints'] or p['requested_arm_order']!=['dice']:
            raise ValueError('Unexpected baseline task')
        if p['budget_flops']!=CAP or p['seeds']!=seeds or p['folds']!=[0,1,2] or p['initial_lrs']!={alias:lr} or p['lr_horizon_updates']!=horizon:
            raise ValueError('Budget/seeds/schedule changed')
        if p['models']!={alias:dict(parameters=profile['parameters'],objectives={'dice':baseline_schedule(profile['update_flops'],10)})}:raise ValueError('Profile/schedule differs')
        if digest(folder/'profile.json')!=p['profile_sha256'] or digest(folder/'nnunet_architecture.json')!=p['nnunet_architecture_sha256']:raise ValueError('Model artifacts differ')
        if digest(stage/'data/MNI.pkl')!=p['pkl_sha256'] or p['pkl_sha256']!=reference['pkl_sha256'] or p['splits_sha256']!=reference['splits_sha256']:
            raise ValueError('MNI data or splits differ')
        splits=read(folder/'splits_10patients.json')
        if digest(folder/'splits_10patients.json')!=p['splits_sha256']:raise ValueError('Split changed')
        for fold in splits:
            train,val=fold['train'],fold['val'];a,b={subject(x) for x in train},{subject(x) for x in val}
            if len(train)!=10 or len(val)!=10 or len(a)!=5 or a&b:raise ValueError('Few-shot subject split differs')
    return plan


def command_for(stage,plan,task,base_path):
    p=read(stage/'protocols'/task['model']/'protocol.json')
    return [plan['conventional_python'],'-m','scripts.unetr_subfields.baseline_job','train',
        '--model',p['trainer_alias'],'--objective','dice','--fold',str(task['fold']),'--seed',str(task['seed']),
        '--pkl',p['data'],'--protocol-dir',str(stage/'protocols'/task['model']),
        '--output',str(stage/'runs'/task['id']),'--resume']


def prepare():
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT;report=ROOT/'reports'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    seeds=gate_seeds(report);stage.mkdir(parents=True,exist_ok=True)
    shutil.copytree(parent/'source',stage/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (stage/'data').mkdir(exist_ok=True);shutil.copy2(parent/'data/MNI.pkl',stage/'data/MNI.pkl')
    shutil.copytree(parent/'protocols/MNI',stage/'reference_protocol',dirs_exist_ok=True)
    selected=('completion.json','config.json','runtime_recipe.json','validation_latest.json','archive_manifest.json','cleanup.json')
    for seed in seeds:
        for fold in range(3):
            name=f'MNI_unetr_vit_b16_geo_recipe_f{fold}s{seed}';target=stage/'gate_evidence'/name;target.mkdir(parents=True,exist_ok=True)
            for file in selected:shutil.copy2(report/'runs'/name/file,target/file)
    write(stage/'gate.json',dict(seeds=seeds,threshold=.45,criterion='Three complete UNETR Geo-recipe seeds, mean final-budget foreground-subfield Dice across folds0/1/2 >=0.45',parent_campaign=PARENT,selection='Seeds selected through UNETR exploration and then fixed for all other models. No per-model seed selection.'))
    for identity in IDENTITIES:
        folder=stage/'protocols'/identity
        shutil.copytree(parent/'protocols/MNI',folder,dirs_exist_ok=True)
        shutil.copy2(ROOT/'reports'/NAME/'profiles'/(identity+'.json'),folder/'profile.json')
        architecture=read(ROOT/'experiments/nnunet_pair95_20261005/baseline/nnunet_architecture.json');architecture['kwargs']['num_classes']=4;write(folder/'nnunet_architecture.json',architecture)
        p=read(ROOT/'experiments/unetr_subfields_cap_20261106/protocols/MNI/protocol.json')
        alias,lr,horizon=SETTINGS[identity];profile=read(folder/'profile.json')
        p.update(name=NAME,model_identity=identity,trainer_alias=alias,seeds=seeds,folds=[0,1,2],augmentation='none',
            data=str(REMOTE/NAME/'data/MNI.pkl'),initial_lrs={alias:lr},lr_horizon_updates=horizon,
            requested_arm_order=['dice'],nnunet_sgd_momentum=.95,nnunet_warmup_fraction=0.,
            models={alias:dict(parameters=profile['parameters'],objectives={'dice':baseline_schedule(profile['update_flops'],10)})},
            profile_sha256=digest(folder/'profile.json'),nnunet_architecture_sha256=digest(folder/'nnunet_architecture.json'),
            training_loss=profile['recipe']['loss'],learning_rate_policy='Historical model-specific horizon: nnUNet polynomial; others 5% warmup then cosine.',
            baseline_only=True,initialization='fresh_random_from_declared_seed')
        write(folder/'protocol.json',p)
    for name in ('baseline_runtime.py','baseline_job.py','profile_baselines.py'):
        shutil.copy2(ROOT/'scripts/unetr_subfields'/name,stage/'source/scripts/unetr_subfields'/name)
    plan=deepcopy(read(parent/'plan.json'))
    for name in ('retirement','threshold','target_seeds','seed_limit','imported_results'):plan.pop(name,None)
    plan.update(name=NAME,tasks=tasks(seeds),num_runs=27,augmentation='none',objective='native_baseline',
        budget_flops_by_dataset={'MNI':CAP},archive_resume_module='scripts.unetr_subfields.baseline_job',
        scope='MNI only: nnUNet, SwinUNETR, BasicUNet16; folds0/1/2 and the three successful UNETR seeds; baselines only.',
        authorization='User requested extending to all other baseline models if the three MNI UNETR seeds proceed well. Gate verified from archived final-budget results.',
        storage={**plan['storage'],'archive':str(ROOT/'reports'/NAME)},submission_status='Prepared; gated, tested preflight and quota required.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False,disabled_tasks=[]))
    write(stage/'source_version.json',dict(parent_campaign=PARENT,parent_manifest_sha256=digest(parent/'manifest.json'),change='Add native MNI baseline adapters; no UNETR retraining and no constraints. Same data and folds.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('hippo-unetr-geo','hippo-mni-base').replace('scripts.unetr_subfields.geo_search run','scripts.unetr_subfields.baseline_job run'))
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f) for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json'})
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*')) if f.is_file() and f.name not in ('manifest.json','control.json')})
    return verify(stage)


def preflight(stage):
    plan=verify(stage)
    import torch,monai,pandas as pd
    from scripts.unetr_subfields import baseline_runtime
    from scripts.unetr_subfields.core import PreparedSubfields
    assert monai.__version__=='1.6.0' and str(torch.__version__).startswith('2.12.1')
    p=read(stage/'protocols/nnunet/protocol.json');frame=pd.read_pickle(p['data'],compression='gzip')
    prepared=PreparedSubfields(frame,frame.subject_id.tolist(),DATASETS['MNI']);checked=[]
    for identity in IDENTITIES:
        folder=stage/'protocols'/identity;profile=read(folder/'profile.json')
        model=baseline_runtime.build_model(identity,read(folder/'nnunet_architecture.json')).to('meta').train()
        outputs=model(torch.zeros((1,1,64,64,64),device='meta'));shapes=[list(o.shape) for o in outputs] if isinstance(outputs,(list,tuple)) else [list(outputs.shape)]
        assert sum(p.numel() for p in model.parameters())==profile['parameters'] and shapes==profile['output_shapes']
        checked.append(dict(model=identity,shapes=shapes,parameters=profile['parameters']))
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,prepared_cases=len(prepared),checks=checked,gate_means=verify_gate(stage)))
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    for command in ('verify','preflight','submit','run'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--model',choices=['nnunet','swinunetr','unet'],required=True);p.add_argument('--objective',choices=['dice'],required=True)
    for name in ('fold','seed'):p.add_argument('--'+name,type=int,required=True)
    for name in ('pkl','output','protocol-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    if args.command=='train':
        from scripts.unetr_subfields.baseline_runtime import train
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
