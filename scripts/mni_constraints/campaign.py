"""Submit and run the finite MNI 14-condition matrix with two GPU workers."""
import argparse
import fcntl
from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import ARMS, read, write, digest, REMOTE, install_signals
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.capped_depth_delay.schedule import training_flops
from scripts.unetr_subfields import cap_extension as batch
from scripts.unetr_subfields.contracts import ROOT, DATASETS, subject

NAME = 'mni_constraints14_cap8_20261107'
MODULE = 'scripts.mni_constraints.campaign'
CAP = 255811605336720
RECIPES = ROOT/'reports/mni_baseline_rescue_20261106/final_recipes.json'


def arm_schedule(costs, arm, timing):
    adjusted=deepcopy(costs)
    # These per-term Python coefficient products are outside tensor dispatch.
    # The shared schedule already charges depth's product and SDF product/ramp.
    for name,cost in adjusted.items():
        cost['update_flops']+=int('bands' in name)+int('edge' in name)
    return schedule(adjusted,arm,timing,10,CAP)


def tasks():
    # Exercise every model's complete objective first; then traverse fixed arms.
    arms = [ARMS[-1], *ARMS[1:-1]]
    return [dict(id=f'MNI_{model}_{arm}_f{fold}s{seed}',kind='conventional',dataset='MNI',
                 model=model,arm=arm,fold=fold,seed=seed,protocol=model+'_'+arm,
                 augmentation='xy_d4_v1' if model=='unetr_vit_b16' else 'none')
            for arm in arms for seed in range(3) for fold in range(3) for model in batch.MODELS]


def freeze(stage):
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f)
        for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json' and '__pycache__' not in f.parts})
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json') and '__pycache__' not in f.parts})


def prepare(target=.1):
    stage=ROOT/'experiments'/NAME
    if (stage/'manifest.json').exists():return verify(stage)
    stage.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'experiments/mni_unetr_geo_cap8_20261106'
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('cap_extension.py','log_squared_dice.py'):
        shutil.copy2(ROOT/'scripts/unetr_subfields'/filename,stage/'source/scripts/unetr_subfields'/filename)
    shutil.copytree(ROOT/'scripts/mni_constraints',stage/'source/scripts/mni_constraints',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    trainer=stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    source=trainer.read_text()
    marker='        original_affine = np.asarray(protocol[\'native_geometry\'][batch[\'case_name\']][\'model_affine\'])'
    replacement=marker+"\n        native_depth = geometry_weights(label, original_affine, protocol['depth_geometry']) if 'depth' in args.objective else None"
    assert source.count(marker)==1;source=source.replace(marker,replacement)
    marker='            image, label = augment(image, label, args.seed, charged)'
    replacement=marker+"\n            if native_depth is not None:\n                from scripts.mni_constraints.objectives import transform_depth\n                native_depth = transform_depth(native_depth, args.seed, charged)"
    assert source.count(marker)==1;source=source.replace(marker,replacement)
    marker="depth = geometry_weights(label, current_affine, protocol['depth_geometry']).to(device) if 'depth' in active_arm else None"
    assert source.count(marker)==1;source=source.replace(marker,"depth = native_depth.to(device) if 'depth' in active_arm else None")
    # Calibration metadata must describe the actual base gradient, including CE.
    source=source.replace('"target_ratio": .1, "p95_cap": .5,', '"target_ratio": protocol["calibration_target_ratio"], "p95_cap": protocol["calibration_p95_cap"],')
    source=source.replace('deep-supervised Dice gradient','native deep-supervised Dice+CE gradient')
    trainer.write_text(source)
    settings=read(ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json')['geometry']
    recipes=read(RECIPES);reuse=[]
    for identity,recipe in recipes.items():
        origin=ROOT/'experiments'/recipe['stage']/'protocols/model'
        profile=read(ROOT/'reports'/NAME/'profiles'/f'{identity}.json')
        assert profile['objective_sha256']==digest(ROOT/'scripts/mni_constraints/objectives.py')
        parent_protocol=recipe['protocol'];alias=parent_protocol['trainer_alias']
        baseline_steps=CAP//profile['costs']['dice']['update_flops']
        ramp=max(1,math.ceil(.1*baseline_steps))
        timing=dict(prefix_updates=ramp,constraint_warmup_updates=ramp,sdf_prefix_updates=ramp+1)
        for arm in ARMS[1:]:
            folder=stage/'protocols'/(identity+'_'+arm);folder.mkdir(parents=True,exist_ok=True)
            for name in ('splits_10patients.json','nnunet_architecture.json'):
                shutil.copy2(origin/name,folder/name)
            write(folder/'profile.json',profile)
            p=deepcopy(parent_protocol);s=arm_schedule(profile['costs'],arm,timing)
            p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),constraints=list(arm.split('_')[1:]),
                budget_origin='Winning shared MNI cap:8 times the historical90-update nnUNet cap,255811605336720 optimization FLOPs.',
                requested_arm_order=[arm],models={alias:dict(parameters=profile['parameters'],objectives={arm:s})},
                lr_horizon_updates=s['max_steps'],constraint_warmup_updates=ramp,
                calibration_target_ratio=target,calibration_p95_cap=.75 if target==.6 else .5,
                depth_geometry=settings,profile_sha256=digest(folder/'profile.json'),
                constraint_recipe='Original balanced two-step union bands; signed face consistency; three subfield SDFs; union-geometry weighted CE. Sequential gradient calibration against the actual winning base loss.',
                geometry_coordinate_convention='Source array coordinates/voxel units from identity source affines; anatomical RAS/mm orientation is not verified. Native depth weights are permuted with Geo D4, never refitted in rotated array coordinates.',
                timing_policy='Baseline-only first10% of baseline updates (ceil);10% ramp. SDF delayed to nearest5/9 of shared optimization FLOPs, with its own10% ramp. Calibration separately reported outside optimization cap.',
                learning_rate_policy='Same winning peak, warmup and decay family, horizon matched to this arm maximum_updates.',
                lr_schedule_details=dict(policy=p['lr_policy'],horizon_updates=s['max_steps'],
                    warmup_updates=0 if identity=='nnunet' else 24 if identity=='unetr_vit_b16' else math.ceil(.05*s['max_steps']),
                    decay_start_index=s['max_steps']//2 if p['lr_policy']=='hold_half' else 24 if identity=='unetr_vit_b16' else math.ceil(.05*s['max_steps'])),
                base_protocol_sha256=digest(origin/'protocol.json'),base_stage=recipe['stage'])
            write(folder/'protocol.json',p)
        for seed in range(3):
            for fold in range(3):
                folder=ROOT/'reports'/recipe['stage']/'runs'/f'MNI_{identity}_f{fold}s{seed}'
                archive=read(folder/'archive_manifest.json');verify_archive(folder,archive)
                assert read(folder/'cleanup.json')['status']=='complete'
                assert read(folder/'completion.json')['status']=='complete'
                reuse.append(dict(model=identity,arm='dice',fold=fold,seed=seed,archive=str(folder),
                    archive_manifest_sha256=digest(folder/'archive_manifest.json'),
                    completion_sha256=digest(folder/'completion.json'),
                    final_dice=read(folder/'validation_latest.json')['summary']['macro_hard_dice'],
                    base_loss=parent_protocol['training_loss'],legacy_arm='dice_ce' if identity=='unetr_vit_b16' else 'dice'))
    write(stage/'baseline_reuse.json',reuse)
    plan=dict(name=NAME,tasks=tasks(),budget_flops=CAP,logical_conditions=list(ARMS),logical_cells=504,
        reused_baselines=36,num_runs=468,models=list(batch.MODELS),folds=[0,1,2],seeds=[0,1,2],
        augmentation='model_specific',calibration_target_ratio=target,excluded_nodes=['gnode04'],
        conventional_python='/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python',
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,gpu_workers=2,
        storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/NAME)),
        authorization='Submit14-condition constraint study on the winning MNI four-model recipes and fixed3folds×3seeds; reuse36 completed baselines; two GPU workers with one running and one queued job maximum; archive and retire each completed run.',
        selection_caveat='Same development-validation folds used to establish baseline recipes; exploratory comparisons, not untouched test estimates.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        final_recipes_sha256=digest(RECIPES),changes='Four-class auxiliary objectives and actual-base calibration; transform native depth weights under D4; per-arm FLOP/LR schedules; verified baseline reuse.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('hippo-mni-extend','hippo-mni-constraints').replace('scripts.unetr_subfields.cap_extension',MODULE).replace('#SBATCH --nodes=1','#SBATCH --nodes=1\n#SBATCH --exclude=gnode04'))
    freeze(stage)
    return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen artifact changed: '+name)
    plan=read(stage/'plan.json');assert plan['tasks']==tasks() and len(plan['tasks'])==468
    assert len({t['id'] for t in plan['tasks']})==468 and plan['budget_flops']==CAP
    assert len(read(stage/'baseline_reuse.json'))==36
    for folder in (stage/'protocols').iterdir():
        p=read(folder/'protocol.json');arm=p['requested_arm_order'][0];alias=p['trainer_alias']
        s=p['models'][alias]['objectives'][arm]
        assert p['budget_flops']==CAP and p['lr_horizon_updates']==s['max_steps']
        assert training_flops(s['max_steps'],s)<=CAP<training_flops(s['max_steps']+1,s)
        assert p['folds']==p['seeds']==[0,1,2] and p['classes']==DATASETS['MNI']['classes']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        for split in read(folder/'splits_10patients.json'):
            assert len(split['train'])==len(split['val'])==10
            assert len({subject(c) for c in split['train']})==5
            assert not {subject(c) for c in split['train']}&{subject(c) for c in split['val']}
    return plan


def train(args):
    from scripts.mni_constraints import objectives
    trainer=batch.bind(args);p=read(args.protocol_dir/'protocol.json');identity=p['model_identity']
    trainer.training_loss=partial(objectives.training_loss,identity=identity)
    trainer.calibration_case=partial(objectives.calibration_case,identity=identity)
    trainer.select_weights=partial(objectives.select_weights,target=p['calibration_target_ratio'],cap=p['calibration_p95_cap'])
    trainer.geometry_weights=objectives.geometry_weights
    trainer.signed_distance_maps=objectives.signed_distance_maps
    s=p['models'][args.model]['objectives'][args.objective]
    trainer.base.validation_steps=lambda maximum,count=20: sorted({next((n for n in range(1,maximum+1) if training_flops(n,s)>=target),maximum) for target in p['validation_target_flops']}|{maximum})
    write(args.output/'runtime_recipe.json',dict(model=identity,base_loss=p['training_loss'],constraints=p['constraints'],
        augmentation=p['augmentation'],protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        objective_sha256=digest(Path(objectives.__file__)),initialization='fresh_from_fixed_seed',
        lr_horizon_updates=p['lr_horizon_updates'],primary_checkpoint='final_budget',baseline_only=False))
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trainer.run(args)


def command_for(stage,plan,task,base_path=None):
    folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
    return [plan['conventional_python'],'-u','-m',MODULE,'train','--model',p['trainer_alias'],
        '--objective',task['arm'],'--fold',str(task['fold']),'--seed',str(task['seed']),
        '--pkl',str(stage/'data/MNI.pkl'),'--protocol-dir',str(folder),'--output',str(stage/'runs'/task['id']),'--resume']


def preflight(stage):
    verify(stage)
    from scripts.mni_constraints.verify import check_stage
    result=check_stage(stage)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),**result))
    return read(stage/'preflight.json')


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--target',type=float,choices=[.1,.6],default=.1)
    for command in ('verify','preflight','submit','run','run-one'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
        if command=='run-one':p.add_argument('--task',required=True)
    p=sub.add_parser('train');p.add_argument('--model',required=True);p.add_argument('--objective',required=True)
    for name in ('fold','seed'):p.add_argument('--'+name,type=int,required=True)
    for name in ('pkl','protocol-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();install_signals()
    controller.verify=verify;controller.command_for=command_for
    batch.verify=verify;batch.command_for=command_for;batch.RUNNER_MODULE=MODULE
    if args.command=='prepare':print(json.dumps(prepare(args.target)))
    elif args.command=='train':train(args)
    elif args.command=='run-one':batch.run_one(args)
    elif args.command=='run':batch.run(args.stage)
    elif args.command=='preflight':print(json.dumps(preflight(args.stage)))
    elif args.command=='submit':print(json.dumps(controller.submit(args.stage)))
    else:print(json.dumps(verify(args.stage)))


if __name__=='__main__':main()
