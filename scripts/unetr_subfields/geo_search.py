"""Transfer the successful Geo UNETR recipe, retaining medical data and FLOP cap."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import traceback
from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.unetr_subfields import seed_search, job, dice_ce_search
from scripts.unetr_subfields.contracts import ROOT, DATASETS, CAP, baseline_schedule

NAME = 'unetr_subfields_geo_recipe_seed_search_20261106'
PARENT = dice_ce_search.NAME
ABANDONED = 'MNI_unetr_vit_b16_dice_ce_f1s0'
POLICY = 'xy_d4_v1'
LOSS_ARGUMENTS = dict(include_background=False, to_onehot_y=True, softmax=True,
    squared_pred=True, smooth_nr=0., smooth_dr=1e-6, lambda_dice=1., lambda_ce=1.)


def tasks():
    return [dict(id=f'{d}_unetr_vit_b16_geo_recipe_f{f}s{s}', kind='conventional', dataset=d,
        model='unetr_vit_b16', fold=f, seed=s, arm='dice_ce', protocol=d, augmentation=POLICY)
        for s in range(32) for f in range(3) for d in DATASETS]


def check_recipe(protocol):
    if protocol['augmentation'] != POLICY or protocol['loss_arguments'] != LOSS_ARGUMENTS:
        raise ValueError('Expected the frozen foreground-squared Dice+CE/paired D4 recipe')
    if protocol['initial_lrs'] != {'unet':.0006} or protocol['dropout_rate'] != .1 or protocol['lr_horizon_updates'] != 477:
        raise ValueError('Learning rate, dropout or horizon differs')


def bind(protocol):
    from scripts.unetr_subfields import core, geo_recipe
    check_recipe(protocol)
    bridge = deepcopy(protocol)
    bridge['requested_arm_order'] = ['dice']
    bridge['models']['unet']['objectives'] = {'dice':protocol['models']['unet']['objectives']['dice_ce']}
    core.bind(bridge)
    spec = DATASETS[protocol['dataset']]
    core.base.build_model = lambda alias, architecture: geo_recipe.build_model(len(spec['classes']), spec['spatial_size'])
    core.trainer.training_loss = geo_recipe.training_loss
    core.trainer.check_no_augmentation = check_recipe
    core.trainer.ARMS = ('dice_ce',)
    return core.trainer


def train(args):
    trainer = bind(read(args.protocol_dir / 'protocol.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(args.output / 'runtime_recipe.json', dict(model='unetr_vit_b16', objective='dice_ce',
                recipe='GeoFWI rescue transferred to medical shapes/classes and original medical cap',
                loss_arguments=LOSS_ARGUMENTS, dropout_rate=.1, initial_lr=.0006, lr_horizon_updates=477,
                augmentation=POLICY, augmentation_scope='Training only, first two voxel axes; paired labels, no class remapping or interpolation',
                augmentation_source_sha256=digest(Path(__file__).with_name('geo_recipe.py')),
                initialization='fresh_random_from_declared_seed', primary_checkpoint='final_budget',
                protocol_sha256=digest(args.protocol_dir / 'protocol.json')))
            trainer.run(args)
    except Exception:
        write(args.output / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        if digest(stage / name) != sha: raise ValueError('Frozen Geo recipe changed: ' + name)
    plan = read(stage / 'plan.json')
    if plan['name'] != NAME or plan['tasks'] != tasks() or plan['threshold'] != .45 or plan['target_seeds'] != 3 or plan['augmentation'] != POLICY:
        raise ValueError('Search scope changed')
    for dataset, spec in DATASETS.items():
        folder = stage / 'protocols' / dataset
        p, profile = read(folder / 'protocol.json'), read(folder / 'profile.json')
        check_recipe(p)
        if p['budget_flops'] != CAP or p['seeds'] != list(range(32)) or p['folds'] != [0,1,2]:
            raise ValueError('Medical cap or folds/seeds changed')
        if p['requested_arm_order'] != ['dice_ce'] or p['models']['unet']['objectives'] != {'dice_ce':baseline_schedule(profile['update_flops'],2*spec['train_subjects'])}:
            raise ValueError('Wrong objective or profiled update limit')
        if profile['kwargs']['dropout_rate'] != .1 or profile['loss_arguments'] != LOSS_ARGUMENTS:
            raise ValueError('Wrong profiling recipe')
        if digest(stage / 'data' / f'{dataset}.pkl') != p['pkl_sha256'] or digest(folder / 'profile.json') != p['profile_sha256']:
            raise ValueError('Data or profile changed')
        original = read(stage / 'parent_protocols' / dataset / 'protocol.json')
        allowed = {'name','models','profile_sha256','training_loss','loss_source_sha256','augmentation',
                   'loss_arguments','dropout_rate','initial_lrs','learning_rate_policy','augmentation_policy'}
        if {k:v for k,v in p.items() if k not in allowed} != {k:v for k,v in original.items() if k not in allowed}:
            raise ValueError('Unrequested change outside recipe')
        if digest(folder / 'splits_10patients.json') != original['splits_sha256']:
            raise ValueError('Training support/validation split changed')
    return plan


def command_for(stage, plan, task, base_path):
    command = dice_ce_search.command_for(stage, plan, task, base_path)
    command[2] = 'scripts.unetr_subfields.geo_search'
    return command


def prepare():
    stage, parent = ROOT / 'experiments' / NAME, ROOT / 'experiments' / PARENT
    if (stage / 'manifest.json').exists(): return verify(stage)
    dice_ce_search.verify(parent)
    stage.mkdir(parents=True, exist_ok=True)
    for name in ('source','data'):
        shutil.copytree(parent / name, stage / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for dataset, spec in DATASETS.items():
        for target in ('protocols','parent_protocols'):
            shutil.copytree(parent / 'protocols' / dataset, stage / target / dataset, dirs_exist_ok=True)
        folder = stage / 'protocols' / dataset
        shutil.copy2(ROOT / 'reports' / NAME / 'profiles' / f'{dataset}.json', folder / 'profile.json')
        measured, p = read(folder / 'profile.json'), read(folder / 'protocol.json')
        p.update(name=NAME, augmentation=POLICY, loss_arguments=LOSS_ARGUMENTS, dropout_rate=.1,
            initial_lrs={'unet':.0006}, profile_sha256=digest(folder / 'profile.json'),
            training_loss='Geo recipe: foreground-only squared softmax Dice + mean CE, weights 1:1, smooth_nr=0, smooth_dr=1e-6; FP32 loss.',
            loss_source_sha256=digest(ROOT / 'scripts/unetr_subfields/geo_recipe.py'),
            learning_rate_policy='AdamW peak 6e-4, weight decay 1e-5; original 24-step warmup and 477-update cosine horizon.',
            augmentation_policy='Stateless Geo eight-way D4 on the first two padded voxel axes, paired image/mask; third axis preserved. Isotropic voxel grids; class IDs encode subfields, not laterality. No interpolation/cropping or validation augmentation.')
        p['models']['unet']['objectives'] = {'dice_ce':baseline_schedule(measured['update_flops'],2*spec['train_subjects'])}
        write(folder / 'protocol.json', p)
    for name in ('geo_recipe.py','profile_geo_recipe.py','geo_search.py','retire_interrupted.py'):
        shutil.copy2(ROOT / 'scripts/unetr_subfields' / name, stage / 'source/scripts/unetr_subfields' / name)
    trainer = stage / 'source/scripts/cross_dataset_campaign/capped_core.py'
    source = trainer.read_text()
    marker = '        active_arm = "dice" if charged < schedule["prefix_updates"] else args.objective'
    assert source.count(marker) == 1
    source = source.replace(marker, '        if protocol["augmentation"] == "xy_d4_v1":\n'
        '            from scripts.unetr_subfields.geo_recipe import augment\n'
        '            image, label = augment(image, label, args.seed, charged)\n' + marker)
    trainer.write_text(source)
    storage = stage / 'source/scripts/cross_dataset_campaign/storage.py'
    source = storage.read_text()
    marker = "config['augmentation']!='none'"
    assert source.count(marker) == 1
    source = source.replace(marker, "(config['augmentation']!='none' and not (config['augmentation']=='xy_d4_v1' and task.get('augmentation')=='xy_d4_v1' and config['model_identity']=='unetr_vit_b16'))")
    storage.write_text(source)
    plan = deepcopy(read(parent / 'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=len(tasks()), augmentation=POLICY,
        archive_resume_module='scripts.unetr_subfields.geo_search',
        scope='Fresh Geo-recipe medical UNETR seed search, seeds0..31; first3 per dataset with mean final-budget Dice >=0.45 over folds0..2.',
        authorization='User requested stopping Dice+CE sweep and trying the winning Geo recipe. Preserve medical cap, few-shot subjects, fold grouping, seed-selection target, and automatic cleanup.',
        retirement=dict(parent_stage=str(REMOTE / PARENT), original_run=ABANDONED,
            output='abandoned_'+ABANDONED, dataset='MNI', fold=1,
            reason='User stopped ordinary Dice+CE sweep and requested fresh training with the full Geo winning recipe.'),
        submission_status='Prepared; preflight, augmentation tests and quota required.')
    plan['storage']['archive'] = str(ROOT / 'reports' / NAME)
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False, disabled_tasks=[]))
    write(stage / 'source_version.json', dict(parent_campaign=PARENT,parent_manifest_sha256=digest(parent / 'manifest.json'),
        change='Foreground squared Dice+CE, dropout0.1, LR6e-4, paired D4; profiled under unchanged medical FLOP cap. Fresh weights; no imported results.'))
    (stage / 'run.sbatch').write_text((parent / 'run.sbatch').read_text().replace('hippo-unetr-dicece','hippo-unetr-geo').replace(
        'scripts.unetr_subfields.dice_ce_search run','scripts.unetr_subfields.geo_search run'))
    write(stage / 'source/source_manifest.json',{str(f.relative_to(stage / 'source')):digest(f)
        for f in sorted((stage / 'source').rglob('*')) if f.is_file() and f.name != 'source_manifest.json'})
    write(stage / 'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    return verify(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    for command in ('verify','preflight','submit','run'):
        p=sub.add_parser(command); p.add_argument('--stage',type=Path,required=True)
    p=sub.add_parser('train')
    p.add_argument('--model',choices=['unet'],required=True)
    p.add_argument('--objective',choices=['dice_ce'],required=True)
    p.add_argument('--fold',type=int,choices=[0,1,2],required=True)
    p.add_argument('--seed',type=int,required=True)
    for name in ('pkl','output','protocol-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    if args.command=='train':
        if args.seed<0 or args.threads<1 or args.smoke_steps<0: parser.error('Invalid training arguments')
        train(args); return
    if args.command=='prepare': prepare(); print(NAME); return
    controller.verify,controller.command_for,controller.summarize=verify,command_for,seed_search.summarize
    if args.command=='verify': print(verify(args.stage)['name'])
    elif args.command=='preflight':
        from scripts.unetr_subfields import core,geo_recipe
        core.build_model=geo_recipe.build_model
        job.verify=verify
        print(job.preflight(args.stage))
    elif args.command=='submit': print(json.dumps(controller.submit(args.stage)))
    else:
        from scripts.unetr_subfields.retire_interrupted import finish
        if finish(args.stage,verify(args.stage)):
            seed_search.summarize(args.stage,verify(args.stage),'starting')
            controller.run(args.stage)


if __name__=='__main__': main()
