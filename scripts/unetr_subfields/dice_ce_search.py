"""Fresh Dice+CE seed search under the unchanged historical optimization cap."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import traceback

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.unetr_subfields import seed_search, job
from scripts.unetr_subfields.contracts import ROOT, DATASETS, CAP, baseline_schedule

NAME = 'unetr_subfields_dice_ce_seed_search_20261106'
PARENT = seed_search.NAME
ABANDONED = 'MNI_unetr_vit_b16_dice_f2s4'


def tasks():
    return [dict(id=f'{d}_unetr_vit_b16_dice_ce_f{f}s{s}', kind='conventional', dataset=d,
                 model='unetr_vit_b16', fold=f, seed=s, arm='dice_ce', protocol=d)
            for s in range(32) for f in range(3) for d in DATASETS]


def patch_trainer(source):
    replacements = {
        'elif args.objective == "dice":': 'elif args.objective in ("dice", "dice_ce"):',
        'args.objective != "dice" and not calibration_path.exists()': 'args.objective not in ("dice", "dice_ce") and not calibration_path.exists()',
        'scale = .2 if active_arm == "dice" else': 'scale = .2 if active_arm in ("dice", "dice_ce") else',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('Unexpected parent trainer; cannot patch baseline recognition')
        source = source.replace(before, after)
    return source


def bind(protocol):
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.ce_loss import training_loss
    if protocol['requested_arm_order'] != ['dice_ce'] or protocol['loss_weights'] != {'dice': 1., 'ce': 1.}:
        raise ValueError('Expected Dice+CE 1:1')
    bridge = deepcopy(protocol)
    bridge['requested_arm_order'] = ['dice']
    bridge['models']['unet']['objectives'] = {'dice': protocol['models']['unet']['objectives']['dice_ce']}
    core.bind(bridge)
    core.trainer.training_loss = training_loss
    core.trainer.ARMS = ('dice_ce',)
    return core.trainer


def train(args):
    trainer = bind(read(args.protocol_dir / 'protocol.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(args.output / 'runtime_recipe.json', dict(model='unetr_vit_b16', objective='dice_ce',
                lambda_dice=1., lambda_ce=1., dice_include_background=True, ce_reduction='mean',
                ce_class_weights=None, augmentation='none', initialization='random_from_declared_seed',
                primary_checkpoint='final_budget', protocol_sha256=digest(args.protocol_dir / 'protocol.json')))
            trainer.run(args)
    except Exception:
        write(args.output / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        if digest(stage / name) != sha:
            raise ValueError('Frozen Dice+CE search changed: ' + name)
    plan = read(stage / 'plan.json')
    if plan['name'] != NAME or plan['tasks'] != tasks() or plan['threshold'] != .45 or plan['target_seeds'] != 3:
        raise ValueError('Seed-selection policy changed')
    for dataset, spec in DATASETS.items():
        folder = stage / 'protocols' / dataset
        p, profile = read(folder / 'protocol.json'), read(folder / 'profile.json')
        if p['budget_flops'] != CAP or p['seeds'] != list(range(32)) or p['folds'] != [0, 1, 2]:
            raise ValueError('Budget/seeds/folds differ')
        if p['requested_arm_order'] != ['dice_ce'] or p['loss_weights'] != {'dice': 1., 'ce': 1.}:
            raise ValueError('Wrong objective')
        if p['models']['unet']['objectives'] != {'dice_ce':baseline_schedule(profile['update_flops'], 2 * spec['train_subjects'])}:
            raise ValueError('Update limit differs from Dice+CE profile')
        if digest(stage / 'data' / f'{dataset}.pkl') != p['pkl_sha256'] or digest(folder / 'profile.json') != p['profile_sha256']:
            raise ValueError('Data/profile changed')
        original = read(stage / 'parent_protocols' / dataset / 'protocol.json')
        allowed = {'name', 'requested_arm_order', 'models', 'profile_sha256', 'training_loss', 'loss_weights', 'loss_source_sha256'}
        if {k:v for k,v in p.items() if k not in allowed} != {k:v for k,v in original.items() if k not in allowed}:
            raise ValueError('Change outside objective/profile metadata')
        if digest(folder / 'splits_10patients.json') != original['splits_sha256']:
            raise ValueError('Split changed')
    return plan


def command_for(stage, plan, task, base_path):
    p = read(stage / 'protocols' / task['protocol'] / 'protocol.json')
    return [plan['conventional_python'], '-m', 'scripts.unetr_subfields.dice_ce_search', 'train',
        '--model', 'unet', '--objective', 'dice_ce', '--fold', str(task['fold']), '--seed', str(task['seed']),
        '--pkl', p['data'], '--protocol-dir', str(stage / 'protocols' / task['protocol']),
        '--output', str(stage / 'runs' / task['id']), '--resume']


def prepare():
    stage, parent = ROOT / 'experiments' / NAME, ROOT / 'experiments' / PARENT
    if (stage / 'manifest.json').exists():
        return verify(stage)
    seed_search.verify(parent)
    stage.mkdir(parents=True, exist_ok=True)
    for name in ('source', 'data'):
        shutil.copytree(parent / name, stage / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for dataset, spec in DATASETS.items():
        for target in ('protocols', 'parent_protocols'):
            shutil.copytree(parent / 'protocols' / dataset, stage / target / dataset, dirs_exist_ok=True)
        folder = stage / 'protocols' / dataset
        profile = ROOT / 'reports' / NAME / 'profiles' / f'{dataset}.json'
        shutil.copy2(profile, folder / 'profile.json')
        measured = read(profile)
        p = read(folder / 'protocol.json')
        p.update(name=NAME, requested_arm_order=['dice_ce'], loss_weights={'dice':1., 'ce':1.},
            profile_sha256=digest(folder / 'profile.json'),
            training_loss='Softmax Dice including background + unweighted mean multiclass cross-entropy, weights 1:1; historical connected-zero term retained.',
            loss_source_sha256=digest(ROOT / 'scripts/unetr_subfields/ce_loss.py'))
        p['models']['unet']['objectives'] = {'dice_ce':baseline_schedule(measured['update_flops'], 2 * spec['train_subjects'])}
        write(folder / 'protocol.json', p)
    for name in ('ce_loss.py', 'profile_dice_ce.py', 'dice_ce_search.py', 'retire_interrupted.py'):
        shutil.copy2(ROOT / 'scripts/unetr_subfields' / name, stage / 'source/scripts/unetr_subfields' / name)
    trainer = stage / 'source/scripts/cross_dataset_campaign/capped_core.py'
    trainer.write_text(patch_trainer(trainer.read_text()))
    plan = deepcopy(read(parent / 'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=len(tasks()), imported_results=[],
        objective='dice_ce', archive_resume_module='scripts.unetr_subfields.dice_ce_search',
        scope='Fresh Dice+CE 1:1 search, seeds0..31; first 3 seeds per dataset averaging >=0.45 over folds0..2.',
        authorization='User stopped Dice-only search and explicitly requested UNETR seed sweep with Dice+CE; previous cap, support sizes, seed-selection criterion and cleanup remain in force.',
        retirement=dict(parent_stage=str(REMOTE / PARENT), original_run=ABANDONED,
                        output='abandoned_' + ABANDONED, dataset='MNI', fold=2),
        submission_status='Prepared; preflight and quota required.')
    plan['storage']['archive'] = str(ROOT / 'reports' / NAME)
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False, disabled_tasks=[]))
    write(stage / 'source_version.json', dict(parent_campaign=PARENT, parent_manifest_sha256=digest(parent / 'manifest.json'),
        change='New Dice+CE loss; trainer recognizes dice_ce as an uncalibrated baseline; freshly profiled update costs. No old results/checkpoints initialize new training.'))
    (stage / 'run.sbatch').write_text((parent / 'run.sbatch').read_text().replace('hippo-unetr-seeds', 'hippo-unetr-dicece').replace(
        'scripts.unetr_subfields.seed_search run', 'scripts.unetr_subfields.dice_ce_search run'))
    write(stage / 'source/source_manifest.json', {str(f.relative_to(stage / 'source')):digest(f)
        for f in sorted((stage / 'source').rglob('*')) if f.is_file() and f.name != 'source_manifest.json'})
    write(stage / 'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json')})
    return verify(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for command in ('verify', 'preflight', 'submit', 'run'):
        p = sub.add_parser(command); p.add_argument('--stage', type=Path, required=True)
    p = sub.add_parser('train')
    p.add_argument('--model', choices=['unet'], required=True)
    p.add_argument('--objective', choices=['dice_ce'], required=True)
    p.add_argument('--fold', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--seed', type=int, required=True)
    for name in ('pkl', 'output', 'protocol-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'train':
        if args.seed < 0 or args.threads < 1 or args.smoke_steps < 0: parser.error('Invalid training arguments')
        train(args); return
    if args.command == 'prepare':
        prepare(); print(NAME); return
    controller.verify, controller.command_for, controller.summarize = verify, command_for, seed_search.summarize
    if args.command == 'verify': print(verify(args.stage)['name'])
    elif args.command == 'preflight':
        job.verify = verify; print(job.preflight(args.stage))
    elif args.command == 'submit': print(json.dumps(controller.submit(args.stage)))
    else:
        from scripts.unetr_subfields.retire_interrupted import finish
        if finish(args.stage, verify(args.stage)):
            seed_search.summarize(args.stage, verify(args.stage), 'starting')
            controller.run(args.stage)


if __name__ == '__main__':
    main()
