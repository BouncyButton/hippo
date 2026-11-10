"""Adaptive, explicitly selected UNETR seeds; unchanged per-run training recipe."""
import argparse
from copy import deepcopy
import csv
import fcntl
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.unetr_subfields import job as original_job
from scripts.unetr_subfields.contracts import ROOT, NAME as PARENT, DATASETS, CAP, baseline_schedule, subject

NAME = 'unetr_subfields_seed_search_20261106'
SEED_LIMIT = 32
THRESHOLD = .45
TARGET = 3
ORIGINAL_SUMMARIZE = controller.summarize


def tasks():
    rows = [dict(id=f'{dataset}_unetr_vit_b16_dice_f{fold}s{seed}', kind='conventional',
                 dataset=dataset, model='unetr_vit_b16', fold=fold, seed=seed,
                 arm='dice', protocol=dataset)
            for seed in range(SEED_LIMIT) for fold in range(3) for dataset in DATASETS]
    # Finish the already charged interrupted seed before creating new artifacts.
    resumed = next(row for row in rows if row['id'] == 'COBRA_unetr_vit_b16_dice_f0s1')
    rows.remove(resumed)
    resumed['protocol'] = 'COBRA_resume'
    return [resumed] + rows


def selection(rows, active_datasets=None):
    """Only full three-fold means qualify; reject early only by a strict upper bound."""
    active_datasets = list(DATASETS) if active_datasets is None else list(active_datasets)
    if not active_datasets or len(set(active_datasets)) != len(active_datasets) or not set(active_datasets) <= set(DATASETS):
        raise ValueError("Invalid active datasets")
    groups = {}
    for dataset in DATASETS:
        seeds = []
        for seed in range(SEED_LIMIT):
            chosen = [r for r in rows if r['dataset'] == dataset and r['seed'] == seed]
            scores = {r['fold']: r['final_dice'] for r in chosen}
            if len(scores) != len(chosen) or any(f not in (0, 1, 2) or not 0 <= s <= 1 for f, s in scores.items()):
                raise ValueError('Invalid or duplicated fold result')
            if not scores:
                continue
            upper = (sum(scores.values()) + 3 - len(scores)) / 3
            mean = statistics.mean(scores.values()) if len(scores) == 3 else None
            seeds.append(dict(seed=seed, fold_dice=scores, mean_dice=mean,
                qualifies=mean is not None and mean >= THRESHOLD,
                rejected_by_upper_bound=upper < THRESHOLD, best_possible_mean=upper))
        winners = [r['seed'] for r in seeds if r['qualifies']][:TARGET]
        groups[dataset] = dict(qualifying_seeds=winners, target=TARGET,
                              target_met=len(winners) == TARGET, attempted_seeds=seeds)
    return dict(threshold=THRESHOLD, criterion='Mean final-budget foreground-subfield Dice over folds 0,1,2',
                groups=groups, active_datasets=active_datasets,
                deferred_datasets=[d for d in DATASETS if d not in active_datasets],
                target_met=all(groups[d]['target_met'] for d in active_datasets),
                inference='Validation-selected exploratory seeds; these scores are not an unbiased baseline estimate.')


def disabled_tasks(plan, result):
    disabled = []
    for task in plan['tasks']:
        group = result['groups'][task['dataset']]
        rejected = {r['seed'] for r in group['attempted_seeds'] if r['rejected_by_upper_bound']}
        if task['dataset'] not in result.get('active_datasets', DATASETS) or group['target_met'] or task['seed'] in rejected:
            disabled.append(task['id'])
    return disabled


def summarize(stage, plan, status, active=None):
    ORIGINAL_SUMMARIZE(stage, plan, status, active)
    state = read(stage / 'status.json')
    control = read(stage / 'control.json')
    result = selection(state['runs'], control.get('active_datasets'))
    control['disabled_tasks'] = disabled_tasks(plan, result)
    write(stage / 'control.json', control)
    state['seed_search'] = result
    state['search_outcome'] = ('target_met' if result['target_met'] else
                               'batch_exhausted' if status == 'complete' else 'searching')
    state['skipped_task_count'] = len(control['disabled_tasks'])
    write(stage / 'status.json', state)


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        if digest(stage / name) != sha:
            raise ValueError('Frozen search changed: ' + name)
    plan = read(stage / 'plan.json')
    if plan['name'] != NAME or plan['tasks'] != tasks() or plan['threshold'] != THRESHOLD or plan['target_seeds'] != TARGET:
        raise ValueError('Search policy differs')
    for dataset, spec in DATASETS.items():
        folder = stage / 'protocols' / dataset
        p, profile = read(folder / 'protocol.json'), read(folder / 'profile.json')
        if p['budget_flops'] != CAP or p['seeds'] != list(range(SEED_LIMIT)) or p['folds'] != [0, 1, 2]:
            raise ValueError('Budget or seed/fold space differs')
        if p['models']['unet']['objectives']['dice'] != baseline_schedule(profile['update_flops'], 2 * spec['train_subjects']):
            raise ValueError('Training schedule changed')
        if digest(stage / 'data' / f'{dataset}.pkl') != p['pkl_sha256']:
            raise ValueError('Dataset changed')
        for split in read(folder / 'splits_10patients.json'):
            if ({subject(c) for c in split['train']} & {subject(c) for c in split['val']}
                    or len(split['train']) != 2 * spec['train_subjects']):
                raise ValueError('Subject leakage or changed training support')
        old = read(stage / 'parent_protocols' / dataset / 'protocol.json')
        allowed = {'name', 'seeds', 'decision_rule', 'primary_metric'}
        if {k:v for k,v in p.items() if k not in allowed} != {k:v for k,v in old.items() if k not in allowed}:
            raise ValueError('Changes extend beyond seed-selection metadata')
    if digest(stage / 'protocols/COBRA_resume/protocol.json') != digest(stage / 'parent_protocols/COBRA/protocol.json'):
        raise ValueError('Interrupted run must retain its original protocol')
    return plan


def command_for(stage, plan, task, base_path):
    command = original_job.command_for(stage, plan, task, base_path)
    return command[:2] + ['scripts.unetr_subfields.seed_search', 'train'] + command[3:]


def train(args):
    from scripts.unetr_subfields import core
    core.bind(read(args.protocol_dir / 'protocol.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (args.output / 'runtime_recipe.json').exists():
                write(args.output / 'runtime_recipe.json', dict(model='unetr_vit_b16', objective='dice',
                    seed_search=True, augmentation='none', primary_checkpoint='final_budget',
                    protocol_sha256=digest(args.protocol_dir / 'protocol.json')))
            core.trainer.run(args)
    except Exception:
        write(args.output / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


def prepare():
    stage, parent = ROOT / 'experiments' / NAME, ROOT / 'experiments' / PARENT
    report, old_report = ROOT / 'reports' / NAME, ROOT / 'reports' / PARENT
    if (stage / 'manifest.json').exists():
        return verify(stage)
    original_job.verify(parent)
    stage.mkdir(parents=True, exist_ok=True)
    for directory in ('source', 'data', 'protocols'):
        shutil.copytree(parent / directory, stage / directory, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    shutil.copytree(parent / 'protocols', stage / 'parent_protocols', dirs_exist_ok=True)
    shutil.copytree(parent / 'protocols/COBRA', stage / 'protocols/COBRA_resume', dirs_exist_ok=True)
    shutil.copyfile(Path(__file__), stage / 'source/scripts/unetr_subfields/seed_search.py')
    for dataset in DATASETS:
        p = read(stage / 'protocols' / dataset / 'protocol.json')
        p.update(name=NAME, seeds=list(range(SEED_LIMIT)),
            decision_rule='Explore seeds in numerical order; select first 3 with mean final-budget Dice >=0.45 across folds0/1/2. Preserve all attempts. Stop each dataset when 3 qualify.',
            primary_metric='Per-volume foreground-subfield macro hard Dice; seed selection uses the mean over folds0/1/2. Validation-selected exploratory result.')
        write(stage / 'protocols' / dataset / 'protocol.json', p)
    imported = []
    for run in sorted((old_report / 'runs').iterdir()):
        if not (run / 'cleanup.json').exists() or read(run / 'cleanup.json')['status'] != 'complete':
            continue
        verify_archive(run, read(run / 'archive_manifest.json'))
        for destination in (stage / 'imported' / run.name, stage / 'runs' / run.name, report / 'runs' / run.name):
            shutil.copytree(run, destination, dirs_exist_ok=True)
        imported.append(dict(id=run.name, original_archive=str(run), manifest_sha256=digest(run / 'archive_manifest.json')))
    plan = deepcopy(read(parent / 'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=len(tasks()), threshold=THRESHOLD, target_seeds=TARGET,
        seed_limit=SEED_LIMIT, imported_results=imported,
        archive_resume_module='scripts.unetr_subfields.seed_search',
        scope='Adaptive seed search, 3 qualifying seeds per dataset, mean Dice >=0.45 over folds0/1/2; first batch seeds0..31.',
        authorization='User explicitly requested interruption and seed exploration until working seeds for both datasets, confirmed 3 seeds per dataset with >=45% mean over folds0..2, with artifact cleanup.',
        submission_status='Requires quota, empty old queue, migration audit and cluster runtime preflight.')
    plan['storage']['archive'] = str(report)
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False, disabled_tasks=[]))
    write(stage / 'source_version.json', dict(parent_campaign=PARENT,
        parent_manifest_sha256=digest(parent / 'manifest.json'),
        unchanged_training_source=True, change='Separate seed-search entry point, orchestration and selection policy only.'))
    batch = (parent / 'run.sbatch').read_text().replace('hippo-unetr-subfields', 'hippo-unetr-seeds').replace(
        'scripts.unetr_subfields.job run', 'scripts.unetr_subfields.seed_search run')
    (stage / 'run.sbatch').write_text(batch)
    write(stage / 'source/source_manifest.json', {str(f.relative_to(stage / 'source')):digest(f)
        for f in sorted((stage / 'source').rglob('*')) if f.is_file() and f.name != 'source_manifest.json'})
    write(stage / 'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json') and 'runs' not in f.relative_to(stage).parts})
    return verify(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for command in ('verify', 'preflight', 'submit', 'run'):
        p = sub.add_parser(command)
        p.add_argument('--stage', type=Path, required=True)
    p = sub.add_parser('train')
    p.add_argument('--model', choices=['unet'], required=True)
    p.add_argument('--objective', choices=['dice'], required=True)
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
        if args.seed < 0 or args.threads < 1 or args.smoke_steps < 0:
            parser.error('Invalid seed, threads or smoke steps')
        train(args)
        return
    if args.command == 'prepare':
        print(json.dumps(prepare(), indent=2))
        return
    controller.verify, controller.command_for, controller.summarize = verify, command_for, summarize
    if args.command == 'verify':
        print(verify(args.stage)['name'])
    elif args.command == 'preflight':
        original_job.verify = verify
        print(original_job.preflight(args.stage))
    elif args.command == 'submit':
        print(json.dumps(controller.submit(args.stage)))
    else:
        summarize(args.stage, verify(args.stage), 'starting')
        controller.run(args.stage)


if __name__ == '__main__':
    main()
