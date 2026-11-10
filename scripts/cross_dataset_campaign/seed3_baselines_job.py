"""Seed-3 baseline sensitivity checks, two datasets, one two-GPU allocation."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import CAP, MODELS, REMOTE, read, write, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'adni_msd_four_models_seed3_baselines_20261015'
MODULE = 'scripts.cross_dataset_campaign.seed3_baselines_job'
BUNDLES = {
    'ADNI': ('adni_colab_3fold_3seed_noaug_20261015', 'e0b87c00231bb355aa66531f6684c361a632b3f8e393f6afbfb4939a37d0caeb'),
    'MSD': ('msd_colab_3fold_3seed_noaug_20261015', '3b9d0c30f71873b6f834ae3b1fd989bf33e37bef7682fdda094b656daff3620b'),
}
DATA = {
    'ADNI': str(REMOTE/'hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'),
    'MSD': str(REMOTE/'hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl'),
}
BASE_SUBMIT = controller.submit
CLI_FILE = 'source/scripts/cross_dataset_campaign/capped_core.py'
CLI_OLD = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)'
CLI_NEW = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3), required=True)'


def enable_seed3_cli(source):
    if source.count(CLI_OLD) != 1 or CLI_NEW in source:
        raise ValueError('Unexpected original seed parser')
    return source.replace(CLI_OLD, CLI_NEW)


def scientific_source_digest(stage, name):
    if name != CLI_FILE:
        return digest(stage/name)
    source = (stage/name).read_text()
    if source.count(CLI_NEW) != 1 or CLI_OLD in source:
        raise ValueError('Seed3 CLI repair missing or ambiguous')
    return hashlib.sha256(source.replace(CLI_NEW, CLI_OLD).encode()).hexdigest()


def tasks():
    return [dict(id=f'{dataset}_{model}_dice_f{fold}s3', kind='conventional',
                 dataset=dataset, model=model, fold=fold, seed=3, arm='dice',
                 protocol=f'{dataset}_{model}_f{fold}s3')
            for fold in range(3) for model in MODELS for dataset in BUNDLES]


def parent_folder(task):
    return task['model'] + (f"_f{task['fold']}_s0" if task['dataset'] == 'ADNI' else '_s0')


def recipe(original, task, references):
    result = deepcopy(original)
    reference = next(r for r in references['models']
                     if (r['dataset'], r['model']) == (task['dataset'], task['model']))
    result.update(name=NAME, data=DATA[task['dataset']], seeds=[3], folds=[task['fold']],
                  constraints=False, requested_arm_order=['dice'], baseline_run_id=task['id'],
                  baseline_reference=dict(initialization_sha256=reference['model_sha256'], seed=3,
                      policy='Supplementary seed sensitivity; preserve all original seeds and fixed cohorts'))
    return result


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    references = read(ROOT/'reports'/NAME/'initialization_reference.json')
    stage.mkdir(parents=True)
    write(stage/'initialization_reference.json', references)
    for dataset, (bundle_name, expected) in BUNDLES.items():
        bundle = ROOT/'reports'/bundle_name/'bundle'
        if digest(bundle/'bundle_manifest.json') != expected:
            raise ValueError('Parent bundle identity changed')
        manifest = read(bundle/'bundle_manifest.json')
        for name, sha in manifest.items():
            if digest(bundle/name) != sha:
                raise ValueError('Parent bundle changed: '+name)
        parent = stage/'parents'/dataset
        parent.mkdir(parents=True)
        shutil.copy2(bundle/'bundle_manifest.json', parent/'bundle_manifest.json')
        shutil.copy2(bundle/'source/source_manifest.json', parent/'source_manifest.json')
        for name in manifest:
            if name.startswith('source/') and name != 'source/source_manifest.json':
                dest = stage/name
                if dest.exists() and digest(dest) != manifest[name]:
                    raise ValueError('Scientific sources conflict: '+name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bundle/name, dest)
        for task in tasks():
            if task['dataset'] != dataset:
                continue
            source = bundle/'protocols'/parent_folder(task)
            dest = parent/'protocols'/parent_folder(task)
            if not dest.exists():
                shutil.copytree(source, dest)
            folder = stage/'protocols'/task['protocol']
            shutil.copytree(source, folder)
            write(folder/'protocol.json', recipe(read(source/'protocol.json'), task, references))
    (stage/CLI_FILE).write_text(enable_seed3_cli((stage/CLI_FILE).read_text()))
    for name in ('seed3_baselines_job.py', 'seed3_baselines_parallel.py', 'seed3_baselines_gpu_worker.py'):
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/name, stage/'source/scripts/cross_dataset_campaign'/name)
    parent = ROOT/'reports/msd_mixed_losses_nnunet60_others10_20261014/frozen'
    for name in ('runtime_benchmark.json', 'runtime_benchmark_manifest.json'):
        shutil.copy2(parent/name, stage/name)
    plan = deepcopy(read(parent/'plan.json'))
    for name in ('base_staging', 'base_sha256', 'archive_resume_module'):
        plan.pop(name, None)
    plan.update(name=NAME, num_runs=24, tasks=tasks(), single_allocation=True,
        parallelism=dict(workers=2, datasets=['ADNI', 'MSD'], gpus_per_worker=1,
                         cpus_per_worker=4, gpu_type='4g.40gb', allocation_memory_gib=80),
        budget_flops_by_dataset={'ADNI': CAP, 'MSD': CAP},
        authorization='User confirmed 24 baseline-only seed3 runs, all four models, ADNI and MSD, folds0/1/2, two GPU workers in one Slurm job; replace pending702776.',
        comparison_policy='Keep original seeds. Seed3 is a supplementary sensitivity experiment; original splits, optimizer, losses, precision, and FLOP schedules unchanged. Cluster GPU environment is recorded.',
        exploration_followup='Only these24 baselines. No constraints or hyperparameter tuning.',
        archive_wait_timeout_seconds=300, replaced_job='702776')
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), estimated_peak_logical_gib=8,
                          mirrored_peak_gib=16, reserve_gib=4, minimum_free_gib=20)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    batch = (ROOT/'experiments/msd_unetr_seed3_baseline_bands_20261015/run.sbatch').read_text()
    batch = batch.replace('hippo-unetr-seed3-2gpu', 'hippo-seed3-baselines-2gpu')
    batch = batch.replace('scripts.cross_dataset_campaign.unetr_seed3_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
                                 if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen campaign changed: '+name)
    references = read(stage/'initialization_reference.json')
    for dataset, (_, expected) in BUNDLES.items():
        parent = stage/'parents'/dataset
        if digest(parent/'bundle_manifest.json') != expected:
            raise ValueError('Parent provenance changed')
        manifest = read(parent/'bundle_manifest.json')
        for name, sha in manifest.items():
            if name.startswith('source/'):
                path = parent/'source_manifest.json' if name == 'source/source_manifest.json' else stage/name
                actual = digest(path) if name == 'source/source_manifest.json' else scientific_source_digest(stage, name)
                if actual != sha:
                    raise ValueError('Scientific source differs: '+name)
        for task in tasks():
            if task['dataset'] != dataset:
                continue
            original = parent/'protocols'/parent_folder(task)
            folder = stage/'protocols'/task['protocol']
            for path in original.iterdir():
                if digest(path) != manifest['protocols/'+parent_folder(task)+'/'+path.name]:
                    raise ValueError('Parent protocol changed')
                if path.name != 'protocol.json' and digest(folder/path.name) != digest(path):
                    raise ValueError('Cohorts/profile/architecture changed')
            p = read(folder/'protocol.json')
            if p != recipe(read(original/'protocol.json'), task, references):
                raise ValueError('Unrequested scientific change')
            schedule = p['models'][p['trainer_alias']]['objectives']['dice']
            if (p['augmentation'] != 'none' or p['budget_flops'] != CAP
                    or schedule['consumed_training_flops'] > CAP or schedule['calibration_flops'] != 0
                    or schedule['max_steps'] != CAP//schedule['flops_per_update']):
                raise ValueError('Wrong baseline budget')
            splits = read(folder/'splits_10patients.json')
            for split in splits:
                train, val = set(split['train']), set(split['val'])
                if len(train) != 10 or len(val) != (54 if dataset == 'ADNI' else 52) or train & val:
                    raise ValueError('Incorrect frozen cohorts')
                if dataset == 'ADNI':
                    subjects = lambda cases: {v.rsplit('_', 1)[0] for v in cases}
                    if len(subjects(train)) != 5 or len(subjects(val)) != 27 or subjects(train) & subjects(val):
                        raise ValueError('ADNI subject leakage')
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 24
            or plan['augmentation'] != 'none' or plan['parallelism']['workers'] != 2
            or plan['parallelism']['datasets'] != ['ADNI', 'MSD']
            or plan.get('single_allocation') is not True or 'archive_resume_module' in plan
            or plan['archive_wait_timeout_seconds'] != 300 or plan['max_job_hours'] != 24
            or plan['storage']['minimum_free_gib'] < 20):
        raise ValueError('Wrong campaign/resource scope')
    return plan


def preflight(stage):
    verify(stage)
    import gc
    import importlib.metadata
    import torch
    from scripts.cross_dataset_campaign import conventional as exp
    if str(torch.__version__) != '2.12.1+cu130' or importlib.metadata.version('monai') != '1.6.0':
        raise ValueError('Original cluster environment changed')
    torch.set_num_threads(2)
    # Exercise the actual trainer parser: initialization alone cannot detect a
    # rejected command-line seed. No optimization or GPU allocation is needed.
    import sys
    import tempfile
    from unittest.mock import patch
    from scripts.cross_dataset_campaign.common import ARMS
    # conventional.configure installs these arms before entering the parser.
    exp.trainer.ARMS = ARMS
    with tempfile.TemporaryDirectory() as directory:
        for task in tasks():
            command = controller.command_for(stage, read(stage/'plan.json'), task, None)
            command[command.index('--output')+1] = str(Path(directory)/task['id'])
            with patch.object(sys, 'argv', ['trainer', *command[3:]]), patch.object(exp.trainer, 'run') as execute:
                exp.trainer.main()
                actual = execute.call_args.args[0]
                if (actual.seed, actual.fold, actual.objective) != (3, task['fold'], 'dice'):
                    raise ValueError('Trainer CLI changed requested identity')
    datasets, checked, models = set(), set(), []
    for task in tasks():
        key = (task['dataset'], task['model'])
        if key in checked:
            continue
        checked.add(key)
        folder = stage/'protocols'/task['protocol']; p = read(folder/'protocol.json')
        if p['data'] not in datasets:
            if digest(p['data']) != p['pkl_sha256']:
                raise ValueError('Cluster dataset differs')
            datasets.add(p['data'])
        exp.base.seed_everything(3)
        model = exp.build_model(p, read(folder/'nnunet_architecture.json'))
        h = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
            raise ValueError('Seed3 initialization changed')
        criterion, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], model, p)
        native = task['model'] == 'nnunet'
        if native:
            if type(optimizer).__name__ != 'SGD' or criterion.loss.dc.do_bg or criterion.loss.weight_ce != 1:
                raise ValueError('Native Dice+CE/SGD changed')
        elif type(optimizer).__name__ != 'AdamW' or not criterion.include_background:
            raise ValueError('Dice/AdamW changed')
        models.append(dict(dataset=task['dataset'], model=task['model'], initialization_sha256=h.hexdigest(),
                           updates=p['models'][p['trainer_alias']]['objectives']['dice']['max_steps']))
        del optimizer, criterion, model
        gc.collect()
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        optimizer_updates=0, models=models, cli_seed3_commands_checked=24,
        checks=['Scientific sources, all24 protocols and saved cohorts match original bundles',
                'Actual datasets and eight seed3 initializations verified in original environment',
                'Original baseline losses, optimizers, no augmentation and measured cap schedules',
                'Two isolated GPUs verified before training; per-run verified archive and cleanup']))


def submit(stage, continuation=None):
    if continuation is not None:
        raise ValueError('One allocation only; no automatic continuation')
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    for row in controller.queue():
        if row[0] in own:
            return read(stage/'submissions'/f'{row[0]}.json')
    return BASE_SUBMIT(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'preflight', 'submit', 'run', 'status'])
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()); return
    stage = args.stage.resolve()
    controller.verify = verify
    controller.submit = submit
    if args.command == 'verify': print(verify(stage)['name'])
    elif args.command == 'preflight': preflight(stage); print('Seed3 baseline preflight passed')
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'status': print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        from scripts.cross_dataset_campaign import seed3_baselines_parallel as parallel
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__':
    main()
