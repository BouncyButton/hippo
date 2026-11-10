"""One two-GPU batch for seeds 4–6, four baselines, two datasets and three folds."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import ARMS, CAP, MODELS, read, write, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'adni_msd_four_models_seeds456_baselines_20261015'
MODULE = 'scripts.cross_dataset_campaign.seeds456_baselines_job'
PARENT = 'adni_msd_four_models_seed3_baselines_20261015'
PARENT_SHA = 'b64b56e4699470de4b6e20ef06f3519fd06026e1c549e3c798ee604bebce8ec4'
SEEDS = (4, 5, 6)
BUNDLES = ('ADNI', 'MSD')
CLI_FILE = 'source/scripts/cross_dataset_campaign/capped_core.py'
CLI_OLD = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3), required=True)'
CLI_NEW = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4, 5, 6), required=True)'
BASE_SUBMIT = controller.submit


def tasks():
    return [dict(id=f'{dataset}_{model}_dice_f{fold}s{seed}', kind='conventional',
                 dataset=dataset, model=model, fold=fold, seed=seed, arm='dice',
                 protocol=f'{dataset}_{model}_f{fold}s{seed}')
            for seed in SEEDS for fold in range(3) for model in MODELS for dataset in BUNDLES]


def parent_folder(task):
    return f"{task['dataset']}_{task['model']}_f{task['fold']}s3"


def recipe(original, task, references):
    result = deepcopy(original)
    matches = [r for r in references['models']
               if (r['dataset'], r['model'], r['seed']) == (task['dataset'], task['model'], task['seed'])]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate initialization reference')
    result.update(name=NAME, seeds=[task['seed']], baseline_run_id=task['id'],
                  baseline_reference=dict(initialization_sha256=matches[0]['model_sha256'],
                    seed=task['seed'], policy='Supplementary seed sensitivity; preserve all seeds and fixed cohorts'))
    return result


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    parent = ROOT/'reports'/PARENT/'frozen'
    if digest(parent/'manifest.json') != PARENT_SHA:
        raise ValueError('Completed parent source identity changed')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Parent archive changed: '+name)
    references = read(ROOT/'reports'/NAME/'initialization_reference.json')
    stage.mkdir(parents=True)
    write(stage/'initialization_reference.json', references)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = (stage/CLI_FILE).read_text()
    if source.count(CLI_OLD) != 1 or CLI_NEW in source:
        raise ValueError('Unexpected seed parser')
    (stage/CLI_FILE).write_text(source.replace(CLI_OLD, CLI_NEW))
    shutil.copy2(ROOT/'scripts/cross_dataset_campaign/seeds456_baselines_job.py',
                 stage/'source/scripts/cross_dataset_campaign/seeds456_baselines_job.py')
    for task in tasks():
        original = parent/'protocols'/parent_folder(task)
        saved = stage/'parent_protocols'/parent_folder(task)
        if not saved.exists():
            shutil.copytree(original, saved)
        folder = stage/'protocols'/task['protocol']
        shutil.copytree(original, folder)
        write(folder/'protocol.json', recipe(read(original/'protocol.json'), task, references))
    for name in ('runtime_benchmark.json', 'runtime_benchmark_manifest.json'):
        shutil.copy2(parent/name, stage/name)
    plan = deepcopy(read(parent/'plan.json'))
    plan.pop('replaced_job', None)
    plan.pop('calibration_policy', None)
    plan.update(name=NAME, tasks=tasks(), num_runs=72, seeds=list(SEEDS),
        authorization='User confirmed one72-run batch: four models, ADNI/MSD, folds0/1/2, seeds4/5/6; supersedes the unsubmitted ADNI U-Net-only seed4 plan. Retain two GPU workers.',
        comparison_policy='Run all72 fixed baseline conditions. Compare complete per-seed results; preserve original seeds and all failures. No validation-based stopping or replacement of poor seeds.',
        exploration_followup='Only seeds4/5/6; no additional seeds or constraint arms are authorized by this batch.')
    plan['storage']['archive'] = str(ROOT/'reports'/NAME)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    batch = (parent/'run.sbatch').read_text().replace('hippo-seed3-baselines-2gpu', 'hippo-seeds456-baselines')
    batch = batch.replace('scripts.cross_dataset_campaign.seed3_baselines_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen batch changed: '+name)
    if digest(stage/'parent_manifest.json') != PARENT_SHA:
        raise ValueError('Parent provenance changed')
    parent = read(stage/'parent_manifest.json')
    for name, sha in parent.items():
        if not name.startswith('source/'):
            continue
        if name == CLI_FILE:
            source = (stage/name).read_text()
            if source.count(CLI_NEW) != 1 or CLI_OLD in source:
                raise ValueError('Requested seed CLI missing')
            actual = hashlib.sha256(source.replace(CLI_NEW, CLI_OLD).encode()).hexdigest()
        else:
            actual = digest(stage/name)
        if actual != sha:
            raise ValueError('Scientific or worker source changed: '+name)
    references = read(stage/'initialization_reference.json')
    expected_refs = {(d, m, s) for d in BUNDLES for m in MODELS for s in SEEDS}
    actual_refs = {(r['dataset'], r['model'], r['seed']) for r in references['models']}
    if actual_refs != expected_refs or len(references['models']) != 24 or references['optimizer_updates'] != 0:
        raise ValueError('Wrong initialization coverage')
    for task in tasks():
        original = stage/'parent_protocols'/parent_folder(task)
        folder = stage/'protocols'/task['protocol']
        for path in original.iterdir():
            if digest(path) != parent['protocols/'+parent_folder(task)+'/'+path.name]:
                raise ValueError('Parent protocol changed')
            if path.name != 'protocol.json' and digest(folder/path.name) != digest(path):
                raise ValueError('Cohort/profile/architecture changed')
        p = read(folder/'protocol.json')
        if p != recipe(read(original/'protocol.json'), task, references):
            raise ValueError('Unrequested scientific change')
        schedule = p['models'][p['trainer_alias']]['objectives']['dice']
        if (p['augmentation'] != 'none' or p['budget_flops'] != CAP or p['constraints']
                or p['requested_arm_order'] != ['dice'] or schedule['calibration_flops'] != 0
                or schedule['max_steps'] != CAP//schedule['flops_per_update']):
            raise ValueError('Baseline loss/budget contract changed')
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 72
            or plan['seeds'] != list(SEEDS) or plan['augmentation'] != 'none'
            or plan['parallelism']['workers'] != 2 or plan['parallelism']['datasets'] != list(BUNDLES)
            or not plan.get('single_allocation') or 'archive_resume_module' in plan
            or plan['storage']['minimum_free_gib'] < 20 or plan['archive_wait_timeout_seconds'] != 300):
        raise ValueError('Wrong batch/resource scope')
    return plan


def preflight(stage):
    plan = verify(stage)
    import gc
    import importlib.metadata
    import sys
    import tempfile
    from unittest.mock import patch
    import torch
    from scripts.cross_dataset_campaign import conventional as exp
    if str(torch.__version__) != '2.12.1+cu130' or importlib.metadata.version('monai') != '1.6.0':
        raise ValueError('Original cluster environment changed')
    torch.set_num_threads(2)
    exp.trainer.ARMS = ARMS  # Installed by conventional.configure before the real CLI.
    with tempfile.TemporaryDirectory() as directory:
        for task in tasks():
            command = controller.command_for(stage, plan, task, None)
            command[command.index('--output')+1] = str(Path(directory)/task['id'])
            with patch.object(sys, 'argv', ['trainer', *command[3:]]), patch.object(exp.trainer, 'run') as execute:
                exp.trainer.main()
                actual = execute.call_args.args[0]
                if (actual.seed, actual.fold, actual.objective) != (task['seed'], task['fold'], 'dice'):
                    raise ValueError('Actual CLI changed requested identity')
    datasets, checked, models = set(), set(), []
    for task in tasks():
        key = (task['dataset'], task['model'], task['seed'])
        if key in checked:
            continue
        checked.add(key)
        folder = stage/'protocols'/task['protocol']; p = read(folder/'protocol.json')
        if p['data'] not in datasets:
            if digest(p['data']) != p['pkl_sha256']:
                raise ValueError('Actual cluster dataset changed')
            datasets.add(p['data'])
        exp.base.seed_everything(task['seed'])
        model = exp.build_model(p, read(folder/'nnunet_architecture.json'))
        h = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
            raise ValueError('Initialization differs from frozen reference')
        criterion, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], model, p)
        if task['model'] == 'nnunet':
            if type(optimizer).__name__ != 'SGD' or criterion.loss.dc.do_bg or criterion.loss.weight_ce != 1:
                raise ValueError('Original Dice+CE/SGD changed')
        elif type(optimizer).__name__ != 'AdamW' or not criterion.include_background:
            raise ValueError('Original Dice/AdamW changed')
        models.append(dict(dataset=task['dataset'], model=task['model'], seed=task['seed'],
                           initialization_sha256=h.hexdigest(), updates=p['models'][p['trainer_alias']]['objectives']['dice']['max_steps']))
        del optimizer, criterion, model
        gc.collect()
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    smoke = subprocess.run(['sbatch', '--test-only', str(stage/'run.sbatch'), str(stage)],
                           check=True, text=True, capture_output=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise ValueError('Insufficient personal quota')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        optimizer_updates=0, models=models, cli_commands_checked=72, quota=usage,
        scheduler_test=smoke.stdout+smoke.stderr,
        checks=['All72 protocols match seed3 apart from seed and run identity',
                'All scientific and worker source identical except accepted CLI seed range',
                'Two actual datasets and24 initializations verified in original environment',
                'Original losses, optimizers, no augmentation and FLOP schedules',
                'Two isolated GPU workers and verified per-run archival/cleanup']))


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
    elif args.command == 'preflight': preflight(stage); print('Seeds4/5/6 preflight passed')
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'status': print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        from scripts.cross_dataset_campaign import seed3_baselines_parallel as parallel
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__':
    main()
