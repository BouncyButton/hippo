"""Matched MSD SwinUNETR and U-Net seed-5 constraints, three folds."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import ARMS as ALL_ARMS, CAP, read, write, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'msd_swin_unet_seed5_constraints_20261018'
MODULE = 'scripts.cross_dataset_campaign.swin_unet_seed5_constraints_job'
PARENT = 'adni_msd_four_models_seeds456_baselines_20261015'
PARENT_SHA = 'e9a168a77c60068f48618ed3bbd4cbd1cf4718a15c59752b8f03767b3805bf49'
ARMS = ('dice_bands', 'dice_bands_edge')
MODELS = ('swinunetr', 'basic_unet16')
BASE_SUBMIT = controller.submit


def tasks():
    return [dict(id=f'MSD_{model}_{arm}_f{fold}s5', kind='conventional',
                 dataset='MSD', model=model, fold=fold, seed=5, arm=arm,
                 protocol=f'MSD_{model}_f{fold}s5')
            for fold in range(3) for model in MODELS for arm in ARMS]


def recipe(original):
    result = deepcopy(original)
    result.update(name=NAME, constraints=True, requested_arm_order=list(ARMS))
    return result


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    parent = ROOT/'reports'/PARENT/'frozen'
    if digest(parent/'manifest.json') != PARENT_SHA:
        raise ValueError('Baseline source identity changed')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Baseline archive changed: '+name)
    from scripts.cross_dataset_campaign.storage import verify_archive
    baselines = []
    for model, fold in ((m, f) for m in MODELS for f in range(3)):
        folder = ROOT/'reports'/PARENT/'runs'/f'MSD_{model}_dice_f{fold}s5'
        manifest = read(folder/'archive_manifest.json')
        verify_archive(folder, manifest)
        ack = read(folder/'archive_acknowledgment.json')
        if (ack['manifest_sha256'] != digest(folder/'archive_manifest.json')
                or ack['source_manifest_sha256'] != PARENT_SHA):
            raise ValueError('Baseline archive/source binding changed')
        done = read(folder/'completion.json')
        if (done['status'], done['fold'], done['seed'], done['updates']) != ('complete', fold, 5, 206 if model == 'swinunetr' else 563):
            raise ValueError('Wrong matched baseline')
        baselines.append(dict(model=model, fold=fold, run=folder.name,
            validation_dice=read(folder/'validation_latest.json')['summary']['macro_hard_dice'],
            archive_manifest_sha256=digest(folder/'archive_manifest.json')))
    stage.mkdir(parents=True)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for module in ('swin_unet_seed5_constraints_job.py', 'seeds789_msd_parallel.py'):
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/module,
                     stage/'source/scripts/cross_dataset_campaign'/module)
    for model, fold in ((m, f) for m in MODELS for f in range(3)):
        name = f'MSD_{model}_f{fold}s5'
        shutil.copytree(parent/'protocols'/name, stage/'parent_protocols'/name)
        shutil.copytree(parent/'protocols'/name, stage/'protocols'/name)
        write(stage/'protocols'/name/'protocol.json', recipe(read(parent/'protocols'/name/'protocol.json')))
    write(stage/'baseline_results.json', baselines)
    for name in ('runtime_benchmark.json', 'runtime_benchmark_manifest.json'):
        shutil.copy2(parent/name, stage/name)
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=12, seeds=[5],
        budget_flops_by_dataset={'MSD': CAP},
        authorization='User corrected scope to MSD seed5 for both SwinUNETR and U-Net,12 runs: bands and bands+edge at original10% target, folds0/1/2. Replace queued SAM3 with this comparison; retain two workers.',
        comparison_policy='Compare all three folds to verified seed5 baseline archives; final-budget Dice primary, FN/FP and A/P swaps secondary. Preserve every result.',
        exploration_followup='Only these12 constraint runs; no additional seeds or strengths.',
        calibration_policy='Original sequential10% logit-gradient target,50% p95 ceiling; calibrate on10 training cases after50 updates, ramp over50 updates. Calibration separately reported outside the optimization-only cap.')
    plan['parallelism'].update(datasets=['MSD'], assignment='alternating_task_index')
    plan['storage']['archive'] = str(ROOT/'reports'/NAME)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    batch = (parent/'run.sbatch').read_text().replace('hippo-seeds456-baselines', 'hippo-swin-unet-seed5')
    batch = batch.replace('scripts.cross_dataset_campaign.seeds456_baselines_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen comparison changed: '+name)
    if digest(stage/'parent_manifest.json') != PARENT_SHA:
        raise ValueError('Parent provenance changed')
    parent = read(stage/'parent_manifest.json')
    for name, sha in parent.items():
        if name.startswith('source/') and digest(stage/name) != sha:
            raise ValueError('Scientific source differs from matched baselines: '+name)
    for model, fold in ((m, f) for m in MODELS for f in range(3)):
        name = f'MSD_{model}_f{fold}s5'
        original = stage/'parent_protocols'/name
        folder = stage/'protocols'/name
        for path in original.iterdir():
            if digest(path) != parent[f'protocols/{name}/{path.name}']:
                raise ValueError('Baseline protocol changed')
            if path.name != 'protocol.json' and digest(folder/path.name) != digest(path):
                raise ValueError('Cohort, profile or architecture changed')
        p = read(folder/'protocol.json')
        if p != recipe(read(original/'protocol.json')):
            raise ValueError('Unrequested scientific change')
        if (p['seeds'] != [5] or p['folds'] != [fold] or p['budget_flops'] != CAP
                or p['augmentation'] != 'none' or p['calibration_target_ratio'] != .1
                or p['calibration_p95_cap'] != .5 or p['weight_ce'] != 0):
            raise ValueError('Wrong scope or loss recipe')
        for arm in ARMS:
            s = p['models'][p['trainer_alias']]['objectives'][arm]
            if (s['max_steps'] != (206 if model == 'swinunetr' else 563 if arm == 'dice_bands' else 562) or s['prefix_updates'] != 50
                    or s['constraint_warmup_updates'] != 50 or s['budget_scope'] != 'optimization_only'
                    or s['consumed_training_flops'] > CAP or s['calibration_flops'] <= 0):
                raise ValueError('Original constraint schedule changed')
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 12
            or plan['parallelism']['workers'] != 2 or plan['parallelism']['datasets'] != ['MSD']
            or not plan['single_allocation'] or 'archive_resume_module' in plan
            or plan['storage']['minimum_free_gib'] < 20 or plan['archive_wait_timeout_seconds'] != 300):
        raise ValueError('Resource scope changed')
    return plan


def preflight(stage):
    plan = verify(stage)
    import importlib.metadata
    import sys
    import tempfile
    from unittest.mock import patch
    import torch
    from scripts.cross_dataset_campaign import conventional as exp
    from scripts.cross_dataset_campaign import msd_mixed_losses as objective
    if str(torch.__version__) != '2.12.1+cu130' or importlib.metadata.version('monai') != '1.6.0':
        raise ValueError('Matched cluster environment changed')
    torch.set_num_threads(2)
    exp.trainer.ARMS = ALL_ARMS
    with tempfile.TemporaryDirectory() as directory:
        for task in tasks():
            command = controller.command_for(stage, plan, task, None)
            command[command.index('--output')+1] = str(Path(directory)/task['id'])
            with patch.object(sys, 'argv', ['trainer', *command[3:]]), patch.object(exp.trainer, 'run') as execute:
                exp.trainer.main()
                args = execute.call_args.args[0]
                if (args.seed, args.fold, args.objective) != (5, task['fold'], task['arm']):
                    raise ValueError('Actual CLI identity differs')
    folder = stage/'protocols/MSD_swinunetr_f0s5'; p = read(folder/'protocol.json')
    if digest(p['data']) != p['pkl_sha256']:
        raise ValueError('MSD data changed')
    initializations = []
    for identity in MODELS:
        folder = stage/'protocols'/f'MSD_{identity}_f0s5'; p = read(folder/'protocol.json')
        exp.base.seed_everything(5)
        model = exp.build_model(p, read(folder/'nnunet_architecture.json'))
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
            raise ValueError('Initialization differs from seed5 baseline')
        criterion, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], model, p)
        if type(optimizer).__name__ != 'AdamW' or not criterion.include_background:
            raise ValueError('Dice/AdamW changed')
        initializations.append(dict(model=identity, sha256=h.hexdigest()))
        del optimizer, model
    labels = torch.zeros((1, 1, 64, 64, 64), dtype=torch.long)
    labels[:, :, 20:32, 20:44, 20:44] = 1
    labels[:, :, 32:44, 20:44, 20:44] = 2
    smoke = []
    for arm in ARMS:
        logits = torch.randn((1, 3, 64, 64, 64), requires_grad=True)
        row = objective.calibration_case(logits, labels, arm)
        weights, stats = exp.select_at_target([deepcopy(row) for _ in range(10)], arm, .1, .5)
        loss = objective.training_loss(logits, labels, arm, weights, 1.)
        gradient = torch.autograd.grad(loss, logits)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite constraint loss/gradient')
        if any(v <= 0 for v in weights.values()) or any(v['weighted_p95_ratio'] > .5+1e-8 for v in stats.values()):
            raise ValueError('Constraint calibration smoke failed')
        zero = objective.training_loss(logits, labels, arm, weights, 0.)
        if not torch.allclose(zero, criterion(logits, labels)):
            raise ValueError('Zero constraint scale differs from baseline Dice')
        smoke.append(dict(arm=arm, weights=weights, statistics=stats, finite_gradient=True))
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    scheduler = subprocess.run(['sbatch', '--test-only', str(stage/'run.sbatch'), str(stage)],
                               check=True, text=True, capture_output=True)
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise ValueError('Insufficient personal quota')
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        cli_commands_checked=12, initializations=initializations, optimizer_updates=0,
        quota=usage, constraint_smoke=smoke, scheduler_test=scheduler.stdout+scheduler.stderr,
        checks=['Scientific source identical to archived seed5 baselines',
                'Only constraint activation and requested arms changed in protocols',
                'Exact dataset, splits, initialization, Dice/AdamW and optimization cap',
                'Both calibrated objectives have finite gradients and preserve zero-scale baseline',
                'Two isolated GPUs, verified per-run archive before cleanup']))


def submit(stage, continuation=None):
    if continuation is not None:
        raise ValueError('Single allocation only')
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    for row in controller.queue():
        if row[0] in own:
            return read(stage/'submissions'/f'{row[0]}.json')
    if (stage/'submission.json').exists():
        raise RuntimeError('Already submitted; inspect before resubmission')
    return BASE_SUBMIT(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'preflight', 'submit', 'run', 'status'])
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()); return
    stage = args.stage.resolve()
    controller.verify = verify; controller.submit = submit
    if args.command == 'verify': print(verify(stage)['name'])
    elif args.command == 'preflight': preflight(stage); print('SwinUNETR/U-Net seed5 constraints preflight passed')
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'status': print((stage/'status.json').read_text())
    else:
        from scripts.cross_dataset_campaign import seeds789_msd_parallel as parallel
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__':
    main()
