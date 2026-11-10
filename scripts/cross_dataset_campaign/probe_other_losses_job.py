"""Six unaugmented MSD runs: three models, Dice with and without CE."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest, NAME, CAP
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

PROBE = 'msd_other_models_loss_pairs_20261014'
MODELS = ('swinunetr', 'unetr_vit_b16', 'basic_unet16')
ORIGINAL_COMMAND = controller.command_for


def prepare():
    stage = ROOT/'experiments'/PROBE
    if (stage/'submission.json').exists():
        raise ValueError('Do not change a submitted comparison')
    tasks = []
    for identity in MODELS:
        profile_path = ROOT/'reports'/PROBE/'profiles'/f'{identity}.json'
        profile = read(profile_path)
        if profile['status'] != 'profiled' or profile['model'] != identity:
            raise ValueError('Each model/loss graph must be profiled')
        origin = ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{identity}_f0_target10'
        for weight_ce, label in ((0, 'dice'), (1, 'dice_ce')):
            folder = stage/'protocols'/f'{identity}_{label}'
            folder.mkdir(parents=True, exist_ok=True)
            for name in ('splits_10patients.json', 'nnunet_architecture.json'):
                shutil.copyfile(origin/name, folder/name)
            shutil.copyfile(profile_path, folder/'profile.json')
            cost = profile['costs_by_ce_weight'][str(weight_ce)]
            p = deepcopy(read(origin/'protocol.json'))
            alias = p['trainer_alias']
            s = schedule({'dice':dict(update_flops=cost, calibration_flops=0)}, 'dice',
                dict(prefix_updates=0, sdf_prefix_updates=0, constraint_warmup_updates=1), 10, CAP)
            p.update(name=PROBE, diagnostic_updates=s['max_steps'], weight_ce=weight_ce,
                augmentation='none', folds=[0], seeds=[0], requested_arm_order=['dice'], constraints=False,
                training_loss=f'Existing MONAI Dice (weight 1, background included) + mean CE (weight {weight_ce})',
                dice_include_background=True, diagnostic='Only CE added; original model-specific optimizer and LR schedule',
                budget_flops=CAP, budget_gflops=CAP/1e9, validation_milestones=20,
                validation_target_flops=[CAP*i//20 for i in range(1, 21)],
                checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
                training_source_sha256={}, profile_sha256=digest(folder/'profile.json'),
                baseline_reference_protocol_sha256=digest(origin/'protocol.json'))
            p['models'] = {alias:dict(parameters=profile['parameters'], objectives={'dice':s})}
            write(folder/'protocol.json', p)
            tasks.append(dict(id=f'MSD_{identity}_{label}_f0s0', kind='conventional',
                dataset='MSD', model=identity, fold=0, seed=0, arm='dice', protocol=folder.name))
    parent = ROOT/'experiments'/NAME
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=PROBE, num_runs=6, tasks=tasks, budget_flops_by_dataset={'MSD':CAP},
        exploration_followup='Remain paused until these comparisons are reviewed')
    plan['storage'].update(archive=str(ROOT/'reports'/PROBE), estimated_peak_logical_gib=5,
        mirrored_peak_gib=10, reserve_gib=2, minimum_free_gib=12)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        request='Other three MSD baselines with Dice-only and Dice+CE at the admitted medical FLOP cap, unaugmented',
        causal_comparison='Within each model preserve the original MONAI Dice and AdamW; toggle CE coefficient 0/1'))
    freeze_source(stage)
    batch = (ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text()
    batch = batch.replace('hippo-exploration-noaug', 'hippo-msd-three-pairs')
    batch = batch.replace('scripts.cross_dataset_campaign.controller run',
        'scripts.cross_dataset_campaign.probe_other_losses_job run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json')})
    print(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen comparison changed: '+name)
    plan = read(stage/'plan.json')
    if plan['name'] != PROBE or len(plan['tasks']) != 6 or plan['augmentation'] != 'none':
        raise ValueError('Unexpected comparison plan')
    combinations = []
    for task in plan['tasks']:
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        cost = read(stage/'protocols'/task['protocol']/'profile.json')['costs_by_ce_weight'][str(p['weight_ce'])]
        s = p['models'][p['trainer_alias']]['objectives']['dice']
        if p['budget_flops'] != CAP or s['max_steps'] != CAP//cost or p['diagnostic_updates'] != s['max_steps']:
            raise ValueError('Run must consume the admitted medical cap')
        if p['augmentation'] != 'none' or p['dice_include_background'] is not True or p['constraints']:
            raise ValueError('Unchanged Dice, no augmentation or constraints required')
        if (task['fold'], task['seed'], task['arm']) != (0, 0, 'dice'):
            raise ValueError('Expected fold0 seed0 baselines')
        combinations.append((task['model'], p['weight_ce']))
    if combinations != [(m, w) for m in MODELS for w in (0, 1)]:
        raise ValueError('Expected exactly three complete pairs')
    return plan


def command_for(stage, plan, task, base_path):
    command = ORIGINAL_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.probe_other_losses'
    return command


def preflight(stage):
    plan = verify(stage)
    import torch
    from scripts.cross_dataset_campaign.probe_other_losses import paired_loss, experiment, original_loss
    torch.set_num_threads(2)
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    labels = torch.randint(0, 3, (1, 1, 8, 8, 8))
    losses = [paired_loss(logits, labels, 'dice', {}, .2, weight_ce=w) for w in (0, 1)]
    original = original_loss(logits, labels, 'dice', {}, .2)
    if not torch.equal(losses[0], original):
        raise ValueError('Dice-only baseline changed')
    ce = torch.nn.functional.cross_entropy(logits.float(), labels[:, 0].long())
    if not torch.allclose(losses[1]-losses[0], ce, atol=1e-6):
        raise ValueError('Loss difference is not exactly CE')
    for loss in losses:
        gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite loss/gradients')
    datasets = set()
    protocols = {}
    for task in plan['tasks']:
        folder = stage/'protocols'/task['protocol']
        p = read(folder/'protocol.json')
        for name, key in [('splits_10patients.json','splits_sha256'),
                          ('nnunet_architecture.json','nnunet_architecture_sha256'), ('profile.json','profile_sha256')]:
            if digest(folder/name) != p[key]:
                raise ValueError('Protocol artifact changed')
        key = (p['data'], p['pkl_sha256'])
        if key not in datasets:
            if digest(p['data']) != p['pkl_sha256']:
                raise ValueError('MSD data changed')
            datasets.add(key)
        split = read(folder/'splits_10patients.json')[0]
        if len(split['train']) != 10 or len(split['val']) != 52 or set(split['train']) & set(split['val']):
            raise ValueError('Unexpected split')
        criterion, optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], torch.nn.Linear(1, 1), p)
        if type(optimizer).__name__ != 'AdamW' or not criterion.include_background:
            raise ValueError('Historical AdamW/Dice recipe differs')
        if optimizer.param_groups[0]['lr'] != p['initial_lrs'][p['trainer_alias']]:
            raise ValueError('Learning rate differs')
        protocols[(task['model'], p['weight_ce'])] = p
    for identity in MODELS:
        a, b = [protocols[(identity, w)] for w in (0, 1)]
        allowed = {'weight_ce', 'training_loss', 'models', 'diagnostic_updates'}
        if {k:v for k,v in a.items() if k not in allowed} != {k:v for k,v in b.items() if k not in allowed}:
            raise ValueError('A non-loss protocol setting differs')
    if len({p['splits_sha256'] for p in protocols.values()}) != 1 or len(datasets) != 1:
        raise ValueError('All runs must use the identical MSD split/data')
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        checks=['All six frozen source/data/split identities', 'Measured loss-specific FLOP costs and admitted cap',
            'Original Dice-only preserved exactly; difference is mean CE only',
            'Finite backward both arms', 'Original AdamW and model-specific LR schedule',
            'All six unaugmented, unconstrained, fold0 seed0 on the same 10/52 split', 'Shell syntax']))
    print('Three-model Dice / Dice+CE comparison preflight passed', flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare()
        return
    controller.verify = verify
    controller.command_for = command_for
    if len(sys.argv) > 1 and sys.argv[1] == 'preflight':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--stage', type=Path, required=True)
        preflight(parser.parse_args().stage)
        return
    controller.main()


if __name__ == '__main__':
    main()
