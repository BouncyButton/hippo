"""Run two matched 90-update MSD baselines in one sequential allocation."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest, NAME, CAP
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

PROBE = 'msd_nnunet_loss_pair_90updates_20261014'
ORIGINAL_COMMAND = controller.command_for


def prepare():
    stage = ROOT/'experiments'/PROBE
    if (stage/'submission.json').exists():
        raise ValueError('Do not change a submitted pair')
    profile_path = ROOT/'reports'/PROBE/'profile.json'
    profile = read(profile_path)
    if profile['status'] != 'profiled':
        raise ValueError('Both native loss graphs must be profiled')
    parent = ROOT/'experiments'/NAME
    origin = parent/'protocols/MSD_nnunet_target10'
    tasks = []
    for weight_ce, label in ((0, 'native_dice'), (1, 'native_dice_ce')):
        folder = stage/'protocols'/label
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('splits_10patients.json', 'nnunet_architecture.json'):
            shutil.copyfile(origin/name, folder/name)
        shutil.copyfile(profile_path, folder/'profile.json')
        cost = profile['costs_by_ce_weight'][str(weight_ce)]
        p = deepcopy(read(origin/'protocol.json'))
        p.update(name=PROBE, diagnostic_updates=90, weight_ce=weight_ce,
            augmentation='none', folds=[0], seeds=[0], requested_arm_order=['dice'], constraints=False,
            training_loss=f'Native foreground Dice (weight 1) + CE (weight {weight_ce}); unchanged deep supervision',
            dice_include_background=False, diagnostic='Matched 90-update CE ablation; nine training passes',
            budget_flops=CAP, budget_gflops=CAP/1e9, validation_milestones=20,
            validation_target_flops=[(90*i//20)*cost for i in range(1, 21)],
            checkpoint_selection='Final update-90 checkpoint is primary; best validation is secondary',
            training_source_sha256={}, parent_exploration_manifest_sha256=digest(parent/'manifest.json'),
            profile_sha256=digest(folder/'profile.json'))
        p['models']['nnunet']['objectives'] = {'dice':schedule(
            {'dice':dict(update_flops=cost, calibration_flops=0)}, 'dice',
            dict(prefix_updates=0, sdf_prefix_updates=0, constraint_warmup_updates=1), 10, CAP)}
        if p['models']['nnunet']['objectives']['dice']['max_steps'] != 90:
            raise ValueError('The fixed medical cap must permit exactly 90 updates')
        write(folder/'protocol.json', p)
        tasks.append(dict(id=f'MSD_nnunet_{label}_90updates_f0s0', kind='conventional',
            dataset='MSD', model='nnunet', fold=0, seed=0, arm='dice', protocol=label))
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=PROBE, num_runs=2, tasks=tasks, diagnostic_updates=90,
        budget_flops_by_dataset={'MSD':CAP}, exploration_followup='Remain paused until these results are reviewed')
    plan['storage']['archive'] = str(ROOT/'reports'/PROBE)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        request='One job: native Dice-only and native Dice+CE, 90 updates each; no augmentation',
        causal_comparison='Only CE coefficient changes from 0 to 1; both Dice terms exclude background'))
    freeze_source(stage)
    batch = (ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text()
    batch = batch.replace('hippo-exploration-noaug', 'hippo-msd-pair90-noaug')
    batch = batch.replace('scripts.cross_dataset_campaign.controller run',
        'scripts.cross_dataset_campaign.probe_loss_pair_job run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json')})
    print(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen pair changed: '+name)
    plan = read(stage/'plan.json')
    if plan['name'] != PROBE or len(plan['tasks']) != 2 or plan['augmentation'] != 'none':
        raise ValueError('Unexpected pair plan')
    weights = []
    for task in plan['tasks']:
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        s = p['models']['nnunet']['objectives']['dice']
        if p['diagnostic_updates'] != 90 or s['max_steps'] != 90 or p['budget_flops'] != CAP:
            raise ValueError('Each baseline must use exactly 90 updates at the medical cap')
        if p['augmentation'] != 'none' or p['dice_include_background'] is not False or p['constraints']:
            raise ValueError('Native unaugmented baseline required')
        if (task['fold'], task['seed'], task['arm']) != (0, 0, 'dice'):
            raise ValueError('Pair must use fold0 seed0 without constraints')
        weights.append(p['weight_ce'])
    if weights != [0, 1]:
        raise ValueError('Expected native Dice first, then Dice+CE')
    return plan


def command_for(stage, plan, task, base_path):
    command = ORIGINAL_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.probe_loss_pair'
    return command


def preflight(stage):
    plan = verify(stage)
    import torch
    import numpy as np
    from scripts.cross_dataset_campaign.probe_loss_pair import loss_optimizer
    torch.set_num_threads(2)
    torch.manual_seed(0)
    outputs = [torch.randn(1, 3, n, n, n, requires_grad=True) for n in (8, 4, 2, 1)]
    labels = [torch.randint(0, 3, (1, 1, n, n, n)) for n in (8, 4, 2, 1)]
    criteria, protocols = [], []
    for task in plan['tasks']:
        folder = stage/'protocols'/task['protocol']
        p = read(folder/'protocol.json')
        for name, key in [('splits_10patients.json','splits_sha256'),
                          ('nnunet_architecture.json','nnunet_architecture_sha256'), ('profile.json','profile_sha256')]:
            if digest(folder/name) != p[key]:
                raise ValueError('Protocol artifact changed')
        if digest(p['data']) != p['pkl_sha256']:
            raise ValueError('MSD data changed')
        split = read(folder/'splits_10patients.json')[0]
        if len(split['train']) != 10 or len(split['val']) != 52 or set(split['train']) & set(split['val']):
            raise ValueError('Unexpected split')
        criterion, optimizer = loss_optimizer('nnunet', torch.nn.Linear(1, 1), p)
        if type(criterion.loss).__name__ != 'DC_and_CE_loss' or criterion.loss.dc.do_bg:
            raise ValueError('Expected native foreground Dice')
        if criterion.loss.weight_ce != p['weight_ce'] or criterion.loss.weight_dice != 1:
            raise ValueError('Incorrect loss coefficients')
        if not np.allclose(criterion.weight_factors, np.array([1., .5, .25, 0.])/1.75):
            raise ValueError('Deep supervision differs')
        if optimizer.param_groups[0]['lr'] != .01 or optimizer.param_groups[0]['momentum'] != .95 or p['lr_horizon_updates'] != 174:
            raise ValueError('Optimizer/schedule differs')
        loss = criterion(outputs, labels)
        gradients = torch.autograd.grad(loss, outputs[:3], retain_graph=True)
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite gradients')
        criteria.append(criterion)
        protocols.append(p)
    ce = sum(w*criteria[1].loss.ce(x, y[:, 0]) for w, x, y in zip(criteria[1].weight_factors, outputs, labels) if w)
    if not torch.allclose(criteria[1](outputs, labels)-criteria[0](outputs, labels), ce, atol=1e-6):
        raise ValueError('The objective difference is not exactly CE')
    allowed = {'weight_ce', 'training_loss', 'validation_target_flops', 'models'}
    if {k:v for k,v in protocols[0].items() if k not in allowed} != {k:v for k,v in protocols[1].items() if k not in allowed}:
        raise ValueError('A non-loss protocol setting differs')
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        checks=['Both frozen source/data/split identities', '90 updates each, no augmentation/constraints',
            'Identical native foreground Dice/deep supervision; CE alone toggled 0/1',
            'Finite backward for both arms and exact additive-CE identity',
            'Matched SGD and original 174-update LR horizon', 'Shell syntax']))
    print('Native Dice / Dice+CE 90-update pair preflight passed', flush=True)


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
