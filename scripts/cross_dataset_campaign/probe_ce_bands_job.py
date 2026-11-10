"""Conditionally gated MSD Dice+CE+bands pilot: four models, three targets."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest, NAME, CAP
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

PROBE = 'msd_dice_ce_bands_pilot_20261014'
MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')
TARGETS = (.1, .4, .6)
ORIGINAL_COMMAND = controller.command_for
ORIGINAL_SUBMIT = controller.submit


def baseline_location(identity):
    if identity == 'nnunet':
        return 'msd_nnunet_loss_pair_90updates_20261014', 'native_dice_ce', 'MSD_nnunet_native_dice_ce_90updates_f0s0'
    return 'msd_other_models_loss_pairs_20261014', identity+'_dice_ce', f'MSD_{identity}_dice_ce_f0s0'


def make_gate(stage):
    from scripts.cross_dataset_campaign.storage import verify_archive
    from scripts.cross_dataset_campaign.report_other_loss_pairs import report
    other = report(ROOT/'reports/msd_other_models_loss_pairs_20261014')
    nn_root = ROOT/'reports/msd_nnunet_loss_pair_90updates_20261014'
    nn = read(nn_root/'comparison.json')
    if nn['status'] != 'complete' or other['status'] != 'complete':
        raise ValueError('All four completed baseline pairs are required')
    pairs = []
    for identity in MODELS:
        if identity == 'nnunet':
            values = []
            initializations = []
            for weight in (0, 1):
                label = 'native_dice_ce' if weight else 'native_dice'
                path = nn_root/'runs'/f'MSD_nnunet_{label}_90updates_f0s0'
                verify_archive(path, read(path/'archive_manifest.json'))
                done, config = read(path/'completion.json'), read(path/'config.json')
                if done['updates'] != 90 or config['augmentation'] != 'none' or config['smoke_test']:
                    raise ValueError('Invalid nnUNet pair')
                initializations.append(read(path/'initialization.json'))
                values.append(100*read(path/'validation_latest.json')['summary']['macro_hard_dice'])
            if initializations[0] != initializations[1]:
                raise ValueError('nnUNet pair initialization differs')
            pair = dict(model=identity, dice_only_validation_percent=values[0],
                        dice_ce_validation_percent=values[1], validation_gain_percentage_points=values[1]-values[0])
        else:
            pair = next(p for p in other['pairs'] if p['model'] == identity)
        pairs.append(pair)
    write(stage/'baseline_gate.json', dict(status='awaiting_user_review', criterion='User must review the final 3D UNet pair before submission',
        augmentation='none', fold=0, seed=0, pairs=pairs,
        user_authorization='Prepare 10/40/60% bands for all four Dice+CE models; UNETR decrease accepted; show final 3D UNet result before submitting',
        comparisons={'nnunet':digest(nn_root/'comparison.json'),
            'other_models':digest(ROOT/'reports/msd_other_models_loss_pairs_20261014/comparison.json')}))


def prepare():
    stage = ROOT/'experiments'/PROBE
    if (stage/'submission.json').exists():
        raise ValueError('Never modify a submitted pilot')
    stage.mkdir(parents=True, exist_ok=True)
    make_gate(stage)
    tasks = []
    for identity in MODELS:
        parent_name, reference_protocol, reference_run = baseline_location(identity)
        baseline = ROOT/'experiments'/parent_name/'protocols'/reference_protocol
        historical = ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{identity}_f0_target10'
        profile_path = ROOT/'reports'/PROBE/'profiles'/f'{identity}.json'
        profile = read(profile_path)
        if profile['status'] != 'profiled' or profile['model'] != identity or profile['calibration_base'] != 'Dice+CE':
            raise ValueError('An exact Dice+CE profile is required')
        baseline_path = ROOT/'reports'/parent_name/'runs'/reference_run
        for target in TARGETS:
            folder = stage/'protocols'/f'{identity}_target{int(target*100)}'
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('splits_10patients.json', 'nnunet_architecture.json'):
                shutil.copyfile(baseline/filename, folder/filename)
            shutil.copyfile(profile_path, folder/'profile.json')
            p = deepcopy(read(baseline/'protocol.json'))
            alias = p['trainer_alias']
            old = read(historical/'protocol.json')['models'][alias]['objectives']['dice_bands']
            s = schedule(profile['costs'], 'dice_bands', old, 10, CAP)
            p.pop('diagnostic_updates', None)
            p.pop('diagnostic', None)
            p.update(name=PROBE, weight_ce=1, augmentation='none', folds=[0], seeds=[0],
                requested_arm_order=['dice_bands'], constraints=True,
                training_loss=('Native foreground Dice+CE with unchanged deep supervision' if identity == 'nnunet'
                    else 'Existing MONAI Dice including background + mean CE, both coefficients one')+' + calibrated bands',
                dice_include_background=identity != 'nnunet',
                calibration_target_ratio=target, calibration_p95_cap=.75 if target == .6 else .5,
                calibration_base='Dice+CE', budget_flops=CAP, budget_gflops=CAP/1e9,
                validation_milestones=20, validation_target_flops=[CAP*i//20 for i in range(1, 21)],
                checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
                training_source_sha256={}, profile_sha256=digest(folder/'profile.json'),
                baseline_reference=dict(campaign=parent_name, run=reference_run,
                    protocol_sha256=digest(baseline/'protocol.json'),
                    initialization_sha256=read(baseline_path/'initialization.json')['model_state_sha256'],
                    validation_sha256=digest(baseline_path/'validation_latest.json')))
            p['models'] = {alias:dict(parameters=profile['parameters'], objectives={'dice_bands':s})}
            write(folder/'protocol.json', p)
            tasks.append(dict(id=f'MSD_{identity}_dice_ce_bands_target{int(target*100)}_f0s0',
                kind='conventional', dataset='MSD', model=identity, fold=0, seed=0,
                arm='dice_bands', target=target, protocol=folder.name))
    plan = deepcopy(read(ROOT/'experiments'/NAME/'plan.json'))
    plan.update(name=PROBE, num_runs=12, tasks=tasks, budget_flops_by_dataset={'MSD':CAP},
        exploration_followup='Exploration remains paused until the bands pilot is assessed',
        authorization='User selected 10/40/60% for all four Dice+CE models and accepted UNETR decrease; submission awaits final 3D UNet result review')
    plan['storage'].update(archive=str(ROOT/'reports'/PROBE), estimated_peak_logical_gib=5,
        mirrored_peak_gib=10, reserve_gib=2, minimum_free_gib=12)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False,
        baseline_review_approved=False, reviewed_comparisons=None))
    write(stage/'source_version.json', dict(request=plan['authorization'],
        source_of_truth='Frozen source files and SHA256 manifest include uncommitted implementation',
        parent_manifest_sha256=digest(ROOT/'experiments'/NAME/'manifest.json')))
    freeze_source(stage)
    batch = (ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text()
    batch = batch.replace('hippo-exploration-noaug', 'hippo-msd-ce-bands')
    batch = batch.replace('scripts.cross_dataset_campaign.controller run',
        'scripts.cross_dataset_campaign.probe_ce_bands_job run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json', 'control.json')})
    print(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen bands pilot changed: '+name)
    plan, gate = read(stage/'plan.json'), read(stage/'baseline_gate.json')
    if plan['name'] != PROBE or len(plan['tasks']) != 12 or plan['augmentation'] != 'none':
        raise ValueError('Unexpected bands pilot plan')
    if gate['status'] != 'awaiting_user_review' or {p['model'] for p in gate['pairs']} != set(MODELS):
        raise ValueError('Missing completed baseline evidence and review gate')
    combinations = []
    for task in plan['tasks']:
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        profile = read(stage/'protocols'/task['protocol']/'profile.json')
        s = p['models'][p['trainer_alias']]['objectives']['dice_bands']
        expected = schedule(profile['costs'], 'dice_bands', s, 10, CAP)
        if s != expected or p['budget_flops'] != CAP:
            raise ValueError('Bands schedule does not match the measured cap')
        if (p['augmentation'] != 'none' or p['weight_ce'] != 1 or not p['constraints']
                or p['calibration_base'] != 'Dice+CE' or p['calibration_target_ratio'] != task['target']
                or p['calibration_p95_cap'] != (.75 if task['target'] == .6 else .5)):
            raise ValueError('Unrequested loss, augmentation or calibration setting')
        if (task['fold'], task['seed'], task['arm']) != (0, 0, 'dice_bands'):
            raise ValueError('Only fold0 seed0 bands runs are authorized')
        combinations.append((task['model'], task['target']))
    if combinations != [(m, t) for m in MODELS for t in TARGETS]:
        raise ValueError('Expected all twelve model/target combinations')
    return plan


def submit(stage, continuation=None):
    control, gate = read(stage/'control.json'), read(stage/'baseline_gate.json')
    if not control.get('baseline_review_approved') or control.get('reviewed_comparisons') != gate['comparisons']:
        raise ValueError('User requested the final 3D UNet result before submission; review approval is required')
    return ORIGINAL_SUBMIT(stage, continuation)


def command_for(stage, plan, task, base_path):
    command = ORIGINAL_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.probe_ce_bands'
    return command


def preflight(stage):
    plan = verify(stage)
    import numpy as np
    import torch
    from scripts.cross_dataset_campaign import probe_ce_bands as objective
    from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
    from nnunet_band_objective import native_loss
    torch.set_num_threads(2)
    torch.manual_seed(0)
    axis = torch.arange(24)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    foreground = ((x-12)/7)**2+((y-12)/6)**2+((z-12)/5)**2 < 1
    labels = torch.where(foreground, torch.where(x < 12, 1, 2), 0)[None, None]
    outputs = [torch.randn(1, 3, n, n, n, requires_grad=True) for n in (24, 12, 6, 3)]
    datasets, checked = set(), set()
    for task in plan['tasks']:
        folder = stage/'protocols'/task['protocol']
        p = read(folder/'protocol.json')
        for name, key in [('splits_10patients.json','splits_sha256'),
                          ('nnunet_architecture.json','nnunet_architecture_sha256'), ('profile.json','profile_sha256')]:
            if digest(folder/name) != p[key]:
                raise ValueError('Protocol artifact changed')
        data_key = (p['data'], p['pkl_sha256'])
        if data_key not in datasets:
            if digest(p['data']) != p['pkl_sha256']:
                raise ValueError('MSD dataset changed')
            datasets.add(data_key)
        split = read(folder/'splits_10patients.json')[0]
        if len(split['train']) != 10 or len(split['val']) != 52 or set(split['train']) & set(split['val']):
            raise ValueError('Expected frozen 10/52 MSD split')
        if task['model'] in checked:
            continue
        checked.add(task['model'])
        criterion, optimizer = objective.experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], torch.nn.Linear(1, 1), p)
        native = criterion if task['model'] == 'nnunet' else None
        logits = outputs if native is not None else outputs[0]
        if native is not None:
            if criterion.loss.dc.do_bg or criterion.loss.weight_ce != 1 or criterion.loss.weight_dice != 1:
                raise ValueError('Native foreground Dice+CE changed')
            if not np.allclose(criterion.weight_factors, np.array([1., .5, .25, 0.])/1.75):
                raise ValueError('Native deep supervision changed')
            if type(optimizer).__name__ != 'SGD' or optimizer.param_groups[0]['momentum'] != .95:
                raise ValueError('Native optimizer changed')
            baseline = native_loss(logits, labels, criterion)
        else:
            if type(optimizer).__name__ != 'AdamW' or not criterion.include_background:
                raise ValueError('Original optimizer/Dice changed')
            baseline = paired_loss(logits, labels, 'dice', {}, 0., weight_ce=1)
        if optimizer.param_groups[0]['lr'] != p['initial_lrs'][p['trainer_alias']]:
            raise ValueError('Initial LR changed')
        actual = objective.training_loss(logits, labels, 'dice', {}, 0., criterion=native)
        if not torch.equal(actual, baseline):
            raise ValueError('Pilot prefix differs from its Dice+CE baseline')
        band = objective.OuterBoundaryBandLoss()(outputs[0], labels)
        combined = objective.training_loss(logits, labels, 'dice_bands', {'bands':.3}, .4, criterion=native)
        if not torch.allclose(combined-baseline, .12*band.loss, atol=1e-6):
            raise ValueError('Bands is not the sole additive loss change')
        grads = torch.autograd.grad(combined, outputs[:3] if native is not None else outputs[:1], retain_graph=True)
        if any(not torch.isfinite(g).all() for g in grads):
            raise ValueError('Nonfinite constrained gradients')
        row = objective.calibration_case(logits, labels, 'dice_bands', criterion=native)
        base_gradient, = torch.autograd.grad(baseline, outputs[0], retain_graph=True)
        if not np.isclose(row['dice_rms'], float(base_gradient.square().mean().sqrt()), rtol=1e-6):
            raise ValueError('Calibration denominator is not joint Dice+CE')
        for target in TARGETS:
            cap = .75 if target == .6 else .5
            weights, stats = objective.experiment.select_at_target([row]*10, 'dice_bands', target, cap)
            if not np.isclose(weights['bands'], min(target, cap)/row['bands_over_dice']):
                raise ValueError('Target calibration differs')
            if stats['bands']['weighted_p95_ratio'] > cap+1e-9:
                raise ValueError('Calibration ceiling exceeded')
    if len(datasets) != 1:
        raise ValueError('Pilot datasets differ')
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        checks=['All four completed baseline pairs verified; final-result review gate enforced at submission', 'Twelve unaugmented fold0 seed0 runs',
            'Exact measured FLOP schedules', 'Original Dice+CE preserved exactly in prefix',
            'Bands alone adds to baseline with finite gradients', 'Calibration denominator is joint Dice+CE',
            '10/40/60% targets with .5/.5/.75 ceilings', 'Frozen dataset/source/split identities',
            'Native deep supervision and original optimizers/LRs preserved', 'Shell syntax']))
    print('Dice+CE+bands pilot preflight passed', flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare()
        return
    controller.verify = verify
    controller.command_for = command_for
    controller.submit = submit
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
