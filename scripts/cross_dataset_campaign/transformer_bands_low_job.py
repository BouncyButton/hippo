"""Four user-authorized MSD transformer bands runs at 10% and 20%."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import CAP, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, schedule

NAME = 'msd_transformer_ce_bands_target10_20_20261014'
PARENT = 'msd_four_models_ce_target60_20261014'
MODULE = 'scripts.cross_dataset_campaign.transformer_bands_low_job'
MODELS = ('swinunetr', 'unetr_vit_b16')
TARGETS = (.1, .2)
ORIGINAL_COMMAND = controller.command_for


def prepare():
    from scripts.cross_dataset_campaign.storage import verify_archive
    parent = ROOT/'reports'/PARENT/'frozen'
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        raise ValueError('Do not overwrite an existing campaign')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Parent frozen source changed: '+name)
    stage.mkdir(parents=True)
    # Preserve every pre-existing implementation file from the actual 60% run.
    shutil.copytree(parent/'source', stage/'source')
    for module in ('transformer_bands_low.py', 'transformer_bands_low_job.py'):
        shutil.copyfile(ROOT/'scripts/cross_dataset_campaign'/module,
                        stage/'source/scripts/cross_dataset_campaign'/module)
    tasks = []
    for target in TARGETS:
        for model in MODELS:
            reference = ROOT/'reports'/PARENT/'runs'/f'MSD_{model}_ce_dice_bands_target60_f0s0'
            baseline = ROOT/'reports'/PARENT/'runs'/f'MSD_{model}_ce_dice_target60_f0s0'
            for run in (reference, baseline):
                verify_archive(run, read(run/'archive_manifest.json'))
            folder = stage/'protocols'/f'{model}_target{int(target*100)}'
            shutil.copytree(parent/'protocols'/model, folder)
            p = read(folder/'protocol.json')
            p.update(name=NAME, requested_arm_order=['dice_bands'], calibration_target_ratio=target,
                calibration_p95_cap=.75,
                matched_60_percent_reference=dict(campaign=PARENT, run=reference.name,
                    archive_manifest_sha256=digest(reference/'archive_manifest.json'),
                    calibration_anchor_sha256=read(reference/'calibration.json')['anchor_model_sha256']),
                baseline_reference=dict(campaign=PARENT, run=baseline.name,
                    initialization_sha256=read(baseline/'initialization.json')['model_state_sha256'],
                    validation_sha256=digest(baseline/'validation_latest.json'),
                    validation_dice=read(baseline/'validation_latest.json')['summary']['macro_hard_dice']))
            p['models'][p['trainer_alias']]['objectives'] = {'dice_bands':p['models'][p['trainer_alias']]['objectives']['dice_bands']}
            write(folder/'protocol.json', p)
            tasks.append(dict(id=f'MSD_{model}_ce_dice_bands_target{int(target*100)}_f0s0',
                kind='conventional', dataset='MSD', model=model, fold=0, seed=0,
                arm='dice_bands', target=target, protocol=folder.name))
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME, num_runs=4, tasks=tasks, archive_resume_module=MODULE,
        authorization='User explicitly requested SwinUNETR and UNETR Dice+CE+bands at 10% and 20%, immediately replacing the abandoned 56-run MSD sweep. No augmentation.',
        exploration_followup='All other experiments remain held; never resume the abandoned MSD56 sweep',
        comparison_policy='Only target changes from matching 60% bands. Same 75% p95 ceiling; fresh seed0 initialization, prefix, sampling, loss, optimizer and FLOP schedule.')
    plan['storage']['archive'] = str(ROOT/'reports'/NAME)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    write(stage/'source_version.json', dict(parent_campaign=PARENT,
        parent_manifest_sha256=digest(parent/'manifest.json'),
        scientific_source='Exact archived parent source, plus two new entry points; loss/calibration/model/trainer code unchanged',
        measured_flops='Reuse exact parent per-objective profile; varying a scalar calibration target does not change the training graph'))
    batch = (parent/'run.sbatch').read_text().replace('hippo-msd-four-ce60','hippo-transformer-bands-low').replace(
        'scripts.cross_dataset_campaign.msd_ce_constraints_job run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    print(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['augmentation'] != 'none' or plan['num_runs'] != 4
            or [(t['model'],t['target']) for t in plan['tasks']] != [(m,t) for t in TARGETS for m in MODELS]):
        raise ValueError('Only four requested transformer experiments are authorized')
    for task in plan['tasks']:
        folder = stage/'protocols'/task['protocol']; p = read(folder/'protocol.json')
        for filename, key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
            if digest(folder/filename) != p[key]:
                raise ValueError('Protocol identity changed: '+filename)
        profile = read(folder/'profile.json')
        s = p['models'][p['trainer_alias']]['objectives']['dice_bands']
        if (s != schedule(profile['costs'],'dice_bands',s,10,CAP)
                or profile['objective_sha256'] != digest(stage/'source/scripts/cross_dataset_campaign/msd_ce_constraints.py')):
            raise ValueError('Exact loss graph profile or schedule changed')
        if ((task['fold'],task['seed'],task['arm']) != (0,0,'dice_bands')
                or p['augmentation'] != 'none' or p['weight_ce'] != 1
                or p['dataset'] != 'MSD' or p['model_identity'] != task['model']
                or p['calibration_base'] != 'Dice+CE' or p['budget_flops'] != CAP
                or p['calibration_target_ratio'] != task['target'] or p['calibration_p95_cap'] != .75
                or s['max_steps'] != (206 if task['model']=='swinunetr' else 245)):
            raise ValueError('Unauthorized experiment setting')
        split = read(folder/'splits_10patients.json')[0]
        if len(split['train']) != 10 or len(split['val']) != 52 or set(split['train']) & set(split['val']):
            raise ValueError('Expected original 10/52 split')
    return plan


def command_for(stage, plan, task, base_path):
    command = ORIGINAL_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.transformer_bands_low'
    return command


def preflight(stage):
    import torch
    from scripts.cross_dataset_campaign import msd_ce_constraints as obj
    from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
    plan = verify(stage)
    torch.set_num_threads(2); torch.manual_seed(0)
    x,y,z = torch.meshgrid(*(torch.arange(64),)*3,indexing='ij')
    fg = ((x-32)/18)**2+((y-32)/12)**2+((z-32)/10)**2 < 1
    labels = torch.where(fg,torch.where(x<32,1,2),0)[None,None]
    outputs = torch.randn(1,3,64,64,64,requires_grad=True)
    baseline = paired_loss(outputs,labels,'dice',{},0.,weight_ce=1)
    assert torch.equal(baseline,obj.training_loss(outputs,labels,'dice',{},0.))
    band = obj.original.OuterBoundaryBandLoss()(outputs,labels).loss
    base_grad = torch.autograd.grad(baseline,outputs,retain_graph=True)[0]
    band_grad = torch.autograd.grad(band,outputs,retain_graph=True)[0]
    row = obj.calibration_case(outputs,labels,'dice_bands')
    data = set()
    for task in plan['tasks']:
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        data.add((p['data'],p['pkl_sha256']))
        criterion, optimizer = obj.experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],torch.nn.Linear(1,1),p)
        assert criterion.include_background and type(optimizer).__name__=='AdamW'
        weights, stats = obj.experiment.select_at_target([deepcopy(row) for _ in range(10)],'dice_bands',task['target'],.75)
        ratio = float((weights['bands']*band_grad).square().mean().sqrt()/base_grad.square().mean().sqrt())
        assert abs(ratio-task['target']) < 1e-6
        assert stats['bands']['weighted_p95_ratio'] <= .75
        loss = obj.training_loss(outputs,labels,'dice_bands',weights,1.)
        assert torch.equal(loss,baseline+weights['bands']*band) or torch.allclose(loss,baseline+weights['bands']*band,atol=1e-6)
        assert torch.isfinite(torch.autograd.grad(loss,outputs,retain_graph=True)[0]).all()
        assert torch.equal(obj.training_loss(outputs,labels,'dice_bands',weights,0.),baseline)
    assert len(data)==1
    for path, sha in data:
        assert digest(path)==sha, 'Cluster dataset changed'
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json', dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),
        checks=['Four exact model/target combinations, fold0 seed0, no augmentation',
            'Original 10-training/52-validation split and actual dataset hash',
            'Exact parent training implementation and measured FLOP schedule',
            'Same Dice+CE baseline, AdamW and background handling',
            '10% and 20% calibration matches direct logit gradients at native 64-cube shape',
            'Same 75% p95 ceiling; unchanged prefix and warmup',
            'Finite objective gradients and zero-scale baseline equality','Shell syntax']))
    print('Transformer bands10/20 preflight passed',flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        prepare(); return
    controller.verify = verify
    controller.command_for = command_for
    if len(sys.argv)>1 and sys.argv[1]=='preflight':
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('--stage',type=Path,required=True)
        preflight(parser.parse_args().stage);return
    controller.main()


if __name__=='__main__':
    main()
