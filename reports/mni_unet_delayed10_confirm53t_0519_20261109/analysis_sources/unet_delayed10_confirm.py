"""Frozen nine-pair confirmation of original UNet bands delayed to update 470."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints import unet_bands_timing_strength as pilot
from scripts.unetr_subfields import cap_extension as batch

ROOT, CAP, PARENT, PILOT = pilot.ROOT, pilot.CAP, pilot.PARENT, pilot.NAME
NAME = 'mni_unet_delayed10_confirm53t_0519_20261109'
MODULE = 'scripts.mni_constraints.unet_delayed10_confirm'
SEEDS = (0, 5, 19)
REUSED_PAIRS = {(1, 0), (1, 5)}


def tasks():
    return [dict(id=f'MNI_unet_late10_f{fold}s{seed}', kind='conventional', dataset='MNI',
                 model='basic_unet16', arm='dice_bands', protocol='late10', fold=fold,
                 seed=seed, target=.1, augmentation='none')
            for seed in SEEDS for fold in range(3) if (fold, seed) not in REUSED_PAIRS]


def changed_protocol(old):
    p = deepcopy(old)
    p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'), folds=[0, 1, 2], seeds=list(SEEDS),
             decision_rule='Complete all seven new pairs and retain both pilot pairs. Consistent improvement requires positive final-budget macro Dice gain over the matched baseline in all nine pairs. No early rejection, seed replacement or retuning.',
             selection_caveat='Two development pairs are reused; report the seven new confirmation pairs separately. Overlapping development-validation folds are not an independent test set.')
    return p


def check_archive(folder):
    verify_archive(folder, read(folder/'archive_manifest.json'))
    assert read(folder/'cleanup.json')['status'] == 'complete'
    ack = read(folder/'archive_acknowledgment.json')
    assert ack['status'] == 'verified' and ack['manifest_sha256'] == digest(folder/'archive_manifest.json')
    cfg, done = read(folder/'config.json'), read(folder/'completion.json')
    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
    assert done['updates'] == done['training_attempts_charged'] == cfg['max_steps']
    assert done['consumed_training_flops'] <= cfg['budget_flops'] == CAP
    assert read(folder/'evaluation_diagnostics.json')['status'] == 'complete'
    return cfg


def prepare(_target=None):
    stage, parent = ROOT/'experiments'/NAME, ROOT/'experiments'/PILOT
    if (stage/'manifest.json').exists():
        return verify(stage)
    pilot.verify(parent)
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), stage/'source/scripts/mni_constraints/unet_delayed10_confirm.py')
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    origin = parent/'protocols/late10'
    folder = stage/'protocols/late10'
    shutil.copytree(origin, folder)
    shutil.copy2(origin/'protocol.json', folder/'parent_protocol.json')
    write(folder/'protocol.json', changed_protocol(read(origin/'protocol.json')))
    controls, donors = [], {}
    for seed in SEEDS:
        for fold in range(3):
            identity = f'MNI_basic_unet16_dice_f{fold}s{seed}'
            source = ROOT/'reports'/PARENT/'runs'/identity
            remote_source = REMOTE/PARENT/'runs'/identity
            if not source.exists():
                source = ROOT/'reports'/PARENT/'frozen/reused_runs'/identity
                remote_source = REMOTE/PARENT/'reused_runs'/identity
            entries = [('baseline', identity, source, remote_source)]
            if (fold, seed) in REUSED_PAIRS:
                identity = f'MNI_unet_late10_f{fold}s{seed}'
                entries.append(('late10', identity, ROOT/'reports'/PILOT/'runs'/identity,
                                REMOTE/PILOT/'runs'/identity))
            for condition, identity, source, remote_source in entries:
                check_archive(source)
                target = stage/'controls'/identity
                shutil.copytree(source, target)
                controls.append(dict(condition=condition, id=identity, seed=seed, fold=fold,
                                     archive=str(source), archive_sha256=digest(source/'archive_manifest.json')))
                for f in target.iterdir():
                    if f.is_file() and f.name != 'archive_acknowledgment.json':
                        donors[str(f.relative_to(stage))] = str(remote_source/f.name)
    write(stage/'controls.json', controls)
    plan = dict(name=NAME, tasks=tasks(), num_runs=7, logical_cells=18, reused_runs=11,
                new_constrained_runs=7, reused_constrained_runs=2, reused_baselines=9,
                models=['basic_unet16'], seeds=list(SEEDS), folds=[0, 1, 2],
                budget_flops=CAP, gpu_workers=2, augmentation='none', excluded_nodes=['gnode04'],
                conventional_python=read(parent/'plan.json')['conventional_python'],
                graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
                storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=2, archive=str(ROOT/'reports'/NAME)),
                authorization='User requests delayed10% original UNet bands on seeds0,5,19 and all3 folds, same cap and2 GPUs; reuse two pilot results. Temporarily pause and restore the existing full sweep.',
                recipe='Unchanged Dice-only AdamW LR0.00016, no augmentation; original union bands; training-only calibration at470 updates,target0.10,p95cap0.75,ramp76;938 optimization updates.',
                decision_rule='Report all9 paired gains and7 new confirmation gains. Consistency requires all9 positive. No automatic full-matrix recipe change.')
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Training and objective runtime byte-identical to the completed pilot. Expand seeds/folds only; all baseline and pilot controls archived and verified.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(pilot.MODULE, MODULE).replace('hippo-unet-delay', 'hippo-unet-confirm'))
    original = read(parent/'manifest.json')
    for f in stage.rglob('*'):
        if f.is_file() and original.get(str(f.relative_to(stage))) == digest(f):
            donors[str(f.relative_to(stage))] = str(REMOTE/PILOT/f.relative_to(stage))
    donors.pop('source/source_manifest.json', None)
    donors['parent_manifest.json'] = str(REMOTE/PILOT/'manifest.json')
    write(stage/'deployment_known_donors.json', donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan, original = read(stage/'plan.json'), read(stage/'parent_manifest.json')
    assert plan['tasks'] == tasks() and plan['num_runs'] == 7 and plan['gpu_workers'] == 2
    assert plan['budget_flops'] == CAP and plan['seeds'] == list(SEEDS) and plan['folds'] == [0, 1, 2]
    for name, sha in original.items():
        if name.startswith('source/') and name != 'source/source_manifest.json':
            assert digest(stage/name) == sha, name
    folder = stage/'protocols/late10'
    p = read(folder/'protocol.json')
    assert digest(folder/'parent_protocol.json') == original['protocols/late10/protocol.json']
    assert p == changed_protocol(read(folder/'parent_protocol.json'))
    s = p['models']['unet']['objectives']['dice_bands']
    assert s['max_steps'] == 938 and s['prefix_updates'] == 470 and s['constraint_warmup_updates'] == 76
    assert campaign.training_flops(938, s) <= CAP < campaign.training_flops(939, s)
    assert p['calibration_target_ratio'] == .1 and p['calibration_p95_cap'] == .75
    assert p['initial_lrs'] == {'unet': .00016} and p['augmentation'] == 'none'
    for name, key in [('profile.json', 'profile_sha256'), ('splits_10patients.json', 'splits_sha256'), ('nnunet_architecture.json', 'nnunet_architecture_sha256')]:
        assert digest(folder/name) == p[key]
    assert digest(stage/'data/MNI.pkl') == p['pkl_sha256']
    controls = read(stage/'controls.json')
    assert {(c['fold'], c['seed']) for c in controls if c['condition'] == 'baseline'} == {(f, s) for f in range(3) for s in SEEDS}
    assert {(c['fold'], c['seed']) for c in controls if c['condition'] == 'late10'} == REUSED_PAIRS
    for c in controls:
        target = stage/'controls'/c['id'];cfg = check_archive(target)
        assert digest(target/'archive_manifest.json') == c['archive_sha256']
        assert (cfg['fold'], cfg['seed']) == (c['fold'], c['seed'])
        assert cfg['model_identity'] == 'basic_unet16' and cfg['initial_lr'] == .00016 and cfg['augmentation'] == 'none'
        assert cfg['dataset_sha256'] == p['pkl_sha256']
        split = read(folder/'splits_10patients.json')[c['fold']]
        assert cfg['train_cases'] == split['train'] and cfg['validation_cases'] == split['val']
        assert cfg['script_sha256'] == digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
        if c['condition'] == 'late10':
            assert cfg['schedule'] == s and cfg['protocol_sha256'] == digest(folder/'parent_protocol.json')
            cal = read(target/'calibration.json')
            assert cal['target_ratio'] == .1 and cal['anchor_successful_updates'] == 470
        else:
            assert cfg['objective'] == 'dice' and cfg['max_steps'] == 939
    return plan


def preflight(stage):
    verify(stage)
    import torch, monai
    from scripts.mni_constraints import objectives
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS, subject
    torch.set_num_threads(4);assert monai.__version__ == '1.6.0'
    folder = stage/'protocols/late10';p = read(folder/'protocol.json');profile = read(folder/'profile.json')
    assert digest(Path(objectives.__file__)) == profile['objective_sha256']
    trainer = batch.bind(SimpleNamespace(protocol_dir=folder, model='unet', objective='dice_bands', fold=0))
    with torch.device('meta'):
        model = trainer.base.build_model('unet', read(folder/'nnunet_architecture.json'))
    assert list(model(torch.zeros((1, 1, 64, 64, 64), device='meta')).shape) == [1, 4, 64, 64, 64]
    assert sum(x.numel() for x in model.parameters()) == profile['parameters'] == 1438836
    _, optimizer = trainer.base.build_loss_and_optimizer('unet', model, p)
    assert isinstance(optimizer, torch.optim.AdamW) and optimizer.param_groups[0]['lr'] == .00016
    frame = core.base._load_pkl_dataframe(stage/'data/MNI.pkl');checks = []
    for fold, split in enumerate(read(folder/'splits_10patients.json')):
        train = core.PreparedSubfields(frame, split['train'], DATASETS['MNI'])
        assert len(train) == 20 and len(core.PreparedSubfields(frame, split['val'], DATASETS['MNI'])) == 10
        assert len({subject(c) for c in split['train']}) == 10
        assert not {subject(c) for c in split['train']} & {subject(c) for c in split['val']}
        torch.manual_seed(17);z = torch.randn((1, 4, 64, 64, 64), requires_grad=True);label = train[0]['label'][None]
        row = objectives.calibration_case(z, label, 'dice_bands', identity='basic_unet16')
        weights, stats = objectives.select_weights([row], 'dice_bands', .1, .75)
        loss = objectives.training_loss(z, label, 'dice_bands', weights, 1., identity='basic_unet16')
        assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss, z)[0]).all()
        assert abs(stats['bands']['weighted_median_ratio']-.1) < 1e-8
        checks.append(dict(fold=fold, training_crops=20, validation_crops=10, statistics=stats))
    result = dict(status='passed', manifest_sha256=digest(stage/'manifest.json'), checks=checks,
                  optimizer_updates=0, new_runs=7, reused_constrained_runs=2, reused_baselines=9,
                  torch=str(torch.__version__), monai=monai.__version__)
    write(stage/'preflight.json', result);return result


def main():
    campaign.MODULE = MODULE
    campaign.prepare, campaign.verify, campaign.preflight = prepare, verify, preflight
    campaign.train = headroom_bands.train
    campaign.main()


if __name__ == '__main__':
    main()
