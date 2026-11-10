"""Fixed-definition UNETR bands: a bounded strength-by-onset experiment."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints.common53_study import ROOT, NAME as PARENT, CAP
from scripts.mni_constraints.shared43_study import NAME as BANDS_PARENT
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_unetr_strength_timing53t_f0s19_20261108'
MODULE = 'scripts.mni_constraints.unetr_constraint_pilot'
CONDITIONS = {
    'original_t05': (.05, 33),
    'original_t20': (.20, 33),
    'delayed_t05': (.05, 102),
    'delayed_t20': (.20, 102),
}
BASELINE = 'MNI_unetr_vit_b16_dice_f0s19'


def tasks():
    return [dict(id=f'MNI_unetr_{key}_f0s19', kind='conventional', dataset='MNI',
                 model='unetr_vit_b16', arm='dice_bands', fold=0, seed=19,
                 protocol=key, target=target, prefix_updates=prefix,
                 augmentation='xy_d4_v1') for key, (target, prefix) in CONDITIONS.items()]


def changed_protocol(old, baseline, key):
    p = deepcopy(old)
    target, prefix = CONDITIONS[key]
    s = p['models']['unet']['objectives']['dice_bands']
    s.update(prefix_updates=prefix, sdf_prefix_updates=prefix)
    maximum = 0
    while campaign.training_flops(maximum + 1, s) <= CAP:
        maximum += 1
    consumed = campaign.training_flops(maximum, s)
    s.update(max_steps=maximum, consumed_training_flops=consumed, consumed_flops=consumed,
             consumed_total_flops=consumed+s['calibration_flops'], unused_flops=CAP-consumed,
             patient_pass_epochs=maximum/20)
    # All conditions fit exactly the same409 updates and LR trajectory.
    assert maximum == baseline['models']['unet']['objectives']['dice']['max_steps'] == 409
    p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'), folds=[0], seeds=[19],
             budget_flops=CAP, lr_horizon_updates=409,
             lr_schedule_details=baseline['lr_schedule_details'],
             validation_target_flops=baseline['validation_target_flops'],
             calibration_target_ratio=target, calibration_p95_cap=.75,
             budget_origin=baseline['budget_origin'],
             primary_metric='Final-budget macro subfield Dice versus matched53T seed19 fold0 baseline.',
             decision_rule='Complete all four cells of the2x2 target/onset pilot; retain every outcome. No automatic expansion or recipe adoption.',
             timing_policy=f'Calibrate after{prefix} baseline updates; apply original bands with unchanged33-update linear ramp. Delay also moves calibration, so it is an onset/calibration-policy comparison.',
             calibration_strength_policy='Fresh all20 training-crop actual-base calibration at declared onset; fixed coefficient thereafter;75% p95 guard.',
             selection_caveat='Exploratory single-fold single-seed development-validation experiment; apparent gains require independent matched seed/fold confirmation.')
    return p


def prepare(_target=None):
    stage = ROOT/'experiments'/NAME
    parent, bands_parent = ROOT/'experiments'/PARENT, ROOT/'experiments'/BANDS_PARENT
    if (stage/'manifest.json').exists():
        return verify(stage)
    for source in (parent, bands_parent):
        for name, sha in read(source/'manifest.json').items():
            assert digest(source/name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), stage/'source/scripts/mni_constraints/unetr_constraint_pilot.py')
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copy2(bands_parent/'manifest.json', stage/'bands_parent_manifest.json')
    origin = bands_parent/'protocols/unetr_vit_b16_dice_bands'
    baseline = read(parent/'protocols/unetr_vit_b16_dice/protocol.json')
    write(stage/'baseline_protocol.json', baseline)
    for key in CONDITIONS:
        folder = stage/'protocols'/key
        shutil.copytree(origin, folder)
        shutil.copy2(folder/'protocol.json', folder/'parent_protocol.json')
        write(folder/'protocol.json', changed_protocol(read(folder/'parent_protocol.json'), baseline, key))
    control = ROOT/'reports'/PARENT/'runs'/BASELINE
    verify_archive(control, read(control/'archive_manifest.json'))
    assert read(control/'cleanup.json')['status'] == 'complete'
    shutil.copytree(control, stage/'controls'/BASELINE)
    plan = dict(name=NAME, tasks=tasks(), num_runs=4, logical_runs=5, reused_runs=1,
        models=['unetr_vit_b16'], folds=[0], seeds=[19], budget_flops=CAP, gpu_workers=2,
        excluded_nodes=['gnode04'], augmentation='xy_d4_v1',
        conventional_python=read(parent/'plan.json')['conventional_python'],
        graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=12, archive=str(ROOT/'reports'/NAME)),
        authorization='User requests diagnosing and improving UNETR constraint benefit, explicitly keeping original definitions. Bounded strength/timing pilot before broad sweep; two GPUs, latest cap, established seed/fold.',
        scope='Original balanced union bands unchanged.2x2 target(.05,.20) and onset(33,102),same33-update ramp,409 updates,LR0.0006,warmup24,dropout0.1,AdamW,xyD4,n10. Existing read-only quarter probes; no new optimizer instrumentation.',
        limitation='Onset and calibration anchor move together. Detached-logit probes do not establish parameter-gradient or Adam-update attribution. Historical class-aware results excluded from this pilot.',
        selection_caveat='Exploratory development-validation tuning; no automatic expansion or best-of-four claim of a general improvement.')
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Training runtime byte-identical to latest53T baseline and historical43T bands; only declared strength/onset and cap/schedule configuration differ.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.common53_study', MODULE).replace('hippo-common53', 'hippo-unetr-probe'))
    original = read(parent/'manifest.json')
    donors = {str(f.relative_to(stage)): str(REMOTE/PARENT/f.relative_to(stage))
              for f in stage.rglob('*') if f.is_file() and original.get(str(f.relative_to(stage))) == digest(f)}
    donors.pop('source/source_manifest.json', None)
    donors['parent_manifest.json'] = str(REMOTE/PARENT/'manifest.json')
    for key in CONDITIONS:
        for f in (stage/'protocols'/key).iterdir():
            if f.name not in ('protocol.json', 'parent_protocol.json'):
                donors[str(f.relative_to(stage))] = str(REMOTE/BANDS_PARENT/'protocols/unetr_vit_b16_dice_bands'/f.name)
    write(stage/'deployment_known_donors.json', donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan = read(stage/'plan.json')
    assert plan['tasks'] == tasks() and plan['num_runs'] == 4 and plan['gpu_workers'] == 2
    assert plan['budget_flops'] == CAP
    parent = read(stage/'parent_manifest.json')
    for name, sha in parent.items():
        if name.startswith('source/') and name != 'source/source_manifest.json':
            assert digest(stage/name) == sha, name
    baseline = read(stage/'baseline_protocol.json')
    assert digest(stage/'baseline_protocol.json') == parent['protocols/unetr_vit_b16_dice/protocol.json']
    control = stage/'controls'/BASELINE
    verify_archive(control, read(control/'archive_manifest.json'))
    cfg = read(control/'config.json')
    assert cfg['fold'] == 0 and cfg['seed'] == 19 and cfg['budget_flops'] == CAP
    assert cfg['script_sha256'] == digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
    assert cfg['max_steps'] == 409
    for key in CONDITIONS:
        folder = stage/'protocols'/key
        old = read(folder/'parent_protocol.json')
        assert digest(folder/'parent_protocol.json') == read(stage/'bands_parent_manifest.json')['protocols/unetr_vit_b16_dice_bands/protocol.json']
        p = read(folder/'protocol.json')
        assert p == changed_protocol(old, baseline, key)
        for name in ('initial_lrs','lr_horizon_updates','lr_schedule_details','augmentation','dropout_rate',
                     'loss_arguments','adamw_weight_decay','pkl_sha256','splits_sha256'):
            assert p[name] == baseline[name], name
        for name, field in [('profile.json','profile_sha256'), ('splits_10patients.json','splits_sha256'),
                            ('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name) == p[field]
        assert digest(stage/'data/MNI.pkl') == p['pkl_sha256']
        split = read(folder/'splits_10patients.json')[0]
        assert cfg['train_cases'] == split['train'] and cfg['validation_cases'] == split['val']
        s = p['models']['unet']['objectives']['dice_bands']
        assert campaign.training_flops(409,s) <= CAP < campaign.training_flops(410,s)
        assert s['constraint_warmup_updates'] == 33 and s['calibration_flops'] == 865633170892
    return plan


def preflight(stage):
    verify(stage)
    import torch
    import monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4)
    assert monai.__version__ == '1.6.0'
    frame = core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    rows = []
    for key in CONDITIONS:
        folder = stage/'protocols'/key
        p = read(folder/'protocol.json')
        batch.bind(SimpleNamespace(protocol_dir=folder, model='unet', objective='dice_bands', fold=0))
        model = core.base.build_model('unet', read(folder/'nnunet_architecture.json')).to('meta').train()
        assert list(model(torch.zeros((1,1,64,64,64),device='meta')).shape) == [1,4,64,64,64]
        assert sum(x.numel() for x in model.parameters()) == 92685572
        split = read(folder/'splits_10patients.json')[0]
        data = core.PreparedSubfields(frame,split['train'],DATASETS['MNI'])
        assert len(data) == 20 and len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI'])) == 10
        label = data[0]['label'][None]
        torch.manual_seed(19)
        z = torch.randn((1,4,64,64,64),requires_grad=True)
        row = objectives.calibration_case(z,label,'dice_bands',identity='unetr_vit_b16')
        weights, statistics = objectives.select_weights([row],'dice_bands',p['calibration_target_ratio'],.75)
        value = objectives.training_loss(z,label,'dice_bands',weights,1.,identity='unetr_vit_b16')
        assert torch.isfinite(value) and torch.isfinite(torch.autograd.grad(value,z)[0]).all()
        assert abs(statistics['bands']['weighted_median_ratio']-p['calibration_target_ratio']) < 1e-7
        assert statistics['bands']['weighted_p95_ratio'] <= .7500001
        rows.append(dict(condition=key, synthetic_statistics=statistics, schedule=p['models']['unet']['objectives']['dice_bands']))
    result = dict(status='passed', manifest_sha256=digest(stage/'manifest.json'), optimizer_updates=0,
                  checks=rows, unchanged_training_runtime=True, matched_archived_baseline=True)
    write(stage/'preflight.json',result)
    return result


def main():
    campaign.MODULE = MODULE
    campaign.prepare, campaign.verify, campaign.preflight = prepare, verify, preflight
    campaign.train = headroom_bands.train
    campaign.main()


if __name__ == '__main__':
    main()
