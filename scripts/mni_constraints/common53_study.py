"""Common seeds0519: a fixed36-cell baseline comparison at a shared53T cap."""
from copy import deepcopy
import math
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands, seed5_all_models
from scripts.mni_constraints.shared43_study import ROOT, NAME as PARENT, CAP as OLD_CAP, MODELS
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_common0519_shared53t_3fold_20261108'
MODULE = 'scripts.mni_constraints.common53_study'
PREVIOUS = 'mni_seed19_all_models_shared43t_20261108'
CAP = 53305109592450
SEEDS = (0, 5, 19)


def tasks():
    # Run the seed being rescued first, then the two established common seeds.
    return [dict(id=f'MNI_{model}_dice_f{fold}s{seed}', kind='conventional', dataset='MNI',
                 model=model, arm='dice', fold=fold, seed=seed, protocol=model + '_dice',
                 augmentation='xy_d4_v1' if model in ('swinunetr', 'unetr_vit_b16') else 'none')
            for seed in (19, 0, 5) for fold in range(3) for model in MODELS]


def extend_protocol(old):
    p = deepcopy(old)
    alias = p['trainer_alias']
    s = deepcopy(p['models'][alias]['objectives']['dice'])
    cost = s['flops_per_update']
    count = CAP // cost
    s.update(max_steps=count, consumed_training_flops=count * cost,
             consumed_flops=count * cost, consumed_total_flops=count * cost,
             unused_flops=CAP - count * cost, patient_pass_epochs=count / 20)
    p['models'][alias]['objectives']['dice'] = s
    horizon = 200 if p['model_identity'] == 'nnunet' else count
    warmup = (0 if p['model_identity'] == 'nnunet' else
              24 if p['model_identity'] == 'unetr_vit_b16' else math.ceil(.05 * horizon))
    p.update(name=NAME, data=str(REMOTE / NAME / 'data/MNI.pkl'), seeds=list(SEEDS),
             budget_flops=CAP, lr_horizon_updates=horizon,
             lr_schedule_details=dict(policy=p['lr_policy'], horizon_updates=horizon,
                 warmup_updates=warmup, decay_start_index=horizon // 2 if p['lr_policy'] == 'hold_half' else warmup),
             validation_target_flops=sorted({CAP * i // 20 for i in range(1, 21)} |
                 {old['models'][alias]['objectives']['dice']['consumed_training_flops']}),
             budget_origin='Shared53305109592450 optimization FLOPs: exactly150 nnUNet updates,23.9456% above43T.',
             learning_rate_policy='Preserve nnUNet200-update horizon/100-update hold. Other models retain their existing LR policy with horizon matched to the larger budget; initial rates unchanged.',
             primary_metric='Final-budget validation foreground macro Dice; report every fold/seed and paired change from43T.',
             decision_rule='Fixed four models x folds0,1,2 x common seeds0,5,19. Complete all36 cells, retaining low scores. Report every-fold50 and mean50 separately. No seed replacement or early stopping.')
    return p


def prepare(_target=None):
    stage = ROOT / 'experiments' / NAME
    parent = ROOT / 'experiments' / PARENT
    if (stage / 'manifest.json').exists():
        return verify(stage)
    for name, sha in read(parent / 'manifest.json').items():
        assert digest(parent / name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent / part, stage / part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('common53_study.py', 'collect_common53_study.py', 'seed5_all_models.py'):
        shutil.copy2(ROOT / 'scripts/mni_constraints' / name, stage / 'source/scripts/mni_constraints' / name)
    shutil.copy2(parent / 'manifest.json', stage / 'parent_manifest.json')
    controls, donors, schedules = [], {}, {}
    for model in MODELS:
        folder = stage / 'protocols' / (model + '_dice')
        shutil.copytree(parent / 'protocols' / (model + '_dice'), folder)
        shutil.copy2(folder / 'protocol.json', folder / 'parent_protocol.json')
        old = read(folder / 'protocol.json')
        p = extend_protocol(old)
        write(folder / 'protocol.json', p)
        alias = p['trainer_alias']
        schedules[model] = dict(old_updates=old['models'][alias]['objectives']['dice']['max_steps'],
                               new_updates=p['models'][alias]['objectives']['dice']['max_steps'],
                               old_lr_horizon=old['lr_horizon_updates'], new_lr_horizon=p['lr_horizon_updates'],
                               initial_lr=p['initial_lrs'][alias])
        for seed in SEEDS:
            for fold in range(3):
                identifier = f'MNI_{model}_dice_f{fold}s{seed}'
                relative = ('frozen/controls' if seed != 19 else
                            'frozen/reused_runs' if model == 'basic_unet16' else 'runs')
                remote_relative = relative.removeprefix('frozen/')
                src = ROOT / 'reports' / PREVIOUS / relative / identifier
                verify_archive(src, read(src / 'archive_manifest.json'))
                assert read(src / 'cleanup.json')['status'] == 'complete'
                cfg = read(src / 'config.json')
                split = read(folder / 'splits_10patients.json')[fold]
                assert cfg['schedule'] == old['models'][alias]['objectives']['dice']
                assert cfg['budget_flops'] == OLD_CAP and cfg['initial_lr'] == old['initial_lrs'][alias]
                assert cfg['lr_horizon_updates'] == old['lr_horizon_updates']
                assert cfg['train_cases'] == split['train'] and cfg['validation_cases'] == split['val']
                assert cfg['dataset_sha256'] == digest(stage / 'data/MNI.pkl')
                assert cfg['script_sha256'] == digest(stage / 'source/scripts/cross_dataset_campaign/capped_core.py')
                dst = stage / 'controls' / identifier
                shutil.copytree(src, dst)
                for f in dst.iterdir():
                    if f.is_file() and f.name != 'archive_acknowledgment.json':
                        donors[str(f.relative_to(stage))] = str(REMOTE / PREVIOUS / remote_relative / identifier / f.name)
                controls.append(dict(id=identifier, model=model, seed=seed, fold=fold,
                    validation_dice=read(src / 'validation_latest.json')['summary']['macro_hard_dice'],
                    archive_manifest_sha256=digest(src / 'archive_manifest.json')))
    write(stage / 'controls.json', controls)
    write(stage / 'deployment_known_donors.json', donors)
    plan = dict(name=NAME, tasks=tasks(), num_runs=36, logical_runs=36, models=list(MODELS),
                seeds=list(SEEDS), folds=[0, 1, 2], budget_flops=CAP, previous_budget_flops=OLD_CAP,
                gpu_workers=2, excluded_nodes=['gnode04'], augmentation='model_specific',
                conventional_python=read(parent / 'plan.json')['conventional_python'],
                graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
                storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=12, archive=str(ROOT / 'reports' / NAME)),
                schedules=schedules,
                authorization='User requests larger cap for all models and explicitly confirms common seeds0,5,19 for folds0,1,2. Proposed150-update nnUNet cap shared across4models;36baseline runs. Two workers, one running and one dependent queued allocation maximum; verified archive and cleanup per run.',
                selection_caveat='Development-validation comparison. nnUNet preserves its LR trajectory; other models stretch their budget-matched schedules, so their comparison includes the schedule extension. UNet seed5 was previously accepted on its mean only. Report all cells without replacing failures.')
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False))
    write(stage / 'source_version.json', dict(parent_manifest_sha256=digest(parent / 'manifest.json'),
        controls_manifest_sha256=digest(ROOT / 'experiments' / PREVIOUS / 'manifest.json'),
        changes='Training source byte-identical. New shared53T cap, frozen common seeds and paired43T archives; nnUNet LR horizon unchanged, others budget-matched. Fresh training; completed historical checkpoints were already cleaned.'))
    (stage / 'run.sbatch').write_text((parent / 'run.sbatch').read_text().replace(
        'scripts.mni_constraints.shared43_study', MODULE).replace('hippo-mni-14', 'hippo-common53'))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        assert digest(stage / name) == sha, name
    parent = read(stage / 'parent_manifest.json')
    plan = read(stage / 'plan.json')
    assert plan['tasks'] == tasks() and plan['num_runs'] == len(tasks()) == 36
    assert len({t['id'] for t in tasks()}) == 36 and plan['gpu_workers'] == 2
    assert plan['budget_flops'] == CAP and plan['seeds'] == list(SEEDS)
    for model in MODELS:
        folder = stage / 'protocols' / (model + '_dice')
        old = read(folder / 'parent_protocol.json')
        assert digest(folder / 'parent_protocol.json') == parent[f'protocols/{model}_dice/protocol.json']
        p = read(folder / 'protocol.json')
        assert p == extend_protocol(old)
        s = p['models'][p['trainer_alias']]['objectives']['dice']
        assert s['max_steps'] * s['flops_per_update'] <= CAP < (s['max_steps'] + 1) * s['flops_per_update']
        assert s['patient_pass_epochs'] == s['max_steps'] / 20
        assert all(0 < x <= CAP for x in p['validation_target_flops'])
        for name, k in [('profile.json', 'profile_sha256'), ('splits_10patients.json', 'splits_sha256'), ('nnunet_architecture.json', 'nnunet_architecture_sha256')]:
            assert digest(folder / name) == p[k]
        assert digest(stage / 'data/MNI.pkl') == p['pkl_sha256']
    for name, sha in parent.items():
        if name.startswith('source/') and name != 'source/source_manifest.json':
            assert digest(stage / name) == sha, name
    controls = read(stage / 'controls.json')
    assert {c['id'] for c in controls} == {t['id'] for t in tasks()} and len(controls) == 36
    for c in controls:
        d = stage / 'controls' / c['id']
        verify_archive(d, read(d / 'archive_manifest.json'))
        assert digest(d / 'archive_manifest.json') == c['archive_manifest_sha256']
    return plan


def preflight(stage):
    seed5_all_models.verify = verify
    result = seed5_all_models.preflight(stage)
    lr_checks = []
    for model in MODELS:
        p = read(stage / 'protocols' / (model + '_dice') / 'protocol.json')
        alias = p['trainer_alias']
        s = p['models'][alias]['objectives']['dice']
        values = [batch.lr_at(i, p['lr_horizon_updates'], p['initial_lrs'][alias], model, p['lr_policy']) for i in range(s['max_steps'])]
        assert all(math.isfinite(v) and 0 < v <= p['initial_lrs'][alias] for v in values)
        decay = p['lr_schedule_details']['decay_start_index']
        assert all(b <= a + 1e-15 for a, b in zip(values[decay:], values[decay + 1:]))
        if model == 'nnunet':
            assert s['max_steps'] == 150 and values[:100] == [.01] * 100
        lr_checks.append(dict(model=model, updates=len(values), first_lr=values[0], final_lr=values[-1], horizon=p['lr_horizon_updates']))
    result['lr_checks'] = lr_checks
    write(stage / 'preflight.json', result)
    return result


def main():
    campaign.MODULE = MODULE
    campaign.prepare, campaign.verify, campaign.preflight = prepare, verify, preflight
    campaign.train = headroom_bands.train
    campaign.main()


if __name__ == '__main__':
    main()
