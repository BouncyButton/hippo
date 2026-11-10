"""Continue fixed-recipe UNet seed recovery, starting with user-selected seed10."""
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands, seed_recovery as runner
from scripts.mni_constraints.shared43_study import ROOT, NAME as PARENT, CAP
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_unet_seed10_recovery_shared43t_20261108'
MODULE = 'scripts.mni_constraints.unet_seed10_recovery'
PREVIOUS = 'mni_seed_recovery_shared43t_20261108'
SEED5 = 'mni_seed5_all_models_shared43t_20261108'
MODELS = ('basic_unet16',)
CANDIDATES = list(range(10, 50))
ALL_MODELS = ('nnunet', 'basic_unet16', 'swinunetr', 'unetr_vit_b16')


def tasks():
    return [dict(id=f'MNI_basic_unet16_dice_f{f}s{s}', kind='conventional',
                 dataset='MNI', model='basic_unet16', arm='dice', fold=f, seed=s,
                 protocol='basic_unet16_dice', augmentation='none')
            for s in CANDIDATES for f in range(3)]


def prepare(_target=None):
    stage = ROOT/'experiments'/NAME
    if (stage/'manifest.json').exists():
        return verify(stage)
    parent = ROOT/'experiments'/PARENT
    for name, sha in read(parent/'manifest.json').items():
        assert digest(parent/name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('unet_seed10_recovery.py', 'seed_recovery.py', 'collect_seed_recovery.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name, stage/'source/scripts/mni_constraints'/name)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    folder = stage/'protocols/basic_unet16_dice'
    shutil.copytree(parent/'protocols/basic_unet16_dice', folder)
    shutil.copy2(folder/'protocol.json', folder/'parent_protocol.json')
    protocol = read(folder/'protocol.json')
    rule = 'Keep seeds0,2,5 for nnUNet/SwinUNETR/UNETR and seeds0,5 for UNet. Test UNet seeds10 onward in numerical order, all three original folds per candidate; stop at the first mean final-budget Dice>=50%. Report every-fold qualification separately.'
    protocol.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'), seeds=CANDIDATES,
                    decision_rule=rule)
    write(folder/'protocol.json', protocol)
    known = []; donors = {}
    for model in ALL_MODELS:
        for seed in (0, 2, 5):
            values = []; archives = []
            for fold in range(3):
                identifier = f'MNI_{model}_dice_f{fold}s{seed}'
                if seed != 5:
                    src = ROOT/'experiments'/PREVIOUS/'known_runs'/identifier
                    remote_src = REMOTE/PREVIOUS/'known_runs'/identifier
                elif model in ('nnunet', 'basic_unet16'):
                    src = ROOT/'reports'/SEED5/'frozen/reused_runs'/identifier
                    remote_src = REMOTE/SEED5/'reused_runs'/identifier
                else:
                    src = ROOT/'reports'/SEED5/'runs'/identifier
                    remote_src = REMOTE/SEED5/'runs'/identifier
                verify_archive(src, read(src/'archive_manifest.json'))
                assert read(src/'cleanup.json')['status'] == 'complete'
                dst = stage/'known_runs'/identifier
                shutil.copytree(src, dst)
                for item in dst.iterdir():
                    if item.is_file() and item.name != 'archive_acknowledgment.json':
                        donors[str(item.relative_to(stage))] = str(remote_src/item.name)
                values.append(100*read(src/'validation_latest.json')['summary']['macro_hard_dice'])
                archives.append(dict(id=identifier, archive_manifest_sha256=digest(src/'archive_manifest.json')))
            known.append(dict(model=model, seed=seed, validation_by_fold_pct=values,
                              mean_validation_pct=sum(values)/3, passes_mean50=sum(values)/3>=50,
                              passes_every_fold50=min(values)>=50, archives=archives))
    write(stage/'known_seeds.json', known)
    write(stage/'deployment_known_donors.json', donors)
    write(stage/'prior_attempts.json', dict(report=str(ROOT/'reports'/PREVIOUS),
          manifest_sha256=digest(ROOT/'experiments'/PREVIOUS/'manifest.json'),
          preserved_interrupted_run='MNI_basic_unet16_dice_f2s8',
          policy='Prior failed attempts and interrupted seed8 remain in their original archives; no deletion or resumption is authorized by this stage.'))
    plan = dict(name=NAME, tasks=tasks(), num_runs=len(tasks()), models=list(MODELS),
                candidate_seeds=CANDIDATES, budget_flops=CAP, gpu_workers=2,
                augmentation='none', folds=[0,1,2], retirements=[],
                conventional_python=read(parent/'plan.json')['conventional_python'],
                graceful_stop_seconds=20700, archive_wait_timeout_seconds=600,
                archive_resume_module=MODULE,
                storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=12,
                             archive=str(ROOT/'reports'/NAME)),
                selection_rule=rule,
                selection_caveat='Development-validation seed selection; mean50 does not imply every fold50. Preserve all attempted candidates.',
                authorization='User: Go with seed10 for UNet, proceed until three working seeds for all models. Test10 first, then11 onward as needed, unchanged recipes/folds/43T cap; stop immediately after qualification. This finite block ends at49 and can be extended if needed under the same user request.')
    write(stage/'plan.json', plan); write(stage/'control.json', dict(pause=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Training source unchanged; candidate order begins10 and known seeds include verified common seed5.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.shared43_study', MODULE).replace('hippo-mni-14', 'hippo-unet-seeds'))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan = read(stage/'plan.json'); parent = read(stage/'parent_manifest.json')
    assert plan['tasks'] == tasks() and plan['budget_flops'] == CAP and plan['gpu_workers'] == 2
    folder = stage/'protocols/basic_unet16_dice'
    p = read(folder/'protocol.json'); old = read(folder/'parent_protocol.json')
    allowed = {'name', 'data', 'seeds', 'decision_rule'}
    assert digest(folder/'parent_protocol.json') == parent['protocols/basic_unet16_dice/protocol.json']
    assert {k:v for k,v in p.items() if k not in allowed} == {k:v for k,v in old.items() if k not in allowed}
    assert p['seeds'] == CANDIDATES and p['folds'] == [0,1,2] and p['budget_flops'] == CAP
    for name, key in [('profile.json','profile_sha256'), ('splits_10patients.json','splits_sha256'), ('nnunet_architecture.json','nnunet_architecture_sha256')]:
        assert digest(folder/name) == p[key]
    assert digest(stage/'data/MNI.pkl') == p['pkl_sha256']
    for name, sha in parent.items():
        if name.startswith('source/') and name != 'source/source_manifest.json':
            assert digest(stage/name) == sha, name
    known = read(stage/'known_seeds.json')
    assert {(r['model'],r['seed']) for r in known} == {(m,s) for m in ALL_MODELS for s in (0,2,5)}
    for row in known:
        values = []
        for item in row['archives']:
            folder = stage/'known_runs'/item['id']
            verify_archive(folder, read(folder/'archive_manifest.json'))
            assert digest(folder/'archive_manifest.json') == item['archive_manifest_sha256']
            values.append(100*read(folder/'validation_latest.json')['summary']['macro_hard_dice'])
        assert values == row['validation_by_fold_pct']
        assert row['passes_mean50'] == (sum(values)/3 >= 50)
    return plan


def bind_runner():
    runner.MODULE=MODULE; runner.MODELS=MODELS; runner.CANDIDATES=CANDIDATES
    runner.RETIREMENTS=(); runner.verify=verify


def preflight(stage):
    bind_runner()
    return runner.preflight(stage)


def selection(stage, plan):
    bind_runner()
    return runner.selection(stage, plan)


def eligible(stage, plan, state):
    bind_runner()
    return runner.eligible(stage, plan, state)


def main():
    bind_runner()
    campaign.MODULE=MODULE; campaign.prepare=prepare; campaign.verify=verify
    campaign.preflight=preflight; campaign.train=headroom_bands.train
    batch.run=runner.run
    campaign.main()


if __name__ == '__main__':
    main()
