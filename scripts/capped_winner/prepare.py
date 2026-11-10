"""Freeze 36 winner-transfer runs against the prior four-model capped benchmark."""
import json
import shutil
from pathlib import Path
import tarfile
from scripts.campaign90 import BUDGET, MODELS, digest, write_json
from scripts.capped_depth_delay.schedule import schedule_costs
from .objectives import ARM, ENTRY

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = 'capped_winner_31976_20261013'
STAGE = ROOT/'experiments'/CAMPAIGN


def main():
    previous = ROOT/'reports/capped_replication_31976_20261007'
    frozen = previous/'verified_metadata_20261010/frozen/source'
    source = STAGE/'source'
    if source.exists():
        raise ValueError('Use a fresh stage; never overwrite a submitted source')
    shutil.copytree(frozen, source, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Keep all already frozen benchmark dependencies byte-identical. Add only
    # missing loss dependencies plus this isolated adapter package.
    for path in (ROOT/'scripts').rglob('*.py'):
        dest = source/path.relative_to(ROOT)
        if 'capped_winner' in path.parts or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    report = ROOT/'reports'/CAMPAIGN
    profile = json.loads((report/'loss_profile.json').read_text())
    shutil.copyfile(report/'loss_profile.json', source/'winner_loss_profile.json')
    (source/'reports'/CAMPAIGN).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(report/'loss_profile.json',source/'reports'/CAMPAIGN/'loss_profile.json')
    prior = json.loads((previous/'final_summary_20261010.json').read_text())
    references = [r for r in prior['runs']+prior['reused_runs'] if r['objective'] in ('dice','dice_bands')]
    assert len(references) == 72
    write_json(source/'paired_references.json', references)
    source_hashes = {str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')}
    runs, schedules = [], {}
    for model, spec in MODELS.items():
        original = source/'experiments/capped_replication_31976_20261007'/model
        p = json.loads((original/'protocol.json').read_text())
        directory = source/'experiments'/CAMPAIGN/model
        directory.mkdir(parents=True)
        for name in ('splits_10patients.json','nnunet_architecture.json','native_geometry.json','split_provenance.json'):
            shutil.copyfile(original/name, directory/name)
        costs = json.loads((source/'reports/capped_depth_delay_31976_20261006/profiles'/f'{model}.json').read_text())['costs']
        prefix = 9 if model == 'nnunet' else 50
        update = spec['baseline_flops']+profile['extra_update_flops_bound']
        calibration = costs['dice_bands_depth']['calibration_flops']+profile['extra_calibration_flops_bound']
        schedule = schedule_costs(spec['baseline_flops'],update,update,calibration,prefix,prefix,False,False)
        assert schedule['max_steps'] >= prefix*2
        schedules[model] = schedule
        p.update(name=CAMPAIGN+'_'+model, requested_arm_order=[ARM],
            models={spec['alias']:dict(parameters=spec['parameters'], objectives={ARM:schedule})},
            training_source_sha256=source_hashes, comparison_endpoints={},
            parent_protocol_sha256=digest(original/'protocol.json'),
            winner_objective=ENTRY, winner_depth_gradient_ratio=.1,
            winner_source=dict(id='combo_depth_g100', validation_dice=.8150157184588936,
                root='/mnt/beegfsstudents/home/3160552/outer_combo_fold0_seed0_5pct_20261012_01'),
            calibration_policy='Each fold/seed uses its own ten training cases at the unchanged native prefix (nnU-Net 9 updates, others 50). Bands target 0.125 of region gradient; endpoints 0.075 of region+bands; depth 0.1 of region+bands+endpoints. Each p95 ratio capped at 0.5. No validation calibration.',
            training_loss='Native region objective + ramp*(focused outer bands + independent 1/3-mm endpoints + depth-balanced CE). No explicit A/P term, SDF, edge or auxiliary encoder.',
            geometry_policy='Outer targets rebuilt from augmented annotations on the 1-mm model grid; native-FOV validity follows the exact spatial augmentation. Distances are millimeters on that augmented grid. Depth retains the previous physical-affine/PCA geometry rule.',
            accounting='Frozen per-model native costs plus a conservative upper bound for sparse new-loss arithmetic. Same optimization-only cap and exclusions as the prior benchmark; calibration separately charged.',
            winner_profile_sha256=digest(source/'winner_loss_profile.json'),
            checkpoint_retention='Save and verify metrics, curves, calibration, ledgers and best/final checkpoint hashes before retiring completed checkpoint files.')
        write_json(directory/'protocol.json',p)
    # Finish one cell across all models first, then repeat the remaining cells.
    order = ('swinunetr','nnunet','unetr_vit_b16','basic_unet16')
    for fold in range(3):
        for seed in range(3):
            for model in order:
                runs.append(dict(model=model, trainer_alias=MODELS[model]['alias'],
                    objective=ARM, fold=fold, seed=seed))
    plan = dict(name=CAMPAIGN, total_runs=36, models=list(order), folds=[0,1,2], seeds=[0,1,2],
        runs=runs, budget_flops=BUDGET, budget_scope='optimization_only', calibration_in_cap=False,
        dataset_sha256=p['pkl_sha256'], references_sha256=digest(source/'paired_references.json'),
        loss_profile_sha256=digest(source/'winner_loss_profile.json'),
        runtime_versions=dict(torch='2.12.1+cu130',monai='1.6.0'),
        constraints='Frozen 81.5016% winner; coefficients recalibrated separately for each model/fold/seed.',
        primary_comparison='Final-budget macro Dice versus the same model/fold/seed native baseline and original bands. Best-checkpoint Dice also retained.',
        inference='Transfer/development experiment: objective selected on SwinUNETR fold0 seed0; not independent test-set evidence.',
        schedules=schedules, single_sequential_worker=True)
    write_json(source/'experiments'/CAMPAIGN/'plan.json',plan)
    manifest = {str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    write_json(STAGE/'source_manifest.json',manifest)
    shutil.copyfile(ROOT/'scripts/capped_winner/run.sbatch', STAGE/'run.sbatch')
    write_json(report/'plan_summary.json',plan)
    with tarfile.open(report/'submission_bundle.tar.gz','w:gz') as tar:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file():
                tar.add(path,arcname=str(path.relative_to(STAGE)),recursive=False)
    write_json(report/'preparation_receipt.json',dict(total_runs=36,source_files=len(manifest),
        source_manifest_sha256=digest(STAGE/'source_manifest.json'),
        bundle_sha256=digest(report/'submission_bundle.tar.gz'), status='prepared'))
    print(json.dumps(dict(runs=36,updates={m:s['max_steps'] for m,s in schedules.items()},
                         bundle=str(report/'submission_bundle.tar.gz'))))


if __name__ == '__main__':
    main()
