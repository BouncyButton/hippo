"""Freeze 56 unaugmented ADNI conditions, source, profiles and immutable splits."""
import json
from pathlib import Path
import shutil
import tarfile

from scripts.adni_baselines.core import ROOT, MODELS, BUDGET, base, PreparedADNI
from scripts.adni_constraints.objectives import ARMS, GEOMETRY, geometry_weights
from scripts.capped_depth_delay.schedule import schedule_costs, training_flops

NAME = 'adni_constraints_noaug_5pct_fold0_seed0_20261013'
PROFILE_NAME = 'adni_constraints_5pct_fold0_seed0_20261013'


def main():
    stage, report = ROOT/'experiments'/NAME, ROOT/'reports'/NAME
    if stage.exists():
        raise FileExistsError('Refusing to overwrite frozen experiment')
    for identity in MODELS:
        profile_path = ROOT/'reports'/PROFILE_NAME/'profiles'/(identity+'.json')
        measured = json.loads(profile_path.read_text())
        if not measured.get('binary_audit_extension') or len(measured['costs']) != 16:
            raise ValueError('All binary arithmetic profiles must be complete and reconciled')
    report.mkdir(parents=True, exist_ok=True)
    previous = ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013'
    msd = ROOT/'reports/capped_replication_31976_20261007/verified_metadata_20261010/frozen/source/experiments/capped_replication_31976_20261007'
    data = ROOT/'datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'
    split = json.loads((previous/'protocols/swinunetr/splits_10patients.json').read_text())[0]
    train = PreparedADNI(base._load_pkl_dataframe(data), split['train'])
    import numpy as np
    geometry_checks = []
    for item in train:
        weights = geometry_weights(item['label'][None], np.eye(4), GEOMETRY)
        geometry_checks.append(dict(case_name=item['case_name'], weight_sum=float(weights.sum()),
                                    roi_voxels=int((weights > 0).sum())))
    source = stage/'source'
    paths = ['scripts/adni_baselines', 'scripts/adni_constraints', 'scripts/capped_depth_delay',
             'scripts/depth_supervision/geometry.py', 'scripts/campaign90.py', 'scripts/optimization90.py',
             'scripts/compute_constraint_objectives.py', 'scripts/nnunet_multi_objective.py',
             'scripts/nnunet_band_objective.py', 'scripts/train_compute_matched_baselines.py',
             'scripts/profile_intermediate_baselines.py', 'scripts/profile_unet_constraint_budget.py',
             'baselines/simple_segmentation.py', 'baselines/swin_unetr/swin_unetr.py',
             'baselines/nnUNet/nnunetv2', 'thesis/new_constraints',
             'reports/compute_budget_5pct_fold0_seed0_20261003/profile_reference.py',
             'reports/compute_budget_5pct_fold0_seed0_20261003/profile_matched_models.py']
    for name in paths:
        src, dst = ROOT/name, source/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:
            shutil.copyfile(src, dst)
    sources = {str(p.relative_to(source)):base.digest(p) for p in source.rglob('*') if p.is_file()}
    runs = []
    for identity, spec in MODELS.items():
        origin = previous/'protocols'/identity
        parent = json.loads((origin/'protocol.json').read_text())
        old = json.loads((msd/identity/'protocol.json').read_text())
        directory = stage/'protocols'/identity
        directory.mkdir(parents=True)
        for name in ('splits_10patients.json','nnunet_architecture.json'):
            shutil.copyfile(origin/name, directory/name)
        profile = ROOT/'reports'/PROFILE_NAME/'profiles'/(identity+'.json')
        shutil.copyfile(profile, directory/'profile.json')
        measured = json.loads(profile.read_text())
        costs, schedules = measured['costs'], {}
        if measured['model'] != identity or measured['num_classes'] != 2:
            raise ValueError('Wrong binary arithmetic profile')
        f0 = costs['dice']['update_flops']
        if f0 != parent['models'][spec['alias']]['flops_per_update']:
            raise ValueError('Baseline arithmetic changed')
        prefix, ramp = (9, 9) if identity == 'nnunet' else (50, 50)
        for arm in ARMS:
            if arm == 'dice':
                schedule = dict(prefix_updates=0, sdf_prefix_updates=0, baseline_flops_per_update=f0,
                                early_flops_per_update=f0, flops_per_update=f0, calibration_flops=0,
                                constraint_warmup_updates=ramp, sdf_warmup_updates=ramp, budget_scope='optimization_only')
            else:
                schedule = schedule_costs(f0, costs[arm.replace('_sdf','')]['update_flops'],
                    costs[arm]['update_flops'], costs[arm]['calibration_flops'], prefix, ramp,
                    'sdf' in arm, 'depth' in arm)
                if old['sdf_policy'][arm] == 'original_shared_ramp':
                    schedule.update(sdf_prefix_updates=prefix, early_flops_per_update=schedule['flops_per_update'],
                                    sdf_delay_fraction_numerator=0, sdf_delay_fraction_denominator=1)
            n = 0
            while training_flops(n+1, schedule) <= BUDGET:
                n += 1
            consumed = training_flops(n, schedule)
            schedule.update(max_steps=n, consumed_training_flops=consumed, consumed_flops=consumed,
                            consumed_total_flops=consumed+schedule['calibration_flops'],
                            unused_flops=BUDGET-consumed, patient_pass_epochs=n/10)
            schedules[arm] = schedule
            runs.append(dict(model=identity, alias=spec['alias'], objective=arm, fold=0, seed=0,
                             updates=n, optimization_gflops=consumed/1e9,
                             calibration_gflops=schedule['calibration_flops']/1e9))
        protocol = {**parent, 'name': NAME+'_'+identity, 'augmentation':'none', 'constraints':True,
                    'trainer_alias':spec['alias'], 'requested_arm_order':list(ARMS),
                    'budget_scope':'optimization_only', 'constraint_warmup_updates':ramp,
                    'models':{spec['alias']:dict(parameters=measured['parameters'], objectives=schedules)},
                    'profile_sha256':base.digest(directory/'profile.json'),
                    'training_source_sha256':sources, 'depth_geometry':GEOMETRY,
                    'native_geometry':{c:dict(model_affine=np.eye(4).tolist()) for c in split['train']+split['val']},
                    'sdf_policy':old['sdf_policy'],
                    'constraint_geometry':'Binary hippocampus; 2-step bands; original signed face edge; full-volume half-voxel SDF clipped at 8 voxels. Depth uses 2-voxel bins in an 8-voxel ROI; plane normal is array axis 0 projected off GT PC1. Source anatomical orientation and physical spacing are unavailable; no mm interpretation.',
                    'calibration_policy':'Ten unaugmented training crops at step 50 (nnU-Net 9), median gradient ratio 0.1/p95 cap 0.5; same frozen MSD calibration order and SDF timing. Outside optimization cap and separately reported.',
                    'augmentation_policy':'None, for every baseline and constraint condition; explicit user instruction.',
                    'checkpoint_retention':'Keep full-precision best model and all metrics. Verify then remove only this run\'s completed latest optimizer checkpoint; final predictions summarized per validation crop.',
                    'accounting':measured['accounting']}
        base.atomic_json(directory/'protocol.json', protocol)
    shutil.copyfile(previous/'split_provenance.json', stage/'split_provenance.json')
    plan = dict(name=NAME, dataset='ADNI', fold=0, seed=0, requested_fraction=.05,
                train_subjects=5, train_crops=10, val_subjects=27, val_crops=54,
                augmentation='none', num_runs=56, new_baselines=4, constraint_runs=52,
                runs=runs, models=list(MODELS), objectives=list(ARMS), sequential=True,
                budget_flops_per_run=BUDGET, budget_scope='optimization_only', calibration_in_cap=False,
                pkl_sha256=base.digest(data),
                remote_pkl='/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl',
                runtime_versions=dict(torch='2.12.1+cu130',monai='1.6.0'),
                checkpoint_retention='Retain best FP32 model and complete metadata for all 56 runs; latest optimizer state retained only while incomplete.',
                comparison='Compare final-budget Dice and the same 20-point best Dice to the new unaugmented baseline within each model. One split/seed is exploratory.',
                prior_augmented_baselines=str(previous), geometry_checks=geometry_checks)
    if len(runs) != 56:
        raise ValueError('Expected 14 conditions times four models')
    base.atomic_json(stage/'plan.json', plan)
    shutil.copyfile(ROOT/'scripts/adni_constraints/run.sbatch', stage/'run.sbatch')
    manifest = {str(f.relative_to(stage)):base.digest(f) for f in sorted(stage.rglob('*')) if f.is_file()}
    base.atomic_json(stage/'manifest.json', manifest)
    base.atomic_json(report/'prepared_plan.json', plan)
    with tarfile.open(report/'submission_bundle.tar.gz', 'w:gz') as archive:
        for f in sorted(stage.rglob('*')):
            if f.is_file():
                archive.add(f, arcname=str(f.relative_to(stage)), recursive=False)
    print(json.dumps(dict(stage=str(stage), runs=len(runs), augmentation='none',
                         manifest_sha256=base.digest(stage/'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
