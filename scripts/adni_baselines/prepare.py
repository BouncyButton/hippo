"""Freeze one four-model ADNI pilot, including subject selection and profiles."""
import json
from pathlib import Path
import shutil
import tarfile

import pandas as pd

from .core import BUDGET, MODELS, ROOT, PreparedADNI, base, select_split

NAME = 'adni_baselines_5pct_fold0_seed0_20261013'


def main():
    stage, report = ROOT / 'experiments' / NAME, ROOT / 'reports' / NAME
    if stage.exists():
        raise FileExistsError('Refusing to overwrite a frozen campaign')
    data = ROOT / 'datasets/Dataset103_ADNI'
    pkl = data / 'adni_hippocampus_full.pkl'
    df = pd.read_pickle(pkl, compression='gzip')
    split, provenance = select_split(df, json.loads((data / 'splits_final.json').read_text()))
    # Validate every input and preserve foreground during the shared padding.
    train, val = PreparedADNI(df, split['train']), PreparedADNI(df, split['val'])
    source = stage / 'source'
    paths = ['scripts/adni_baselines', 'scripts/train_compute_matched_baselines.py',
             'baselines/simple_segmentation.py', 'baselines/swin_unetr/swin_unetr.py',
             'baselines/nnUNet/nnunetv2', 'thesis/new_constraints/training_augmentation.py',
             'reports/compute_budget_5pct_fold0_seed0_20261003/profile_reference.py',
             'reports/compute_budget_5pct_fold0_seed0_20261003/profile_matched_models.py']
    for name in paths:
        src, dest = ROOT / name, source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copyfile(src, dest)
    architecture = json.loads((ROOT / 'experiments/compute_matched_baselines_20261003/nnunet_architecture.json').read_text())
    architecture['kwargs']['num_classes'] = 2
    architecture['spacing_assumption'] = None
    architecture['plan_origin'] = 'Frozen MSD matched-input topology, adapted to two output labels; not a new ADNI automatic plan'
    previous = ROOT / 'reports/capped_replication_31976_20261007/verified_metadata_20261010/frozen/source/experiments/capped_replication_31976_20261007'
    runs = []
    for identity, spec in MODELS.items():
        profile_path = report / 'profiles' / (identity + '.json')
        profile = json.loads(profile_path.read_text())
        if profile['model'] != identity or profile['classes'] != 2 or not profile['next_update_exceeds_budget']:
            raise ValueError('Binary profile identity mismatch')
        alias = spec['alias']
        parent = json.loads((previous / identity / 'protocol.json').read_text())
        if parent['budget_flops'] != BUDGET or parent['lr_horizon_updates'] != spec['horizon'] or parent['initial_lrs'][alias] != spec['lr']:
            raise ValueError('Historical MSD schedule differs')
        directory = stage / 'protocols' / identity
        directory.mkdir(parents=True)
        base.atomic_json(directory / 'splits_10patients.json', [split])
        base.atomic_json(directory / 'nnunet_architecture.json', architecture)
        shutil.copyfile(profile_path, directory / 'profile.json')
        protocol = {key:parent[key] for key in (
            'batch_size','spatial_size','required_versions','augmentation','precision',
            'adamw_weight_decay','sgd_weight_decay','nnunet_sgd_momentum','nnunet_warmup_fraction')}
        protocol.update(name=NAME + '_' + identity, model_identity=identity, num_classes=2,
            pkl_sha256=base.digest(pkl), splits_sha256=base.digest(directory / 'splits_10patients.json'),
            nnunet_architecture_sha256=base.digest(directory / 'nnunet_architecture.json'),
            profile_sha256=base.digest(directory / 'profile.json'),
            models={alias:dict(parameters=profile['parameters'], flops_per_update=profile['flops_per_update'])},
            initial_lrs={alias:spec['lr']}, lr_horizon_updates=spec['horizon'],
            budget_flops=BUDGET, folds=[0], seeds=[0], validation_milestones=20,
            validation_target_flops=[i * BUDGET // 20 for i in range(1,21)],
            checkpoint_selection='Maximum validation whole-hippocampus hard Dice; also report final-budget Dice',
            early_stopping=False, pretrained_weights=False, constraints=False,
            parent_protocol_sha256=base.digest(previous / identity / 'protocol.json'),
            learning_rate_policy='Frozen MSD trajectory: original warmup/cosine or nnU-Net polynomial, with unchanged horizon',
            preprocessing='Per-crop nonzero z-score; symmetric zero padding to 64 cubed; no resizing',
            geometry='Native 32x48x40 array grid; source physical geometry unavailable; no millimetre metrics',
            label_budget='5 of 108 training subjects (4.63%), both hemispheres; identical across models')
        base.atomic_json(directory / 'protocol.json', protocol)
        runs.append(dict(model=identity, alias=alias, updates=profile['max_steps'],
                         gflops_per_update=profile['flops_per_update']/1e9,
                         parameters=profile['parameters'], initial_lr=spec['lr']))
    provenance.update(dataset_sha256=base.digest(pkl), saved_folds_sha256=base.digest(data / 'splits_final.json'))
    base.atomic_json(stage / 'split_provenance.json', provenance)
    references = json.loads((ROOT / 'reports/capped_replication_31976_20261007/final_summary_20261010.json').read_text())
    refs = [{k:r[k] for k in ('model','updates','best_dice','final_dice','completion_sha256')}
            for r in references['runs'] + references['reused_runs']
            if r['objective']=='dice' and r['fold']==r['seed']==0]
    plan = dict(name=NAME, dataset='ADNI', fold=0, seed=0, requested_fraction=.05,
                train_subjects=5, train_crops=len(train), val_subjects=27, val_crops=len(val),
                budget_flops_per_model=BUDGET, runs=runs, num_runs=4, sequential=True,
                runtime_versions=dict(torch='2.12.1+cu130',monai='1.6.0'),
                pkl_sha256=base.digest(pkl), pkl_bytes=pkl.stat().st_size,
                remote_pkl='/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl',
                msd_references=refs,
                comparison='ADNI whole-hippocampus Dice is a different task from MSD anterior/posterior macro Dice; compare trajectories without treating absolute scores as equivalent',
                budget_exclusions='Validation, preprocessing, initialization, checkpoint I/O, offline profiling',
                metrics=['training loss','validation hard Dice','validation soft Dice','best and final Dice','per-crop Dice'],
                checkpoint_retention='Keep best and latest checkpoints for all four models')
    base.atomic_json(stage / 'plan.json', plan)
    shutil.copyfile(ROOT / 'scripts/adni_baselines/run.sbatch', stage / 'run.sbatch')
    manifest = {str(f.relative_to(stage)):base.digest(f) for f in sorted(stage.rglob('*')) if f.is_file()}
    base.atomic_json(stage / 'manifest.json', manifest)
    base.atomic_json(report / 'preflight.json', dict(status='passed', checks=[
        'True subject-grouped fold-0 selection; both sides retained',
        'All 64 selected/validation crops finite and binary; foreground preserved during padding',
        'Real CPU forward/backward/optimizer step for all four binary architectures',
        'Binary per-update arithmetic profiled with original MSD runtime and counter'],
        manifest_sha256=base.digest(stage/'manifest.json'), runs=runs))
    with tarfile.open(report/'submission_bundle.tar.gz','w:gz') as archive:
        for f in sorted(stage.rglob('*')):
            if f.is_file():
                archive.add(f,arcname=str(f.relative_to(stage)),recursive=False)
    print(json.dumps(dict(stage=str(stage), runs=runs, data_crops=[len(train),len(val)])),flush=True)


if __name__ == '__main__':
    main()
