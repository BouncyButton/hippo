"""Freeze data, subject-disjoint splits, measured schedules and execution source."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.prepare import freeze_source
from scripts.unetr_subfields.contracts import ROOT, NAME, DATASETS, CAP, FOLDS, SEEDS, tasks, subject, baseline_schedule


def select_splits(folds, spec):
    result, provenance = [], []
    for index in FOLDS:
        saved = folds[index]
        pool = sorted({subject(case) for case in saved['train']})
        validation = sorted({subject(case) for case in saved['val']})
        if set(pool) & set(validation):
            raise ValueError('Subject leakage in original folds')
        chosen = sorted(np.random.RandomState(0).permutation(pool)[:spec['train_subjects']].tolist())
        train = sorted(case for case in saved['train'] if subject(case) in chosen)
        val = sorted(saved['val'])
        if len(chosen) != spec['train_subjects'] or len(train) != 2 * len(chosen):
            raise ValueError('Training support size differs')
        for participant in chosen + validation:
            sides = [case.rsplit('_', 1)[1] for case in train + val if subject(case) == participant]
            if sorted(sides) != ['L', 'R']:
                raise ValueError('Both hemispheres must stay together')
        result.append(dict(train=train, val=val))
        provenance.append(dict(fold=index, subset_seed=0, full_training_subjects=pool,
                               training_subjects=chosen, validation_subjects=validation))
    return result, provenance


def prepare_data(stage, dataset, spec):
    folder = ROOT / 'datasets' / spec['folder']
    records, sources, geometry = [], {}, {}
    for label_path in sorted((folder / 'labelsTr').glob('*.nii.gz')):
        case = label_path.name[:-7]
        image_path = folder / 'imagesTr' / f'{case}_0000.nii.gz'
        image, label = nib.load(image_path), nib.load(label_path)
        if image.shape != tuple(spec['native_shape']) or image.shape != label.shape or not np.allclose(image.affine, label.affine):
            raise ValueError(f'Unaligned source: {case}')
        x, y = image.get_fdata(dtype=np.float32), label.get_fdata(dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.equal(y, np.rint(y)).all():
            raise ValueError('Invalid source values')
        if set(np.unique(y)) != set(range(len(spec['classes']))):
            raise ValueError(f'Unexpected or absent class in {case}')
        records.append(dict(subject_id=case, participant_id=subject(case),
                            image_data=x, label_data=y.astype(np.uint8)))
        sources[case] = dict(image=str(image_path.relative_to(ROOT)), image_sha256=digest(image_path),
                             label=str(label_path.relative_to(ROOT)), label_sha256=digest(label_path))
        pad_before = (np.asarray(spec['spatial_size']) - image.shape) // 2
        affine = image.affine.copy()
        affine[:3, 3] = nib.affines.apply_affine(image.affine, -pad_before)
        geometry[case] = dict(model_affine=affine.tolist(), padding_before=pad_before.tolist())
    expected = 50 if dataset == 'MNI' else 10
    if len(records) != expected:
        raise ValueError('Incomplete source dataset')
    data = stage / 'data' / f'{dataset}.pkl'
    data.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_pickle(data, compression=dict(method='gzip', mtime=0))
    write(stage / 'data' / f'{dataset}_sources.json', sources)
    return data, geometry, {row['subject_id'] for row in records}


def main():
    stage = ROOT / 'experiments' / NAME
    if (stage / 'manifest.json').exists():
        from scripts.unetr_subfields.job import verify
        verify(stage)
        print('Existing frozen campaign verified:', stage)
        return
    stage.mkdir(parents=True, exist_ok=True)
    remote = REMOTE / NAME
    parent = ROOT / 'experiments/adni_constraints_noaug_5pct_fold0_seed0_20261013/protocols/unetr_vit_b16/protocol.json'
    template = read(parent)
    for dataset, spec in DATASETS.items():
        data, geometry, cases = prepare_data(stage, dataset, spec)
        source_split = ROOT / 'datasets' / spec['folder'] / 'splits_final.json'
        splits, provenance = select_splits(read(source_split), spec)
        if any(not set(split['train'] + split['val']) <= cases for split in splits):
            raise ValueError('Split references missing case')
        folder = stage / 'protocols' / dataset
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / 'splits_10patients.json', splits)
        write(folder / 'split_provenance.json', dict(
            original_split_sha256=digest(source_split), folds=provenance,
            selection='First 3 of existing 5 subject folds; support selected with RandomState(0), independent of model seed.',
            validation_coverage='Partial five-fold CV: only folds 0, 1, 2; not a new full three-fold partition.'))
        write(folder / 'nnunet_architecture.json', {'purpose': 'Compatibility artifact only; UNETR construction uses its explicit kwargs.'})
        profile_path = ROOT / 'reports' / NAME / 'profiles' / f'{dataset}.json'
        profile = read(profile_path)
        for name, sha in profile['source_sha256'].items():
            if digest(ROOT / name) != sha:
                raise ValueError('Profiled source changed; reprofile before freezing')
        if profile['kwargs']['out_channels'] != len(spec['classes']) or profile['input_shape'][2:] != list(spec['spatial_size']):
            raise ValueError('Profile does not match new task')
        shutil.copyfile(profile_path, folder / 'profile.json')
        count = spec['train_subjects'] * 2
        schedule = baseline_schedule(profile['update_flops'], count)
        protocol = deepcopy(template)
        for key in ['sdf_policy', 'depth_geometry', 'constraint_geometry', 'calibration_policy', 'native_geometry']:
            protocol.pop(key, None)
        protocol.update(name=NAME, dataset=dataset, model_identity='unetr_vit_b16', trainer_alias='unet',
            classes=spec['classes'], num_classes=len(spec['classes']), spatial_size=list(spec['spatial_size']),
            data=str(remote / 'data' / data.name), pkl_sha256=digest(data),
            splits_sha256=digest(folder / 'splits_10patients.json'),
            nnunet_architecture_sha256=digest(folder / 'nnunet_architecture.json'),
            profile_sha256=digest(folder / 'profile.json'),
            models={'unet': dict(parameters=profile['parameters'], objectives={'dice': schedule})},
            train_count=count, folds=list(FOLDS), seeds=list(SEEDS), constraints=False,
            requested_arm_order=['dice'], budget_flops=CAP, budget_scope='optimization_only',
            budget_origin='Unchanged historical medical cap: 90 nnUNet updates = 9 passes over 10 crops, at 355293896301 FLOPs/update. Not recomputed from the new fold sizes.',
            checkpoint_selection='Final FLOP-budget checkpoint primary; best validation checkpoint secondary.',
            primary_metric='Mean across volumes of foreground-subfield macro hard Dice; average the 9 run scores equally.',
            secondary_metrics=['per-subfield Dice/TP/FN/FP', 'whole-hippocampus union Dice', 'integer confusion matrices', 'training Dice', 'best validation Dice'],
            decision_rule='Report mean, sample SD, range, all nine scores and fraction >=0.50. Around 0.50 is descriptive [0.45,0.55]; no seed/fold replacement or adaptive extension to hit a target.',
            native_geometry=geometry, augmentation='none', early_stopping=False,
            label_budget=f"{spec['train_subjects']} training subjects / {count} crops in every fold",
            preprocessing='Per-crop nonzero z-score and symmetric zero padding only; no resize/crop/augmentation; every foreground voxel retained.',
            geometry='Processed source grids retained; no millimetre accuracy metrics.',
            learning_rate_policy='Historical UNETR AdamW 1e-4, weight decay 1e-5, 477-update warmup/cosine horizon retained; no tuning.',
            constraint_warmup_updates=1, calibration_policy='None: baseline only.',
            checkpoint_retention='Archive and verify metrics/configuration/source/logs outside BeeGFS after each run, then retire best/latest checkpoints.',
            training_loss='Historical softmax Dice including background plus connected-zero term; no auxiliary objectives.')
        write(folder / 'protocol.json', protocol)
    plan = dict(name=NAME, tasks=tasks(), num_runs=18, augmentation='none',
                budget_flops_by_dataset={name: CAP for name in DATASETS},
                conventional_python=str(REMOTE / '.conda/envs/hippocampus/bin/python'),
                max_job_hours=6, graceful_stop_seconds=6 * 3600 - 900,
                archive_wait_timeout_seconds=300,
                storage=dict(minimum_free_gib=12, estimated_peak_logical_gib=5, mirrored_peak_gib=10,
                             reserve_gib=2, archive=str(ROOT / 'reports' / NAME),
                             policy='One run at a time; verify archive before retiring bulky outputs.'),
                scope='Initial 18 UNETR-only runs. Other three architectures are follow-up scope, not included in this submission.',
                authorization='User confirmed MNI 5 training subjects, COBRA all 4, existing folds0/1/2 and seeds0/1/2; usual historical FLOP cap; continue.',
                submission_status='Prepared; cluster queue, quota and runtime preflight required before submission.')
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False, disabled_tasks=[]))
    write(stage / 'source_version.json', dict(
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_worktree=True, source_of_truth='Frozen source manifest and exact local prepared-data hashes.'))
    freeze_source(stage)
    shutil.copytree(ROOT / 'scripts/unetr_subfields', stage / 'source/scripts/unetr_subfields',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
    write(stage / 'source/source_manifest.json', {
        str(p.relative_to(stage / 'source')): digest(p) for p in sorted((stage / 'source').rglob('*'))
        if p.is_file() and p.name != 'source_manifest.json'})
    shutil.copyfile(ROOT / 'scripts/unetr_subfields/run.sbatch', stage / 'run.sbatch')
    write(stage / 'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
                                 if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    from scripts.unetr_subfields.job import verify
    verify(stage)
    print(json.dumps(dict(stage=str(stage), runs=18, schedules={
        dataset: read(stage / 'protocols' / dataset / 'protocol.json')['models']['unet']['objectives']['dice']
        for dataset in DATASETS}), indent=2))


if __name__ == '__main__':
    main()
