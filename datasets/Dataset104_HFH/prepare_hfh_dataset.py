"""Convert the official HFH Analyze release to native-grid binary NIfTI data.

Extract HFH.rar into Dataset104_HFH/raw first. No resampling or anatomical
registration is performed. Analyze affines use NiBabel's header conventions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def volume(image):
    array = np.asanyarray(image.dataobj)
    if array.ndim == 4 and array.shape[3] == 1:
        array = array[..., 0]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError('Expected a finite 3D volume or singleton fourth axis')
    return array


def save_verified(array, affine, path):
    image = nib.Nifti1Image(array, affine)
    # Preserve the Analyze-derived geometry without asserting scanner alignment.
    image.set_sform(affine, code=2)
    image.set_qform(affine, code=0)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    saved = nib.load(path)
    if not np.array_equal(array, np.asanyarray(saved.dataobj)):
        raise ValueError(f'Voxel mismatch after conversion: {path}')
    if not np.allclose(affine, saved.affine, atol=1e-5):
        raise ValueError(f'Affine mismatch after conversion: {path}')


def prepare(target, report):
    report.mkdir(parents=True, exist_ok=True)
    for name in ['imagesTr', 'labelsTr', 'imagesTs']:
        (target / name).mkdir(exist_ok=True)
    rows = []
    for partition, output in [('Train', 'imagesTr'), ('Test', 'imagesTs')]:
        paths = sorted((target / 'raw' / partition).glob('HFH_*.hdr'))
        if len(paths) != 25:
            raise ValueError(f'Expected 25 {partition} scans, found {len(paths)}')
        for path in paths:
            image = nib.load(path)
            array = volume(image)
            case = 'hippocampus_hfh_' + path.stem.split('_')[1]
            destination = target / output / f'{case}_0000.nii.gz'
            save_verified(array, image.affine, destination)
            row = dict(case=case, official_partition=partition,
                       original_shape=list(image.shape), shape=list(array.shape),
                       spacing_mm=[round(float(v), 6) for v in image.header.get_zooms()[:3]],
                       dtype=str(array.dtype), image=str(destination.relative_to(target)),
                       image_sha256=digest(destination),
                       source_sha256={str(p.relative_to(target)): digest(p)
                                      for p in [path, path.with_suffix('.img')]},
                       affine=image.affine.tolist(), source_format=type(image).__name__)
            if partition == 'Train':
                label_path = path.parent / 'Labels' / f'{path.stem}_Hipp_Labels.hdr'
                label = nib.load(label_path)
                raw_label = volume(label)
                if raw_label.shape != array.shape or not np.allclose(label.affine, image.affine):
                    raise ValueError(f'Image/label geometry mismatch: {case}')
                values, counts = np.unique(raw_label, return_counts=True)
                if set(values) != {0, 1, 2}:
                    raise ValueError(f'Unexpected label values: {case}: {values}')
                binary = (raw_label > 0).astype(np.uint8)
                label_output = target / 'labelsTr' / f'{case}.nii.gz'
                save_verified(binary, image.affine, label_output)
                row.update(label=str(label_output.relative_to(target)),
                           label_sha256=digest(label_output),
                           source_label_counts={str(int(v)): int(n) for v, n in zip(values, counts)},
                           hippocampus_volume_mm3=float(binary.sum() * np.prod(image.header.get_zooms()[:3])))
                row['source_sha256'].update({str(p.relative_to(target)): digest(p)
                                            for p in [label_path, label_path.with_suffix('.img')]})
            rows.append(row)
            print(f'Verified {case} ({partition})', flush=True)
    # Official test subjects remain separate. CV uses only the 25 labelled subjects.
    train = [r for r in rows if r['official_partition'] == 'Train']
    test = [r['case'] for r in rows if r['official_partition'] == 'Test']
    rng = np.random.default_rng(42)
    strata = [[r['case'] for r in train if (r['shape'][0] == 512) == high]
              for high in [False, True]]
    folds = [[] for _ in range(5)]
    for stratum in strata:
        rng.shuffle(stratum)
        for index, case in enumerate(stratum):
            folds[index % 5].append(case)
    pool = {r['case'] for r in train}
    splits = [dict(train=sorted(pool - set(val)), val=sorted(val)) for val in folds]
    assert sorted(c for split in splits for c in split['val']) == sorted(pool)
    assert not pool.intersection(test)
    assert all(len(s['train']) == 20 and len(s['val']) == 5 and
               not set(s['train']).intersection(s['val']) for s in splits)
    write_json(target / 'dataset.json', dict(name='HFH', channel_names={'0': 'T1 MRI'},
               labels={'background': 0, 'hippocampus': 1}, numTraining=25,
               numTest=25, file_ending='.nii.gz'))
    write_json(target / 'splits_final.json', splits)
    write_json(target / 'split_provenance.json', dict(seed=42, rng='numpy.default_rng',
               scheme='Five-fold subject CV, stratified by 256/512 acquisition matrix',
               source_train_subjects=sorted(pool), official_unlabelled_test_subjects=test,
               training_subjects_per_fold=20, validation_subjects_per_fold=5,
               test_labels_available=False))
    counts = Counter(tuple(r['spacing_mm']) for r in rows)
    archive = target / 'downloads/HFH.rar'
    summary = dict(source_url='https://www.nitrc.org/frs/download.php/3030/HFH.rar',
                   archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
                   subjects=50, labelled_subjects=25, unlabelled_test_subjects=25,
                   source_label_values=[0, 1, 2], prepared_labels={'0': 'background', '1': 'hippocampus union'},
                   resolution_counts=[dict(spacing_mm=list(k), subjects=v) for k, v in sorted(counts.items())],
                   conversion='Remove singleton fourth axis only; native voxel values and affines verified after saving; labels 1 and 2 merged to binary foreground.',
                   orientation_caveat='Analyze files do not establish scanner-world orientation; NiBabel-derived affines are preserved, not independently verified anatomical laterality or registration.',
                   verification='All 50 images and 25 binary labels reloaded and compared voxel-for-voxel; all 25 source image/label grids match; CV has no subject overlap.',
                   cases=rows)
    write_json(report / 'inventory.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[2] / 'reports/hfh_dataset_20261111')
    args = parser.parse_args()
    prepare(args.target, args.report)


if __name__ == '__main__':
    main()
