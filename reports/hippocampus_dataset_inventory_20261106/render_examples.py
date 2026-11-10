"""Audit local pairs/splits and render real MRI/annotation examples."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COLORS = ['#000000', '#ffb547', '#46c4cf', '#e878a7', '#a998ff', '#9cde70']


def render(name, case, class_names, title):
    folder = ROOT / 'datasets' / name
    image_path = folder / 'imagesTr' / f'{case}_0000.nii.gz'
    label_path = folder / 'labelsTr' / f'{case}.nii.gz'
    image = nib.as_closest_canonical(nib.load(image_path))
    label = nib.as_closest_canonical(nib.load(label_path))
    assert image.shape == label.shape and np.allclose(image.affine, label.affine)
    img = image.get_fdata(dtype=np.float32)
    lab = label.get_fdata(dtype=np.float32).astype(np.uint8)
    coords = np.argwhere(lab > 0)
    low = np.maximum(coords.min(axis=0) - 7, 0)
    high = np.minimum(coords.max(axis=0) + 8, lab.shape)
    window = tuple(slice(a, b) for a, b in zip(low, high))
    img, lab = img[window], lab[window]
    vmin, vmax = np.percentile(img[img > 0], [1, 99])
    cmap = ListedColormap(COLORS[:len(class_names) + 1])
    norm = BoundaryNorm(np.arange(len(class_names) + 2) - .5, cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor='#101724')
    slices = []
    for column, (axis, plane) in enumerate([(1, 'Coronal'), (0, 'Sagittal'), (2, 'Axial')]):
        # Use the slice showing the greatest annotated cross-sectional area.
        index = int(np.argmax((lab > 0).sum(axis=tuple(i for i in range(3) if i != axis))))
        slices.append({'plane': plane, 'canonical_index': int(index + low[axis])})
        im2 = np.take(img, index, axis=axis).T
        la2 = np.take(lab, index, axis=axis).T
        for row in range(2):
            ax = axes[row, column]
            ax.imshow(im2, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
            if row:
                ax.imshow(np.ma.masked_equal(la2, 0), cmap=cmap, norm=norm,
                          origin='lower', alpha=.65, interpolation='nearest')
            ax.set_axis_off()
            if row == 0:
                ax.set_title(plane, color='#dce5f2', fontsize=12, pad=9)
    fig.suptitle(title, color='white', fontsize=19, y=.99)
    fig.text(.5, .935, f'{case}  |  top: MRI  ·  bottom: manual labels over MRI',
             color='#b4c2d5', ha='center', fontsize=10)
    legend = fig.legend([Patch(facecolor=COLORS[i]) for i in class_names],
                       [f'{i}  {s}' for i, s in class_names.items()],
                       loc='lower center', ncol=min(3, len(class_names)), frameon=False,
                       bbox_to_anchor=(.5, -.005), fontsize=10)
    for text in legend.get_texts():
        text.set_color('white')
    fig.subplots_adjust(top=.87, bottom=.12, hspace=.07, wspace=.03)
    fig.savefig(OUT / f'{name}_example.png', dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return {'image': str(image_path), 'label': str(label_path), 'slices': slices}


def audit(name):
    folder = ROOT / 'datasets' / name
    metadata = json.loads((folder / 'dataset.json').read_text())
    labels = sorted((folder / 'labelsTr').glob('*.nii.gz'))
    cases, shapes, values = set(), set(), set()
    for path in labels:
        case = path.name[:-7]
        image = nib.load(folder / 'imagesTr' / f'{case}_0000.nii.gz')
        label = nib.load(path)
        assert image.shape == label.shape
        assert np.allclose(image.affine, label.affine)
        vals = set(np.unique(np.asarray(label.dataobj)).astype(int).tolist())
        assert vals <= set(metadata['labels'].values())
        shapes.add(image.shape)
        values.update(vals)
        cases.add(case)
    folds = json.loads((folder / 'splits_final.json').read_text())
    results = []
    validation_cases = []
    for index, fold in enumerate(folds):
        train, val = set(fold['train']), set(fold['val'])
        train_subjects = {case.rsplit('_', 1)[0] for case in train}
        val_subjects = {case.rsplit('_', 1)[0] for case in val}
        assert not train & val and not train_subjects & val_subjects
        assert train | val == cases
        validation_cases.extend(val)
        results.append({'fold': index, 'train_cases': len(train), 'val_cases': len(val),
                        'train_subjects': len(train_subjects), 'val_subjects': len(val_subjects),
                        'validation_subject_ids': sorted(val_subjects)})
    assert len(validation_cases) == len(set(validation_cases)) == len(cases)
    return {'metadata': metadata, 'verified_pairs': len(cases), 'shapes': sorted(shapes),
            'observed_label_values': sorted(values), 'folds': results, 'test_split': None}


def verify_cobra_source_crops():
    """Compare saved voxel values and locations with the original MINC grids."""
    folder = ROOT / 'datasets' / 'Dataset105_COBRA'
    mapping = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5,
               101: 1, 102: 2, 104: 3, 105: 4, 106: 5}
    verified = []
    for subject in range(1, 6):
        for side in ['L', 'R']:
            case = f'hippocampus_cobra_{subject}_{side}'
            for kind in ['image', 'label']:
                if kind == 'image':
                    raw = nib.load(folder / 'raw' / 'brains_t2' / f'brain{subject}_t2.mnc')
                    output = nib.load(folder / 'imagesTr' / f'{case}_0000.nii.gz')
                else:
                    raw = nib.load(folder / 'raw' / 'labels' / f'brain{subject}_labels.mnc')
                    output = nib.load(folder / 'labelsTr' / f'{case}.nii.gz')
                transform = np.linalg.inv(raw.affine) @ output.affine
                assert np.allclose(transform[:3, :3], 2 * np.eye(3), atol=1e-4)
                start = np.rint(transform[:3, 3]).astype(int)
                assert np.allclose(transform[:3, 3], start, atol=1e-4)
                end = start + 2 * np.asarray(output.shape)
                assert np.all(start >= 0) and np.all(end <= raw.shape)
                expected = np.asarray(raw.dataobj[tuple(slice(a, b, 2) for a, b in zip(start, end))], dtype=np.float32)
                if kind == 'label':
                    mapped = np.zeros_like(expected)
                    for old, new in mapping.items():
                        mapped[expected == old] = new
                    expected = mapped
                assert np.allclose(expected, np.asarray(output.dataobj), atol=1e-5)
            verified.append(case)
    return verified


if __name__ == '__main__':
    configurations = [
        ('Dataset102_MNI', 'hippocampus_mni_s01_L',
         {1: 'CA1–3', 2: 'Subiculum', 3: 'CA4 / dentate gyrus'}, 'MNI / HiSub25 · left hippocampus · T1 MRI'),
        ('Dataset105_COBRA', 'hippocampus_cobra_1_L',
         {1: 'CA1', 2: 'Subiculum', 3: 'CA4 / dentate gyrus', 4: 'CA2–3', 5: 'SR / SL / SM'},
         'COBRA / Winterburn · left hippocampus · T2 MRI'),
    ]
    output = {}
    for name, case, classes, title in configurations:
        if not (ROOT / 'datasets' / name / 'imagesTr' / f'{case}_0000.nii.gz').exists():
            continue
        output[name] = audit(name)
        if name == 'Dataset105_COBRA':
            output[name]['source_crops_verified'] = verify_cobra_source_crops()
        output[name]['example'] = render(name, case, classes, title)
        print(name, output[name]['verified_pairs'], 'pairs verified', flush=True)
    (OUT / 'inventory.json').write_text(json.dumps(output, indent=2) + '\n')
