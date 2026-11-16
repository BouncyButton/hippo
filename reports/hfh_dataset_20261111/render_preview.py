"""Render actual HFH MRI and original annotations at physical aspect ratio."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets/Dataset104_HFH'


def main():
    candidates = sorted((DATA / 'raw/Train').glob('HFH_*.hdr'))
    selected = [next(p for p in candidates if nib.load(p).shape[0] == size)
                for size in [256, 512]]
    preview = []
    for zoom in [False, True]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 9), facecolor='#101724')
        for row, path in enumerate(selected):
            source = nib.squeeze_image(nib.load(path))
            label_source = nib.squeeze_image(nib.load(path.parent / 'Labels' / f'{path.stem}_Hipp_Labels.hdr'))
            image = nib.as_closest_canonical(source)
            labels = nib.as_closest_canonical(label_source)
            img = image.get_fdata(dtype=np.float32)
            lab = np.asanyarray(labels.dataobj)
            spacing = np.array(image.header.get_zooms()[:3])
            coordinates = np.argwhere(lab > 0)
            # Center on the labelled anatomy, using each plane's largest section.
            slices = [int(np.argmax((lab > 0).sum(axis=tuple(j for j in range(3) if j != axis))))
                      for axis in [1, 0, 2]]
            vmin, vmax = np.percentile(img[img > 0], [1, 99.7])
            low, high = np.zeros(3, dtype=int), np.array(img.shape)
            if zoom:
                pad = np.ceil(12 / spacing).astype(int)
                low = np.maximum(coordinates.min(axis=0) - pad, 0)
                high = np.minimum(coordinates.max(axis=0) + pad + 1, img.shape)
                region = img[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
                vmin, vmax = np.percentile(region[region > 0], [1, 99.7])
            for col, (axis, plane, index) in enumerate(zip([1, 0, 2], ['Coronal', 'Sagittal', 'Axial'], slices)):
                window = [slice(int(a), int(b)) for a, b in zip(low, high)]
                window[axis] = index
                im2, la2 = img[tuple(window)].T, lab[tuple(window)].T
                remaining = [j for j in range(3) if j != axis]
                extent = [0, im2.shape[1] * spacing[remaining[0]], 0, im2.shape[0] * spacing[remaining[1]]]
                ax = axes[row, col]
                ax.set_facecolor('#000000')
                ax.imshow(im2, origin='lower', cmap='gray', extent=extent, vmin=vmin, vmax=vmax, interpolation='nearest')
                if zoom:
                    ax.imshow(np.ma.masked_equal(la2, 0), origin='lower', extent=extent,
                              cmap=ListedColormap(['black', '#42d9da', '#ffb45b']),
                              norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), alpha=.65, interpolation='nearest')
                ax.set_aspect('equal')
                ax.axis('off')
                title = f'{path.stem} · {plane}'
                if col == 0:
                    title += '\n' + ' × '.join(f'{v:.3g}' for v in source.header.get_zooms()[:3]) + ' mm'
                ax.set_title(title, color='#f1f5f9', fontsize=12)
            if not zoom:
                preview.append(dict(subject=path.stem, shape=list(source.shape),
                                    spacing_mm=list(map(float, source.header.get_zooms()[:3])),
                                    canonical_slice_indices=slices))
        fig.suptitle('HFH · manual hippocampus annotations' if zoom else 'HFH · original T1-weighted MRI',
                     color='white', fontsize=21, y=.99)
        fig.text(.5, .945, 'Two acquisition resolutions · native voxels · panels scaled in millimetres',
                 color='#b8c5d6', fontsize=11, ha='center')
        if zoom:
            legend = fig.legend([Patch(color='#42d9da'), Patch(color='#ffb45b')],
                                ['Original label 1', 'Original label 2'], loc='lower center',
                                bbox_to_anchor=(.5, .025), ncol=2, frameon=False)
            for text in legend.get_texts():
                text.set_color('white')
        fig.text(.5, .012, 'Views follow Analyze header conventions; anatomical left/right has not been independently verified.',
                 color='#99a9bd', fontsize=9, ha='center')
        fig.subplots_adjust(top=.85, bottom=.12, hspace=.38, wspace=.12)
        filename = 'hfh_labels_preview.png' if zoom else 'hfh_mri_preview.png'
        fig.savefig(OUT / filename, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
    (OUT / 'preview_provenance.json').write_text(json.dumps(preview, indent=2) + '\n')


if __name__ == '__main__':
    main()
