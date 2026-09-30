"""Label-guided, linked-plane inspection of the last anterior MSD voxels.

Descriptive anatomy discovery, explicitly unblinded. The selected interface
voxel is an annotation anchor, NOT an identified uncal-apex landmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-uncal-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AXES = ("Sagittal", "Coronal", "Axial")
DIMENSIONS = ((1, 2), (0, 2), (0, 1))


def interface_voxels(labels):
    """Anterior voxels immediately adjacent to posterior along native RAS y."""
    face = np.zeros(labels.shape, dtype=bool)
    face[:, 1:, :] = (labels[:, 1:, :] == 1) & (labels[:, :-1, :] == 2)
    points = np.argwhere(face)
    if not len(points):
        raise ValueError("No adjacent A/P face found")
    return face, points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["001", "274", "345", "321"])
    parser.add_argument("--anchor", nargs=3, type=int, metavar=("X", "Y", "Z"),
                        help="Explicit anterior voxel near the interface; requires one case")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/uncal_interface_inspection_20260921")
    args = parser.parse_args()
    if args.anchor is not None and len(args.cases) != 1:
        parser.error("--anchor requires exactly one case")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "datasets/Dataset101_MSD"
    records = []
    for case in args.cases:
        name = case if case.startswith("hippocampus_") else f"hippocampus_{case.zfill(3)}"
        ni = nib.load(dataset / "imagesTr" / f"{name}_0000.nii.gz")
        nl = nib.load(dataset / "labelsTr" / f"{name}.nii.gz")
        if not np.allclose(ni.affine, nl.affine) or not np.allclose(ni.affine[:3, :3], np.eye(3)):
            raise ValueError("Expected aligned native 1-mm RAS arrays")
        image, labels = np.asarray(ni.dataobj), np.asarray(nl.dataobj)
        face, points = interface_voxels(labels)
        centre = points.mean(axis=0)
        anchor = points[np.argmin(((points-centre)**2).sum(axis=1))]
        if args.anchor is not None:
            anchor = np.asarray(args.anchor)
            if np.any(anchor < 0) or np.any(anchor >= np.asarray(labels.shape)) or labels[tuple(anchor)] != 1:
                raise ValueError("Explicit anchor must be a labelled anterior voxel")
        assert labels[tuple(anchor)] == 1
        if args.anchor is None:
            assert labels[anchor[0], anchor[1]-1, anchor[2]] == 2
        low, high = np.percentile(image, (1, 99))
        cmap = ListedColormap(["black", "#ee5863", "#568cdd"])

        def show(ax, axis, index, overlay=False, crosshair=False):
            ax.imshow(np.take(image, index, axis=axis).T, origin="lower", cmap="gray",
                      vmin=low, vmax=high, interpolation="nearest")
            if overlay:
                mask = np.take(labels, index, axis=axis).T
                ax.imshow(np.ma.masked_equal(mask, 0), origin="lower", cmap=cmap,
                          vmin=0, vmax=2, alpha=.36, interpolation="nearest")
                surface = np.take(face, index, axis=axis).T
                if surface.any():
                    ax.contour(surface, levels=[.5], colors=["#ffe36b"], linewidths=.8)
            h, v = DIMENSIONS[axis]
            if crosshair:
                ax.axvline(anchor[h], color="#35e0d1", linewidth=.65, alpha=.75)
                ax.axhline(anchor[v], color="#35e0d1", linewidth=.65, alpha=.75)
            ax.set_title(f"{AXES[axis]} · {'xyz'[axis]}={index}" + (" · labels" if overlay else " · raw"), fontsize=10)
            ax.set_xlabel(f"{'xyz'[h]} native voxel index", fontsize=8)
            ax.set_ylabel(f"{'xyz'[v]} native voxel index", fontsize=8)
            ax.tick_params(labelsize=7)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
        for axis in range(3):
            show(axes[0, axis], axis, int(anchor[axis]), crosshair=True)
            show(axes[1, axis], axis, int(anchor[axis]), overlay=True, crosshair=True)
        fig.suptitle(f"{name} · same anterior voxel {tuple(anchor.tolist())}\nRed = anterior; blue = posterior; yellow = anterior face touching posterior. Anchor ≠ verified apex.", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(args.output_dir / f"{name}_linked.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(3, 5, figsize=(16, 10))
        # Coronal first: decreasing y follows anterior toward posterior.
        for row, axis in enumerate((1, 0, 2)):
            for column, delta in enumerate((2, 1, 0, -1, -2)):
                index = int(np.clip(anchor[axis]+delta, 0, image.shape[axis]-1))
                show(axes[row, column], axis, index)
        fig.suptitle(f"{name} · raw neighbouring views around selected anterior voxel\nTop row: anterior → posterior. Middle/bottom: neighbouring sagittal/axial sections. No labels or crosshairs.", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(args.output_dir / f"{name}_neighbours.png", dpi=170)
        plt.close(fig)
        y_indices, counts = np.unique(points[:, 1], return_counts=True)
        records.append({"case": name, "anchor_native_xyz": anchor.tolist(),
                        "last_anterior_native_y": int(np.where(labels == 1)[1].min()),
                        "interface_y_counts": dict(zip(map(str, y_indices), map(int, counts))),
                        "selection": ("Explicit native anterior voxel" if args.anchor else "Nearest actual interface voxel to interface centroid") + "; no anatomical landmark identification",
                        "slice_label_counts": [{"y": int(y), "anterior": int((labels[:, y] == 1).sum()),
                                                "posterior": int((labels[:, y] == 2).sum())}
                                               for y in range(max(0, int(y_indices.min())-2), min(labels.shape[1], int(y_indices.max())+3))]})
        print(name, records[-1]["anchor_native_xyz"], flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
