"""Render raw MRI and foreground-union views for fold-0 phenotype medoids."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-uncal-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs/experiments/uncal_cut_centered_phenotypes_20260921"
DATASET = ROOT / "datasets/Dataset101_MSD"
SHORT_FEATURE = {
    "image_current__superior_band_intensity_mean": "superior-band mean",
    "image_current__central_superior_band_intensity_mean": "central superior mean",
    "image_current__superior_band_intensity_std": "superior-band SD",
    "image_current__superior_band_left_right_difference_abs": "superior L/R difference",
    "transition__novel_superior_side_difference": "new superior-side difference",
    "transition__slice_dice": "adjacent-slice Dice",
    "transition__superior_boundary_rise_max": "superior-boundary rise",
    "clear_shape_delta__area": "area change",
    "clear_shape_delta__superior_notch_depth": "notch-depth change",
    "clear_shape_current__multirun_side_max": "multi-run side",
}


def load_case(name: str) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    image = np.asarray(
        nib.load(DATASET / "imagesTr" / f"{name}_0000.nii.gz").dataobj,
        dtype=np.float32,
    )
    labels = np.asarray(
        nib.load(DATASET / "labelsTr" / f"{name}.nii.gz").dataobj,
        dtype=np.uint8,
    )
    cut = int(np.where(labels == 1)[1].min())
    low, high = np.percentile(image, (1, 99))
    return image, labels, cut, float(low), float(high)


def show_panel(
    ax,
    image: np.ndarray,
    labels: np.ndarray,
    axis: int,
    index: int,
    low: float,
    high: float,
    cut: int,
) -> None:
    raw = np.take(image, index, axis).T
    foreground = np.take(labels > 0, index, axis).T
    ax.imshow(raw, origin="lower", cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    if foreground.any():
        ax.contour(foreground, levels=[0.5], colors=["#ffffff"], linewidths=0.7)
    coordinates = np.argwhere(labels > 0)
    dimensions = [dimension for dimension in range(3) if dimension != axis]
    lo = np.maximum(coordinates.min(axis=0) - 4, 0)
    hi = np.minimum(coordinates.max(axis=0) + 5, labels.shape)
    ax.set_xlim(lo[dimensions[0]] - 0.5, hi[dimensions[0]] - 0.5)
    ax.set_ylim(lo[dimensions[1]] - 0.5, hi[dimensions[1]] - 0.5)
    if axis == 0:
        ax.axvline(cut - 0.5, color="#ed4b82", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    summary = json.loads((REPORT / "summary.json").read_text())
    fold = next(item for item in summary["folds"] if item["fold"] == 0)
    for cluster in fold["training_clusters"]:
        names = cluster["representative_cases"]
        fig, axes = plt.subplots(len(names), 4, figsize=(11, 2.9 * len(names)), squeeze=False)
        for row, name in enumerate(names):
            image, labels, cut, low, high = load_case(name)
            foreground_near_cut = (labels[:, max(0, cut - 1) : cut + 2, :] > 0).sum(axis=(1, 2))
            sagittal_x = int(np.argmax(foreground_near_cut))
            for column, (axis, index) in enumerate(
                ((1, cut + 1), (1, cut), (1, cut - 1), (0, sagittal_x))
            ):
                show_panel(axes[row, column], image, labels, axis, index, low, high, cut)
                title = f"coronal y={index}" if axis == 1 else f"sagittal x={index}"
                axes[row, column].set_title(title, fontsize=9)
            axes[row, 0].set_ylabel(name[-3:], fontsize=10)
        shifts = ", ".join(
            f"{SHORT_FEATURE.get(row['feature'], row['feature'].split('__')[-1])} "
            f"{row['mean_shift']:+.2f}"
            for row in cluster["largest_standardized_feature_shifts"][:3]
        )
        fig.suptitle(
            f"Fold 0 training | cut-centred phenotype {cluster['cluster']} | n={cluster['size']}\n"
            "White: whole-hippocampus outline; pink: annotated sagittal cut; no A/P colours\n"
            f"Largest standardized feature shifts: {shifts}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(REPORT / f"fold0_cluster_{cluster['cluster']}_medoids.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
