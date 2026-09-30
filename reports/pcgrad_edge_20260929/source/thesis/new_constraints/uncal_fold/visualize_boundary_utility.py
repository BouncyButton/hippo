"""Render the boundary audit and a raw-MRI review sheet, with no inferred apex dot."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-uncal-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments/uncal_boundary_utility_20260921"


def main():
    summary = json.loads((OUTPUT / "summary.json").read_text())
    methods = ("native", "native_plane", "feature_plane", "geometry_image_plane", "median_position_plane", "oracle_label_plane")
    labels = ("Native model", "Native cut → plane", "Feature cut → plane", "Shape + MRI cut → plane", "Position prior → plane", "Label oracle → plane")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ("unaugmented", "augmented")):
        values = [summary["models"][model][m]["band_mean_ap_dice"] for m in methods]
        bars = ax.barh(labels, values, color=["#536f86"] * 5 + ["#bc813a"])
        ax.set(xlim=(0, 1), xlabel="Mean A/P Dice within ±2 mm of reference plane", title=model.capitalize())
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("MSD fold 0 · 52 validation crops · foreground held fixed", fontsize=14)
    fig.text(.5, .01, "Exploratory counterfactual. Label oracle uses validation labels; it is not a deployable detector.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUTPUT / "boundary_utility.png", dpi=170)
    plt.close(fig)

    # Deterministic training case: first in the saved split, independent of cut error.
    dataset = ROOT / "datasets/Dataset101_MSD"
    case = sorted(json.loads((dataset / "splits_final.json").read_text())[0]["train"])[0]
    image = np.asarray(nib.load(dataset / "imagesTr" / f"{case}_0000.nii.gz").dataobj)
    union = np.asarray(nib.load(dataset / "labelsTr" / f"{case}.nii.gz").dataobj) > 0
    occupied = np.flatnonzero(union.any(axis=(0, 2)))
    # Whole foreground extent, never a window centred on the A/P ground-truth cut.
    indices = occupied[::-1]
    ncols = 8
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 2.1), squeeze=False)
    low, high = np.percentile(image, (1, 99))
    for ax in axes.flat:
        ax.axis("off")
    for ax, y in zip(axes.flat, indices):
        ax.imshow(image[:, y, :].T, origin="lower", cmap="gray", vmin=low, vmax=high)
        ax.set_title(f"y = {y}", fontsize=9)
    fig.suptitle(f"{case} · raw T1 coronal sequence · anterior → posterior (decreasing y)\nLook for disappearance of the double-level fold; no A/P labels or proposed landmark shown", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUTPUT / "training_case_blinded_coronal.png", dpi=160)
    plt.close(fig)
    with (OUTPUT / "manual_review_template.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "reviewer", "last_visible_uncus_native_y", "uncertainty_low_y", "uncertainty_high_y", "visibility_clear_uncertain_not_visible", "notes"])
        writer.writerow([case, "", "", "", "", "", ""])
    print(OUTPUT)


if __name__ == "__main__":
    main()
