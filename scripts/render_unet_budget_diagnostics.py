#!/usr/bin/env python3
"""Render scientific plots from the verified U-Net diagnostic tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def error_plot(root, summary):
    pooled = summary["pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    keys = ["foreground_fp", "foreground_fn", "anterior_posterior_swap"]
    values = [100 * pooled["error_composition"][key] for key in keys]
    axes[0].bar(["Extra foreground\n(false positive)", "Missed foreground\n(false negative)", "A/P label swaps"],
                values, color=["#b51666", "#166baa", "#c09100"])
    for i, value in enumerate(values):
        axes[0].text(i, value + 1, f"{value:.1f}%", ha="center")
    axes[0].set_ylim(0, max(values) + 10)
    axes[0].set_ylabel("Share of all mislabeled voxels (%)")
    axes[0].set_title("What kinds of errors remain?")
    for name, label, color in [("all", "All errors", "#222222"), ("fp", "False positives", "#b51666"),
                               ("fn", "False negatives", "#166baa"), ("swap", "A/P swaps", "#c09100")]:
        values = [100 * pooled["distance_fractions"][f"{name}_outer_le_{i}"] for i in range(4)]
        axes[1].plot(range(4), values, marker="o", label=label, color=color)
    axes[1].set_xlabel("Distance to GT outer contour (grid voxels)")
    axes[1].set_ylabel("Cumulative share of each error type (%)")
    axes[1].set_ylim(0, 103)
    axes[1].set_xticks(range(4))
    axes[1].set_title("How close are errors to the contour?")
    axes[1].legend(fontsize=8, loc="lower right")
    rates = [100 * pooled["inner_shell_foreground_error_rate"], 100 * pooled["deep_foreground_error_rate"]]
    axes[2].bar(["GT foreground shell\n≤2 grid voxels", "GT foreground interior\n>2 grid voxels"], rates,
                color=["#006e87", "#78909c"])
    for i, value in enumerate(rates):
        axes[2].text(i, value + 0.5, f"{value:.1f}%", ha="center")
    axes[2].set_ylim(0, max(rates) + 8)
    axes[2].set_ylabel("Mislabeled / foreground region voxels (%)")
    axes[2].set_title("Error rate accounts for region size")
    axes[2].text(0.5, 0.92, f"Shell contains {100 * pooled['gt_inner_shell_population_share']:.1f}% of GT foreground",
                 transform=axes[2].transAxes, fontsize=8, ha="center")
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Selected U-Net checkpoints: 9 runs × 52 validation cases", fontsize=12)
    fig.savefig(root / "spatial_error_summary.png", dpi=180)
    fig.savefig(root / "spatial_error_summary.pdf")
    plt.close(fig)


def example_plot(root, name):
    with np.load(root / f"example_{name}.npz") as data:
        truth, pred = data["label"], data["prediction"]
    metadata = json.loads((root / f"example_{name}.json").read_text())
    record = metadata["record"]
    codes = np.zeros(truth.shape, dtype=np.uint8)
    codes[(truth == 0) & (pred > 0)] = 1
    codes[(truth > 0) & (pred == 0)] = 2
    codes[(truth > 0) & (pred > 0) & (truth != pred)] = 3
    coordinates = np.argwhere(truth > 0)
    centers = np.median(coordinates, axis=0).astype(int)
    foreground = np.argwhere((truth > 0) | (pred > 0))
    lower = np.maximum(0, foreground.min(0) - 3)
    upper = np.minimum(truth.shape, foreground.max(0) + 4)
    segmentation_colors = ListedColormap(["#f7f8fa", "#008b8b", "#e98948"])
    error_colors = ListedColormap(["#f7f8fa", "#b51666", "#166baa", "#c09100"])
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), constrained_layout=True)
    for axis in range(3):
        other_axes = [i for i in range(3) if i != axis]
        crop = tuple(slice(lower[i], upper[i]) for i in other_axes)
        gt = np.take(truth, centers[axis], axis=axis)[crop]
        proposal = np.take(pred, centers[axis], axis=axis)[crop]
        errors = np.take(codes, centers[axis], axis=axis)[crop]
        for column, (array, cmap, vmax) in enumerate(((gt, segmentation_colors, 2), (proposal, segmentation_colors, 2), (errors, error_colors, 3))):
            ax = axes[axis, column]
            ax.imshow(array.T, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
            if (gt > 0).any() and not (gt > 0).all():
                ax.contour((gt > 0).T.astype(float), levels=[0.5], colors="#222222", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if axis == 0:
                ax.set_title(("Ground truth", "Prediction", "Voxel errors")[column])
            if column == 0:
                ax.set_ylabel(f"Grid axis {axis}\nslice {centers[axis]}", fontsize=9)
    score = (record["dice_anterior"] + record["dice_posterior"]) / 2
    fig.suptitle(f"Fold 0, seed 0: {name} validation case by macro Dice ({score:.3f})\nGT-centered slices; black line = GT outer contour", fontsize=11)
    fig.legend(handles=[Patch(color="#008b8b", label="Anterior"), Patch(color="#e98948", label="Posterior"),
                        Patch(color="#b51666", label="False positive"), Patch(color="#166baa", label="False negative"),
                        Patch(color="#c09100", label="A/P swap")], loc="outside lower center", ncol=5, fontsize=8)
    fig.savefig(root / f"example_{name}.png", dpi=180)
    fig.savefig(root / f"example_{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "spatial_summary.json").read_text())
    error_plot(args.root, summary)
    for name in ("median", "worst"):
        example_plot(args.root, name)


if __name__ == "__main__":
    main()
