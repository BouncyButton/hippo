"""Native-resolution, label-guided inspection of coronal hippocampal profiles.

Component and column-run counts are geometric diagnostics, NOT uncal detectors.
All slice indices are zero-based native indices; blue is anterior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-uncal-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from .foldedness import best_fit_first_anterior_slice

ROOT = Path(__file__).resolve().parents[3]
BLUE, ORANGE = "#168ce5", "#ef951c"


def profile_geometry(mask: np.ndarray) -> dict:
    """Describe disconnected sections and vertically separated tissue runs."""
    sizes = {}
    for connectivity in (1, 2):
        cc, _ = ndimage.label(mask, ndimage.generate_binary_structure(2, connectivity))
        sizes[str(connectivity)] = sorted(np.bincount(cc.ravel())[1:].tolist(), reverse=True)
    double_columns = []
    for x, column in enumerate(mask):
        edges = np.diff(np.pad(column.astype(np.int8), 1))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        if np.sum(lengths >= 2) >= 2:
            double_columns.append(x)
    return {"component_sizes_4": sizes["1"], "component_sizes_8": sizes["2"],
            "double_columns_min_run2": double_columns,
            "split_min2": sum(s >= 2 for s in sizes["2"]) >= 2,
            "double_run_min2cols": len(double_columns) >= 2}


def load_case(dataset: Path, name: str) -> dict:
    ni = nib.load(dataset / "imagesTr" / f"{name}_0000.nii.gz")
    nl = nib.load(dataset / "labelsTr" / f"{name}.nii.gz")
    if not np.allclose(ni.affine, nl.affine) or not np.allclose(ni.affine[:3, :3], np.eye(3)):
        raise ValueError(f"{name}: expected aligned, axis-aligned 1-mm arrays")
    image, labels = np.asarray(ni.dataobj), np.asarray(nl.dataobj)
    if image.shape != labels.shape or set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError(f"{name}: invalid arrays")
    cut, cost, _ = best_fit_first_anterior_slice(labels)
    fg = np.argwhere(labels > 0)
    bounds = np.stack([np.maximum(fg.min(0)-4, 0), np.minimum(fg.max(0)+5, labels.shape)])
    low, high = np.percentile(image, (1, 99))
    return dict(name=name, image=image, labels=labels, cut=int(cut), cost=int(cost),
                bounds=bounds, low=low, high=high)


def panel(ax, case: dict, axis: int, index: int, outlined: bool = False):
    raw, lab = np.take(case["image"], index, axis), np.take(case["labels"], index, axis)
    ax.imshow(raw.T, origin="lower", cmap="gray", vmin=case["low"], vmax=case["high"], interpolation="nearest")
    if outlined:
        for label, color in ((1, BLUE), (2, ORANGE)):
            if (lab == label).any():
                ax.contour((lab == label).T, levels=[.5], colors=[color], linewidths=.65)
    dims = [d for d in range(3) if d != axis]
    ax.set_xlim(case["bounds"][0, dims[0]]-.5, case["bounds"][1, dims[0]]-.5)
    ax.set_ylim(case["bounds"][0, dims[1]]-.5, case["bounds"][1, dims[1]]-.5)
    ax.set_xticks([]); ax.set_yticks([])
    if axis == 0:
        ax.axvline(case["cut"]-.5, color="#e04576", lw=.6, alpha=.8)
    for spine in ax.spines.values(): spine.set_visible(False)


def detailed(case: dict, out: Path):
    name = case["name"]
    label = case["labels"]
    for axis, title in ((1, "coronal"), (0, "sagittal"), (2, "axial")):
        if axis == 1:
            end = min(case["cut"]-5, np.where(label == 1)[1].min()-2)
            indices = list(range(int(np.where(label > 0)[1].max()), max(0, end)-1, -1))
        else:
            indices = list(range(int(np.where(label > 0)[axis].min()), int(np.where(label > 0)[axis].max())+1))
        for start in range(0, len(indices), 10):
            selected = indices[start:start+10]
            fig, axs = plt.subplots(4, 5, figsize=(13, 10), squeeze=False)
            for ax in axs.flat: ax.axis("off")
            for k, index in enumerate(selected):
                for overlay in (0, 1):
                    ax = axs[2*(k//5)+overlay, k%5]
                    panel(ax, case, axis, index, bool(overlay))
                    ax.set_title(f"{'xyz'[axis]}={index}" + (" outlines" if overlay else " raw"), fontsize=9)
            fig.suptitle(f"{name} | {title} | native indices | blue anterior, orange posterior\nLabel-guided inspection; boundary y={case['cut']-.5:g}; no verified apex marker", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .94))
            fig.savefig(out/f"{name}_{title}_{start//10+1}.png", dpi=140); plt.close(fig)


def comparison(cases: list[dict], out: Path):
    """Show contrasting observations, without asserting landmark identity."""
    selection = [("185", 18, "separate small profile at the cut"),
                 ("205", 17, "connected returning lip"),
                 ("164", 18, "ambiguous landmark at the cut")]
    fig, axs = plt.subplots(6, 4, figsize=(12, 16))
    for row, (suffix, sagittal_x, observation) in enumerate(selection):
        case = next(c for c in cases if c["name"].endswith(suffix))
        planes = [(1, case["cut"]+1), (1, case["cut"]),
                  (1, case["cut"]-1), (0, sagittal_x)]
        for col, (axis, index) in enumerate(planes):
            for overlay in (0, 1):
                ax = axs[2*row+overlay, col]
                panel(ax, case, axis, index, bool(overlay))
                if not overlay:
                    title = f"{'Coronal y' if axis == 1 else 'Sagittal x'}={index}"
                    if col == 1: title += " (last anterior)"
                    ax.set_title(title, fontsize=10)
        axs[2*row, 0].set_ylabel(f"{suffix}\nraw MRI", fontsize=11)
        axs[2*row+1, 0].set_ylabel(f"{suffix}\nlabel outlines", fontsize=11)
        axs[2*row+1, 1].set_xlabel(observation, fontsize=10)
    fig.suptitle("Fold 0 validation: contrasting boundary appearances\n"
                 "Blue = anterior; orange = posterior. Native, zero-based slice indices.\n"
                 "Coronal sequence runs anterior to posterior. Pink line = label interface, not a verified apex.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2)
    fig.savefig(out/"comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"experiments/uncal_validation_profiles_20260921")
    parser.add_argument("--details", nargs="*", default=[])
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    dataset = ROOT/"datasets/Dataset101_MSD"
    split_path = dataset/"splits_final.json"
    split = json.loads(split_path.read_text())[0]
    assert not set(split["train"]) & set(split["val"])
    cases = [load_case(dataset, name) for name in sorted(split["val"])]
    records = []
    for case in cases:
        labels, cut = case["labels"], case["cut"]
        per_slice = [{"y": y, "anterior": int((labels[:, y] == 1).sum()),
                      "posterior": int((labels[:, y] == 2).sum()),
                      **profile_geometry(labels[:, y] > 0)}
                     for y in range(labels.shape[1]) if (labels[:, y] > 0).any()]
        record = {"case": case["name"], "shape": list(labels.shape), "best_fit_cut": cut,
                  "plane_disagreement_voxels": case["cost"],
                  "last_anterior_y": int(np.where(labels == 1)[1].min()),
                  "last_posterior_y": int(np.where(labels == 2)[1].max()), "slices": per_slice}
        for metric in ("split_min2", "double_run_min2cols"):
            record[metric+"_at_cut"] = any(r[metric] for r in per_slice if r["y"] == cut)
            record[metric+"_near_cut"] = any(r[metric] for r in per_slice if abs(r["y"]-cut) <= 3)
            record[metric+"_any_anterior"] = any(r[metric] for r in per_slice if r["anterior"] > 0)
            record[metric+"_any_posterior"] = any(r[metric] for r in per_slice if r["posterior"] > 0)
            record[metric+"_disappears_at_cut"] = any(r[metric] for r in per_slice if r["y"] == cut) and not any(r[metric] for r in per_slice if r["y"] == cut-1)
        records.append(record)
    for start in range(0, len(cases), 4):
        subset = cases[start:start+4]
        fig, axs = plt.subplots(8, 7, figsize=(16, 18))
        for row, case in enumerate(subset):
            for col, delta in enumerate((3, 2, 1, 0, -1, -2, -3)):
                index = case["cut"]+delta
                for overlay in (0, 1):
                    ax = axs[2*row+overlay, col]
                    panel(ax, case, 1, index, bool(overlay))
                    if not overlay:
                        ax.set_title(f"{case['name'][-3:]} y={index}"+(' *' if delta==0 else ''), fontsize=10)
                if col == 0:
                    axs[2*row+1,col].set_ylabel(f"{case['name'][-3:]} labels", fontsize=10)
        fig.suptitle(f"Fold 0 validation | cases {start+1}-{start+len(subset)} of {len(cases)}\nEach pair: raw MRI then contours. Left to right: anterior to posterior. * = best-fit last anterior plane.\nBlue anterior; orange posterior. Profiles are not independently identified anatomical landmarks.", fontsize=13)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(out/f"screen_{start//4+1:02}.png",dpi=140); plt.close(fig)
    for case in cases:
        if case["name"][-3:] in args.details: detailed(case, out)
    comparison(cases, out)
    summary = {"fold": 0, "n": len(records), "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
               "nonplanar_cases": sum(r["plane_disagreement_voxels"]>0 for r in records),
               "metric_counts": {key: sum(r[key] for r in records) for key in records[0] if key.startswith(("split_min2_", "double_run_min2cols_"))},
               "interpretation": "Geometric diagnostics of reference masks only; not anatomical visibility or detector accuracy.",
               "cases": records}
    (out/"audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'},indent=2))


if __name__ == "__main__": main()
