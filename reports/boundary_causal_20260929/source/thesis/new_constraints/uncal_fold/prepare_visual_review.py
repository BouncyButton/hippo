"""Prepare a raw-image-only uncal review packet from predetermined train IDs.

This script never opens segmentation labels, model outputs or locator scores.
The full native crop is shown; no label-derived ROI or cut-centred window is used.
"""
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

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments/uncal_visual_review_20260921"


def render_grid(image, indices, axis, path, title, columns=8):
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns*2.6, rows*2.7), squeeze=False)
    low, high = np.percentile(image, (1, 99))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        if axis == 1:
            plane = image[:, i, :].T
            label = f"coronal y={i}"
        else:
            plane = image[i, :, :].T
            label = f"sagittal x={i}"
        ax.imshow(plane, origin="lower", cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        ax.set_title(label, fontsize=12)
        if axis == 0:
            ax.axis("on")
            ax.set_xticks(np.arange(0, image.shape[1], 10))
            ax.set_yticks([])
            ax.tick_params(labelsize=9)
            ax.set_xlabel("P ← native y → A", fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=2.5)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None)
    parser.add_argument("--y-range", type=int, nargs=2)
    args = parser.parse_args()
    dataset = ROOT / "datasets/Dataset101_MSD"
    split_path = dataset / "splits_final.json"
    split = json.loads(split_path.read_text())[0]
    chosen = np.random.default_rng(20260922).choice(sorted(split["train"]), 12, replace=False).tolist()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.exists():
        manifest = {"schema": "uncal_visual_review.v1", "selection_seed": 20260922,
                    "count": 12, "cases": chosen, "selection": "12 random fold-0 training crop IDs; no label/prediction inspection",
                    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    "reviewer_type": "AI provisional; not a qualified human anatomical rater",
                    "prior_exposure": "Reviewer knows aggregate MSD results and general positional/anatomical context from preceding conversation; not an independent naive rater",
                    "coordinate_convention": "native RAS array index; anterior-to-posterior is decreasing y; last visible slice remains anterior",
                    "input_images": {}}
        for name in chosen:
            path = dataset / "imagesTr" / f"{name}_0000.nii.gz"
            manifest["input_images"][name] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        assert json.loads(manifest_path.read_text())["cases"] == chosen
    for name in ([args.case] if args.case else chosen):
        if name not in chosen:
            raise ValueError("Case must belong to the fixed pilot sample")
        nii = nib.load(dataset / "imagesTr" / f"{name}_0000.nii.gz")
        if not np.allclose(nii.affine[:3, :3], np.eye(3)):
            raise ValueError("Expected native axis-aligned 1-mm RAS")
        image = np.asarray(nii.dataobj)
        if args.y_range:
            lo, hi = args.y_range
            render_grid(image, list(range(hi, lo-1, -1)), 1, OUTPUT / f"{name}_detail_{lo}_{hi}.png",
                        f"{name} · raw T1 · anterior → posterior · no labels", columns=5)
        else:
            render_grid(image, list(range(image.shape[1]-1, -1, -1)), 1, OUTPUT / f"{name}_coronal.png",
                        f"{name} · raw T1 coronal · anterior → posterior (decreasing y) · no labels")
            render_grid(image, list(range(0, image.shape[0], 2)), 0, OUTPUT / f"{name}_sagittal.png",
                        f"{name} · raw T1 sagittal context · increasing x · no labels", columns=6)
        print(name, image.shape, flush=True)


if __name__ == "__main__":
    main()
