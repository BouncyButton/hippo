"""Visual explanation of the compact uncal-transition descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from .foldedness import best_fit_first_anterior_slice
from .refined import extract_refined_slice_features, extract_transition_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/uncal_foldedness_refined_20260921"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("hippocampus_017", "hippocampus_164"),
    )
    return parser.parse_args()


def _draw_base(axis: plt.Axes, image: np.ndarray, low: float, high: float) -> None:
    axis.imshow(image.T, origin="lower", cmap="gray", vmin=low, vmax=high)
    axis.set_xticks([])
    axis.set_yticks([])


def visualize_case(root: Path, case: str, output: Path) -> None:
    image = np.asarray(
        nib.load(str(root / "imagesTr" / f"{case}_0000.nii.gz")).dataobj,
        dtype=np.float64,
    )
    labels = np.asarray(
        nib.load(str(root / "labelsTr" / f"{case}.nii.gz")).dataobj,
        dtype=np.uint8,
    )
    cut, _, _ = best_fit_first_anterior_slice(labels)
    previous = labels[:, cut - 1, :] != 0
    current = labels[:, cut, :] != 0
    added = current & ~previous
    novel = current & ~ndimage.binary_dilation(previous, iterations=1)
    coordinates = np.argwhere(current)
    low_coord, high_coord = coordinates.min(axis=0), coordinates.max(axis=0)
    z_mid = int((low_coord[1] + high_coord[1] + 1) // 2)
    superior_band = ndimage.binary_dilation(current, iterations=2) & ~current
    superior_band[:, :z_mid] = False

    shape = extract_refined_slice_features(current)
    transition = extract_transition_features(previous, current)
    display_low, display_high = np.quantile(image, (0.01, 0.99))

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    _draw_base(axes[0, 0], image[:, cut - 1, :], display_low, display_high)
    axes[0, 0].contour(previous.T, levels=[0.5], colors=["cyan"], linewidths=2)
    axes[0, 0].set_title(f"Posterior neighbour: y={cut - 1}")

    _draw_base(axes[0, 1], image[:, cut, :], display_low, display_high)
    axes[0, 1].contour(previous.T, levels=[0.5], colors=["cyan"], linewidths=1.5)
    axes[0, 1].contour(current.T, levels=[0.5], colors=["red"], linewidths=2)
    axes[0, 1].set_title(f"Annotated last-head slice: y={cut}\ncyan=previous, red=current")

    _draw_base(axes[1, 0], image[:, cut, :], display_low, display_high)
    added_overlay = np.ma.masked_where(~added.T, added.T)
    novel_overlay = np.ma.masked_where(~novel.T, novel.T)
    axes[1, 0].imshow(added_overlay, origin="lower", cmap="autumn", alpha=0.55, vmin=0, vmax=1)
    axes[1, 0].imshow(novel_overlay, origin="lower", cmap="spring", alpha=0.9, vmin=0, vmax=1)
    axes[1, 0].contour(current.T, levels=[0.5], colors=["white"], linewidths=1)
    axes[1, 0].set_title("Cross-sectional expansion\nyellow=added, magenta=novel after 1-pixel tolerance")

    _draw_base(axes[1, 1], image[:, cut, :], display_low, display_high)
    band_overlay = np.ma.masked_where(~superior_band.T, superior_band.T)
    axes[1, 1].imshow(band_overlay, origin="lower", cmap="summer", alpha=0.55, vmin=0, vmax=1)
    axes[1, 1].contour(current.T, levels=[0.5], colors=["red"], linewidths=2)
    axes[1, 1].set_title("Superior T1 band\nintensity/asymmetry measured in green region")

    figure.suptitle(
        f"{case} — interpretable uncal-transition evidence\n"
        f"area change={transition['relative_area_change']:+.2f}, "
        f"max superior rise={transition['superior_boundary_rise_max']:.1f} vox, "
        f"side protrusion={shape['superior_protrusion_max']:.2f}, "
        f"notch={shape['superior_notch_depth']:.2f}, "
        f"multirun={shape['multirun_side_max']:.2f}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        visualize_case(
            args.dataset_root,
            case,
            args.output_dir / f"{case}_descriptor_explanation.png",
        )


if __name__ == "__main__":
    main()
