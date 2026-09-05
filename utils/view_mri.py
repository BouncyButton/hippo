#!/usr/bin/env python3
"""Interactively view an MSD or MNI MRI with its segmentation overlay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MSD_ROOT = REPOSITORY_ROOT / "datasets" / "Dataset101_MSD"
MNI_ROOT = REPOSITORY_ROOT / "mri_dataset"


@dataclass(frozen=True)
class ViewerCase:
    image: np.ndarray
    mask: np.ndarray
    title: str
    class_names: Mapping[int, str]
    class_colors: Mapping[int, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View an MRI with its ground-truth segmentation mask."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=str.lower,
        choices=("msd", "mni"),
        help="Dataset to view.",
    )
    parser.add_argument(
        "--patient",
        required=True,
        help="MSD three-digit training case ID or MNI subject ID (01-25).",
    )
    return parser.parse_args(argv)


def _load_volume(path: Path, *, canonical: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required NIfTI file not found: {path}")

    image = nib.load(path)
    if canonical:
        image = nib.as_closest_canonical(image)
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {path}, found shape {data.shape}")
    return data, image.affine


def _validate_pair(
    image: np.ndarray,
    image_affine: np.ndarray,
    mask: np.ndarray,
    mask_affine: np.ndarray,
    *,
    context: str,
) -> None:
    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask shapes do not match for {context}: "
            f"{image.shape} versus {mask.shape}"
        )
    if not np.allclose(image_affine, mask_affine, atol=1e-4):
        raise ValueError(f"Image and mask affines do not match for {context}")


def _normalise_msd_patient(patient: str) -> str:
    if not patient.isdigit() or len(patient) > 3:
        raise ValueError("MSD --patient must be a three-digit case ID, for example 001")

    case_id = patient.zfill(3)
    image_path = MSD_ROOT / "imagesTr" / f"hippocampus_{case_id}_0000.nii.gz"
    label_path = MSD_ROOT / "labelsTr" / f"hippocampus_{case_id}.nii.gz"
    if image_path.is_file() and label_path.is_file():
        return case_id

    available = sorted(
        path.name.removeprefix("hippocampus_").removesuffix("_0000.nii.gz")
        for path in (MSD_ROOT / "imagesTr").glob("hippocampus_*_0000.nii.gz")
        if not path.name.startswith("._")
    )
    preview = ", ".join(available[:20])
    suffix = " ..." if len(available) > 20 else ""
    raise ValueError(
        f"MSD case {case_id} is not in the training set. "
        f"Available case IDs begin: {preview}{suffix}"
    )


def load_msd_case(patient: str) -> ViewerCase:
    case_id = _normalise_msd_patient(patient)
    image_path = MSD_ROOT / "imagesTr" / f"hippocampus_{case_id}_0000.nii.gz"
    label_path = MSD_ROOT / "labelsTr" / f"hippocampus_{case_id}.nii.gz"

    image, image_affine = _load_volume(image_path)
    mask, mask_affine = _load_volume(label_path)
    _validate_pair(image, image_affine, mask, mask_affine, context=f"MSD {case_id}")

    return ViewerCase(
        image=image.astype(np.float32, copy=False),
        mask=np.rint(mask).astype(np.uint8, copy=False),
        title=f"MSD training case {case_id}",
        class_names={1: "Anterior", 2: "Posterior"},
        class_colors={1: "#e63946", 2: "#3a86ff"},
    )


def _normalise_mni_patient(patient: str) -> str:
    if not patient.isdigit():
        raise ValueError("MNI --patient must be a number from 01 to 25")
    patient_number = int(patient)
    if not 1 <= patient_number <= 25:
        raise ValueError("MNI --patient must be a number from 01 to 25")
    return f"{patient_number:02d}"


def load_mni_case(patient: str) -> ViewerCase:
    patient_id = _normalise_mni_patient(patient)
    subject = f"s{patient_id}"
    subject_root = MNI_ROOT / subject
    image_path = subject_root / f"{subject}_t1w_standard_defaced_MNI.nii.gz"
    left_path = subject_root / f"{subject}_hippolabels_t1w_standard_L_MNI.nii.gz"
    right_path = subject_root / f"{subject}_hippolabels_t1w_standard_R_MNI.nii.gz"

    image, image_affine = _load_volume(image_path)
    left_mask, left_affine = _load_volume(left_path)
    right_mask, right_affine = _load_volume(right_path)
    _validate_pair(
        image, image_affine, left_mask, left_affine, context=f"MNI {subject} left"
    )
    _validate_pair(
        image, image_affine, right_mask, right_affine, context=f"MNI {subject} right"
    )

    left_mask = np.rint(left_mask).astype(np.uint8, copy=False)
    right_mask = np.rint(right_mask).astype(np.uint8, copy=False)
    overlap = (left_mask > 0) & (right_mask > 0) & (left_mask != right_mask)
    if np.any(overlap):
        raise ValueError(f"Conflicting left/right labels found for MNI {subject}")
    mask = np.maximum(left_mask, right_mask)

    return ViewerCase(
        image=image.astype(np.float32, copy=False),
        mask=mask,
        title=f"MNI {subject} - standard T1 in MNI152 space",
        class_names={1: "CA1-3", 2: "Subiculum", 3: "CA4-DG"},
        class_colors={1: "#ff9f1c", 2: "#3a86ff", 3: "#2a9d8f"},
    )


def load_case(data: str, patient: str) -> ViewerCase:
    if data == "msd":
        return load_msd_case(patient)
    if data == "mni":
        return load_mni_case(patient)
    raise ValueError(f"Unsupported dataset: {data}")


def _display_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def _initial_indices(mask: np.ndarray) -> list[int]:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return [size // 2 for size in mask.shape]
    return [int(np.median(foreground[:, axis])) for axis in range(3)]


def _intensity_window(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, (1, 99))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def show_case(case: ViewerCase) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch
        from matplotlib.widgets import Slider
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for the viewer. Install the project requirements first."
        ) from exc

    axis_names = ("Sagittal", "Coronal", "Axial")
    indices = _initial_indices(case.mask)
    vmin, vmax = _intensity_window(case.image)
    max_class = max(case.class_names)
    colors = [(0.0, 0.0, 0.0, 0.0)] + [
        case.class_colors.get(label, "#ffffff") for label in range(1, max_class + 1)
    ]
    mask_cmap = ListedColormap(colors)
    mask_norm = BoundaryNorm(np.arange(-0.5, max_class + 1.5), mask_cmap.N)

    figure, axes = plt.subplots(1, 3, figsize=(15, 6))
    figure.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.20, wspace=0.08)
    figure.suptitle(case.title, fontsize=15)
    try:
        figure.canvas.manager.set_window_title(case.title)
    except AttributeError:
        pass

    image_artists = []
    mask_artists = []
    sliders = []
    for axis, (plot_axis, axis_name, index) in enumerate(
        zip(axes, axis_names, indices, strict=True)
    ):
        image_artists.append(
            plot_axis.imshow(
                _display_slice(case.image, axis, index),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        )
        mask_artists.append(
            plot_axis.imshow(
                np.ma.masked_equal(_display_slice(case.mask, axis, index), 0),
                cmap=mask_cmap,
                norm=mask_norm,
                alpha=0.55,
                interpolation="nearest",
            )
        )
        plot_axis.set_title(f"{axis_name} | slice {index}")
        plot_axis.set_axis_off()

        slider_axis = figure.add_axes([0.055 + axis * 0.325, 0.09, 0.27, 0.035])
        slider = Slider(
            slider_axis,
            "",
            0,
            case.image.shape[axis] - 1,
            valinit=index,
            valstep=1,
        )
        slider.valtext.set_visible(False)
        sliders.append(slider)

    def update_axis(axis: int, value: float) -> None:
        index = int(value)
        image_artists[axis].set_data(_display_slice(case.image, axis, index))
        mask_artists[axis].set_data(
            np.ma.masked_equal(_display_slice(case.mask, axis, index), 0)
        )
        axes[axis].set_title(f"{axis_names[axis]} | slice {index}")
        figure.canvas.draw_idle()

    for axis, slider in enumerate(sliders):
        slider.on_changed(lambda value, axis=axis: update_axis(axis, value))

    def on_scroll(event: object) -> None:
        event_axis = getattr(event, "inaxes", None)
        if event_axis not in axes:
            return
        axis = list(axes).index(event_axis)
        step = 1 if getattr(event, "button", None) == "up" else -1
        new_value = np.clip(sliders[axis].val + step, sliders[axis].valmin, sliders[axis].valmax)
        sliders[axis].set_val(new_value)

    figure.canvas.mpl_connect("scroll_event", on_scroll)
    legend_handles = [
        Patch(facecolor=case.class_colors[label], label=name)
        for label, name in case.class_names.items()
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        case = load_case(args.data, args.patient)
        show_case(case)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
