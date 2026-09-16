#!/usr/bin/env python3
"""Interactively inspect baseline predictions on an exact MSD validation fold."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL = REPOSITORY_ROOT / "datasets" / "Dataset101_MSD" / "msd_hippocampus_full.pkl"
DEFAULT_SPLITS = REPOSITORY_ROOT / "datasets" / "Dataset101_MSD" / "splits_final.json"
DEFAULT_WEIGHTS_BY_SEED = {
    0: REPOSITORY_ROOT
    / "experiments"
    / "augmentation_family_b_20260907"
    / "checkpoints"
    / "baseline_seed0_checkpoint_latest.pt",
    1: REPOSITORY_ROOT
    / "experiments"
    / "augmentation_family_b_20260907"
    / "checkpoints"
    / "baseline_seed1_checkpoint_latest.pt",
}

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@dataclass(frozen=True)
class PredictionCase:
    image: np.ndarray
    ground_truth: np.ndarray | None
    prediction: np.ndarray
    title: str
    dice_anterior: float | None
    dice_posterior: float | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a seeded baseline SwinUNETR on one case from its exact MSD "
            "validation fold. Eligible patient IDs are selected interactively."
        )
    )
    parser.add_argument(
        "--split",
        choices=("val",),
        default="val",
        help="Validation is the only supported split (default: val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Training seed of the model checkpoint. This selects the default "
            "checkpoint and is checked against checkpoint metadata."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional full training checkpoint; defaults to the local baseline for --seed.",
    )
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Auto uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize to --spatial-size instead of the baseline's center crop/pad.",
    )
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Add a third row showing each directed GT-to-prediction error.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optionally save the initial synchronized view as an image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window; useful together with --save.",
    )
    return parser.parse_args(argv)


def _normalise_case_name(patient: str) -> str:
    if patient.startswith("hippocampus_"):
        suffix = patient.removeprefix("hippocampus_")
    else:
        suffix = patient
    if not suffix.isdigit() or len(suffix) > 3:
        raise ValueError(
            "--patient must be an MSD numeric ID such as 001 or hippocampus_001."
        )
    return f"hippocampus_{suffix.zfill(3)}"


def _resolve_weights(seed: int, supplied: Path | None) -> Path:
    if seed < 0:
        raise ValueError("--seed must be non-negative.")
    path = supplied
    if path is None:
        path = DEFAULT_WEIGHTS_BY_SEED.get(seed)
        if path is None:
            available = ", ".join(str(value) for value in sorted(DEFAULT_WEIGHTS_BY_SEED))
            raise ValueError(
                f"No default checkpoint is registered for seed {seed}; "
                f"registered seeds: {available}. Pass --weights explicitly."
            )
    path = path.expanduser().resolve()
    if not path.is_file():
        source = "default " if supplied is None else ""
        raise FileNotFoundError(
            f"The {source}seed-{seed} model checkpoint was not found: {path}. "
            "Use --weights with a persistent full checkpoint path."
        )
    return path


def _checkpoint_selection(
    checkpoint: object,
    *,
    requested_seed: int,
) -> tuple[dict[str, object], int]:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(
            "Validation selection requires a full training checkpoint containing "
            "both 'model' weights and embedded 'run' metadata."
        )
    run = checkpoint.get("run")
    if not isinstance(run, dict):
        raise ValueError("Checkpoint lacks embedded run metadata for fold selection.")
    try:
        checkpoint_seed = int(run["seed"])
        fold = int(run["fold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint run metadata lacks a valid seed or fold.") from exc
    if checkpoint_seed != requested_seed:
        raise ValueError(
            f"Requested seed {requested_seed}, but the checkpoint records seed "
            f"{checkpoint_seed}."
        )
    if run.get("dataset") != "MSD":
        raise ValueError("The selected checkpoint is not an MSD run.")
    return checkpoint["model"], fold


def _validation_case_names(splits_path: Path, fold: int) -> list[str]:
    if not splits_path.is_file():
        raise FileNotFoundError(f"MSD split file not found: {splits_path}")
    try:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read split file: {splits_path}") from exc
    if not isinstance(splits, list) or not 0 <= fold < len(splits):
        raise ValueError(f"Checkpoint fold {fold} is unavailable in {splits_path}.")
    validation = splits[fold].get("val") if isinstance(splits[fold], dict) else None
    if not isinstance(validation, list) or not validation:
        raise ValueError(f"Fold {fold} has no validation cases in {splits_path}.")
    names = sorted(str(name) for name in validation)
    if len(names) != len(set(names)):
        raise ValueError(f"Fold {fold} validation list contains duplicate case names.")
    return names


def _prompt_validation_case(
    validation_names: Sequence[str],
    *,
    seed: int,
    fold: int,
    input_fn: Callable[[str], str] | None = None,
) -> str:
    read_input = input if input_fn is None else input_fn
    patient_ids = [name.removeprefix("hippocampus_") for name in validation_names]
    print(f"Validation patients for checkpoint seed {seed}, fold {fold}:")
    for start in range(0, len(patient_ids), 13):
        print("  " + "  ".join(patient_ids[start : start + 13]))
    allowed = set(validation_names)
    while True:
        try:
            raw = read_input("Choose a validation patient ID: ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise ValueError("Patient selection was cancelled.") from exc
        try:
            case_name = _normalise_case_name(raw)
        except ValueError as exc:
            print(f"Invalid selection: {exc}", file=sys.stderr)
            continue
        if case_name in allowed:
            return case_name
        print(
            f"Patient {case_name.removeprefix('hippocampus_')} is not in fold-{fold} "
            "validation. Choose one of the IDs listed above.",
            file=sys.stderr,
        )


def _resolve_device(requested: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model inference.") from exc

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denominator = int(mask_a.sum() + mask_b.sum())
    if denominator == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(mask_a, mask_b).sum()) / denominator


def _error_codes(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    codes = np.zeros(ground_truth.shape, dtype=np.uint8)
    for truth_label, predicted_label, code in (
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 3),
        (2, 0, 4),
        (1, 2, 5),
        (2, 1, 6),
    ):
        codes[(ground_truth == truth_label) & (prediction == predicted_label)] = code
    return codes


def load_prediction_case(args: argparse.Namespace) -> PredictionCase:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model inference.") from exc

    from baselines.swin_unetr.swin_unetr import (
        _build_monai_dataset_from_pkl,
        _infer_num_classes,
        _load_pkl_dataframe,
    )
    from thesis.new_constraints.train_swinunetr_constraints import build_swinunetr

    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("--opacity must be between 0 and 1.")
    weights_path = _resolve_weights(args.seed, args.weights)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict, fold = _checkpoint_selection(
        checkpoint,
        requested_seed=args.seed,
    )
    validation_names = _validation_case_names(args.splits_json, fold)
    case_name = _prompt_validation_case(
        validation_names,
        seed=args.seed,
        fold=fold,
    )

    num_classes = 3
    if not args.pkl.is_file():
        raise FileNotFoundError(f"MSD dataframe not found: {args.pkl}")
    dataframe = _load_pkl_dataframe(args.pkl)
    inferred_classes = _infer_num_classes(dataframe, "MSD")
    if inferred_classes != num_classes:
        raise ValueError(f"The baseline expects three classes, found {inferred_classes}.")
    dataset = _build_monai_dataset_from_pkl(
        dataframe,
        "MSD",
        num_classes,
        spatial_size=tuple(args.spatial_size),
        do_resize=args.resize,
    )
    matching_indices = [
        index
        for index, item in enumerate(dataset.data)
        if item["case_name"] == case_name
    ]
    if len(matching_indices) != 1:
        raise ValueError(f"Validation case {case_name} is unavailable in the MSD dataframe.")
    sample = dataset[matching_indices[0]]

    image_tensor = sample["image"].float()
    label_tensor = sample["label"].long()

    device = _resolve_device(args.device)
    model = build_swinunetr(tuple(args.spatial_size), num_classes, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    image = image_tensor[0].cpu().numpy().astype(np.float32, copy=False)
    ground_truth = label_tensor[0].cpu().numpy().astype(np.uint8, copy=False)
    dice_anterior = _dice(ground_truth == 1, prediction == 1)
    dice_posterior = _dice(ground_truth == 2, prediction == 2)
    return PredictionCase(
        image=image,
        ground_truth=ground_truth,
        prediction=prediction,
        title=(f"Baseline seed {args.seed} | MSD fold-{fold} validation "
               f"{case_name.removeprefix('hippocampus_')}"),
        dice_anterior=dice_anterior,
        dice_posterior=dice_posterior,
    )


def _display_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def _initial_indices(case: PredictionCase) -> list[int]:
    if case.ground_truth is None:
        combined_foreground = case.prediction > 0
    else:
        combined_foreground = np.logical_or(
            case.ground_truth > 0, case.prediction > 0
        )
    foreground = np.argwhere(combined_foreground)
    if foreground.size == 0:
        return [size // 2 for size in case.image.shape]
    return [int(np.median(foreground[:, axis])) for axis in range(3)]


def _intensity_window(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, (1, 99))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def show_prediction_case(
    case: PredictionCase,
    *,
    opacity: float,
    save_path: Path | None,
    show: bool,
    show_errors: bool,
) -> None:
    try:
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch
        from matplotlib.widgets import Slider
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for the prediction viewer.") from exc

    axis_names = ("Sagittal", "Coronal", "Axial")
    if show_errors and case.ground_truth is None:
        raise ValueError("--show-errors requires a labeled validation case.")
    indices = _initial_indices(case)
    vmin, vmax = _intensity_window(case.image)
    class_colors = {1: "#e63946", 2: "#3a86ff"}
    mask_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), class_colors[1], class_colors[2]])
    mask_norm = BoundaryNorm(np.arange(-0.5, 3.5), mask_cmap.N)
    error_names = {
        1: "Background → anterior",
        2: "Background → posterior",
        3: "Anterior → background",
        4: "Posterior → background",
        5: "Anterior → posterior",
        6: "Posterior → anterior",
    }
    error_colors = {
        1: "#ff595e",
        2: "#00b4d8",
        3: "#9d0208",
        4: "#023e8a",
        5: "#8338ec",
        6: "#ffbe0b",
    }
    error_cmap = ListedColormap(
        [(0.0, 0.0, 0.0, 0.0)] + [error_colors[code] for code in range(1, 7)]
    )
    error_norm = BoundaryNorm(np.arange(-0.5, 7.5), error_cmap.N)

    if case.ground_truth is None:
        row_specs = [
            ("MRI (test GT unavailable)", np.zeros_like(case.prediction), mask_cmap, mask_norm),
            ("Baseline prediction", case.prediction, mask_cmap, mask_norm),
        ]
    else:
        row_specs = [
            ("Ground truth", case.ground_truth, mask_cmap, mask_norm),
            ("Baseline prediction", case.prediction, mask_cmap, mask_norm),
        ]
        if show_errors:
            row_specs.append(
                (
                    "Directed errors",
                    _error_codes(case.ground_truth, case.prediction),
                    error_cmap,
                    error_norm,
                )
            )

    figure, axes = plt.subplots(
        len(row_specs),
        3,
        squeeze=False,
        figsize=(15, 4.25 * len(row_specs)),
    )
    figure.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.13,
        wspace=0.06,
        hspace=0.18,
    )
    if case.dice_anterior is None or case.dice_posterior is None:
        title = f"{case.title} | official test labels unavailable; Dice not computed"
    else:
        macro_dice = 0.5 * (case.dice_anterior + case.dice_posterior)
        title = (
            f"{case.title} | Dice anterior={case.dice_anterior:.3f}, "
            f"posterior={case.dice_posterior:.3f}, mean={macro_dice:.3f}"
        )
    if show_errors:
        title += " | directed error overlay"
    figure.suptitle(title, fontsize=14)
    try:
        figure.canvas.manager.set_window_title(case.title)
    except AttributeError:
        pass

    image_artists: list[list[object]] = [[] for _ in row_specs]
    mask_artists: list[list[object]] = [[] for _ in row_specs]
    for row, (row_name, mask, cmap, norm) in enumerate(row_specs):
        for axis, (axis_name, index) in enumerate(zip(axis_names, indices, strict=True)):
            plot_axis = axes[row, axis]
            image_artists[row].append(
                plot_axis.imshow(
                    _display_slice(case.image, axis, index),
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
            )
            mask_artists[row].append(
                plot_axis.imshow(
                    np.ma.masked_equal(_display_slice(mask, axis, index), 0),
                    cmap=cmap,
                    norm=norm,
                    alpha=opacity,
                    interpolation="nearest",
                )
            )
            plot_axis.set_title(f"{row_name} | {axis_name} slice {index}")
            plot_axis.set_axis_off()

    sliders = []
    for axis, index in enumerate(indices):
        slider_axis = figure.add_axes([0.055 + axis * 0.325, 0.065, 0.27, 0.03])
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
        for row, (row_name, mask, _cmap, _norm) in enumerate(row_specs):
            image_artists[row][axis].set_data(_display_slice(case.image, axis, index))
            mask_artists[row][axis].set_data(
                np.ma.masked_equal(_display_slice(mask, axis, index), 0)
            )
            axes[row, axis].set_title(
                f"{row_name} | {axis_names[axis]} slice {index}"
            )
        figure.canvas.draw_idle()

    for axis, slider in enumerate(sliders):
        slider.on_changed(lambda value, axis=axis: update_axis(axis, value))

    def on_scroll(event: object) -> None:
        event_axis = getattr(event, "inaxes", None)
        positions = np.argwhere(axes == event_axis)
        if positions.size == 0:
            return
        axis = int(positions[0, 1])
        step = 1 if getattr(event, "button", None) == "up" else -1
        new_value = np.clip(
            sliders[axis].val + step,
            sliders[axis].valmin,
            sliders[axis].valmax,
        )
        sliders[axis].set_val(new_value)

    figure.canvas.mpl_connect("scroll_event", on_scroll)
    legend_handles = [
        Patch(facecolor=class_colors[1], label="Anterior"),
        Patch(facecolor=class_colors[2], label="Posterior"),
    ]
    if show_errors:
        legend_handles.extend(
            Patch(facecolor=error_colors[code], label=error_names[code])
            for code in range(1, 7)
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4 if show_errors else 2,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        case = load_prediction_case(args)
        show_prediction_case(
            case,
            opacity=args.opacity,
            save_path=args.save,
            show=not args.no_show,
            show_errors=args.show_errors,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
