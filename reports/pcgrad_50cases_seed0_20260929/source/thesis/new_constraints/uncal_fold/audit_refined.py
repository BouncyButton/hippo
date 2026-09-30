"""Held-out audit of refined mask and raw-MRI uncal-fold descriptors."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .foldedness import CaseFeatures, extract_case_features
from .refined import (
    IMAGE_FEATURE_NAMES,
    REFINED_SLICE_FEATURE_NAMES,
    TRANSITION_FEATURE_NAMES,
    RefinedCaseFeatures,
    extract_refined_case_features,
)


CLEAR_SLICE_FEATURES = (
    "area",
    "height_over_width",
    "superior_notch_depth",
    "superior_bilobedness",
    "superior_peak_separation",
    "superior_protrusion_max",
    "superior_extension_area_max",
    "superior_quadrant_asymmetry",
    "multirun_side_max",
    "hole_fraction",
)

CLEAR_TRANSITION_FEATURES = (
    "relative_area_change",
    "relative_height_change",
    "slice_dice",
    "novel_fraction_after_dilation",
    "novel_superior_fraction",
    "novel_superior_side_max",
    "novel_superior_side_difference",
    "superior_boundary_rise_max",
    "superior_boundary_rise_mean",
)

COMPACT_GEOMETRY_FEATURES = (
    "clear_shape_delta__area",
    "clear_shape_next_delta__area",
    "clear_shape_delta__superior_notch_depth",
    "clear_shape_current__multirun_side_max",
    "transition__superior_boundary_rise_max",
    "transition__novel_superior_side_difference",
    "transition__slice_dice",
)

COMPACT_IMAGE_FEATURES = (
    "image_current__superior_band_left_right_difference_abs",
    "image_current__superior_band_intensity_mean",
    "image_current__superior_band_intensity_std",
    "image_current__central_superior_band_intensity_mean",
    "image_delta__superior_band_intensity_mean",
    "image_current__mask_gradient_p90",
)


@dataclass(frozen=True)
class LoadedCase:
    geometry: CaseFeatures
    refined: RefinedCaseFeatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/uncal_foldedness_refined_20260921"),
    )
    return parser.parse_args()


def _load_cases(root: Path) -> dict[str, LoadedCase]:
    cases = {}
    for label_path in sorted((root / "labelsTr").glob("hippocampus_*.nii.gz")):
        name = label_path.name.removesuffix(".nii.gz")
        image_path = root / "imagesTr" / f"{name}_0000.nii.gz"
        label_nifti = nib.load(str(label_path))
        image_nifti = nib.load(str(image_path))
        if nib.aff2axcodes(label_nifti.affine) != ("R", "A", "S"):
            raise ValueError(f"{name}: expected RAS label")
        labels = np.asarray(label_nifti.dataobj, dtype=np.uint8)
        image = np.asarray(image_nifti.dataobj, dtype=np.float32)
        geometry = extract_case_features(name, labels)
        refined = extract_refined_case_features(
            image,
            labels != 0,
            geometry.low,
            geometry.high,
        )
        cases[name] = LoadedCase(geometry, refined)
    return cases


def _zscore_mapping(
    mapping: dict[int, dict[str, float]],
    names: tuple[str, ...],
) -> dict[int, dict[str, float]]:
    keys = sorted(mapping)
    matrix = np.asarray([[mapping[key][name] for name in names] for key in keys])
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    matrix = (matrix - mean) / scale
    return {
        key: {name: float(matrix[row, column]) for column, name in enumerate(names)}
        for row, key in enumerate(keys)
    }


def _sequence_features(
    mapping: dict[int, dict[str, float]],
    names: tuple[str, ...],
    candidates: list[int],
    high: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    normalized = _zscore_mapping(mapping, names)
    rows = []
    for cut in candidates:
        previous = normalized[cut - 1]
        current = normalized[cut]
        following = normalized[min(cut + 1, high)]
        row = []
        row.extend(current[name] for name in names)
        row.extend(current[name] - previous[name] for name in names)
        row.extend(following[name] - current[name] for name in names)
        row.extend(current[name] - 0.5 * (previous[name] + following[name]) for name in names)
        rows.append(row)
    feature_names = [
        f"{prefix}_{operation}__{name}"
        for operation in ("current", "delta", "next_delta", "local_contrast")
        for name in names
    ]
    return np.asarray(rows), feature_names


def _transition_features(
    case: LoadedCase,
    names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    normalized = _zscore_mapping(case.refined.transition_shape, names)
    matrix = np.asarray(
        [[normalized[cut][name] for name in names] for cut in case.geometry.candidates]
    )
    return matrix, [f"transition__{name}" for name in names]


def _training_position_statistics(
    train_names: list[str],
    cases: dict[str, LoadedCase],
) -> tuple[float, float]:
    positions = np.asarray(
        [
            (cases[name].geometry.target_cut - cases[name].geometry.low)
            / max(cases[name].geometry.high - cases[name].geometry.low, 1)
            for name in train_names
        ]
    )
    median = float(np.median(positions))
    robust_scale = float(1.4826 * np.median(np.abs(positions - median)))
    return median, max(robust_scale, 0.075)


def _candidate_matrix(
    case: LoadedCase,
    mode: str,
    position_statistics: tuple[float, float],
) -> tuple[np.ndarray, list[str]]:
    geometry = case.geometry
    candidates = geometry.candidates
    shape, shape_names = _sequence_features(
        case.refined.slice_shape,
        REFINED_SLICE_FEATURE_NAMES,
        candidates,
        geometry.high,
        "shape",
    )
    transition, transition_names = _transition_features(
        case,
        TRANSITION_FEATURE_NAMES,
    )
    image, image_names = _sequence_features(
        case.refined.slice_image,
        IMAGE_FEATURE_NAMES,
        candidates,
        geometry.high,
        "image",
    )
    clear_shape, clear_shape_names = _sequence_features(
        case.refined.slice_shape,
        CLEAR_SLICE_FEATURES,
        candidates,
        geometry.high,
        "clear_shape",
    )
    clear_transition, clear_transition_names = _transition_features(
        case,
        CLEAR_TRANSITION_FEATURES,
    )
    relative = np.asarray(
        [(cut - geometry.low) / max(geometry.high - geometry.low, 1) for cut in candidates]
    )
    median, scale = position_statistics
    soft_position = -0.5 * ((relative - median) / scale) ** 2
    position = np.column_stack((relative, relative**2, soft_position))
    position_names = ["position__relative", "position__relative_squared", "position__soft_log_prior"]

    clear_matrix = np.column_stack((clear_shape, clear_transition))
    clear_names = clear_shape_names + clear_transition_names
    compact_geometry_indices = [clear_names.index(name) for name in COMPACT_GEOMETRY_FEATURES]
    compact_geometry = clear_matrix[:, compact_geometry_indices]
    compact_image_indices = [image_names.index(name) for name in COMPACT_IMAGE_FEATURES]
    compact_image = image[:, compact_image_indices]
    soft_prior = position[:, 2:3]

    groups = {
        "refined_shape": (np.column_stack((shape, transition)), shape_names + transition_names),
        "image_only": (image, image_names),
        "refined_shape_plus_image": (
            np.column_stack((shape, transition, image)),
            shape_names + transition_names + image_names,
        ),
        "clear_anatomical": (
            clear_matrix,
            clear_names,
        ),
        "clear_anatomical_plus_soft_position": (
            np.column_stack((clear_shape, clear_transition, position)),
            clear_shape_names + clear_transition_names + position_names,
        ),
        "compact_geometry": (
            compact_geometry,
            list(COMPACT_GEOMETRY_FEATURES),
        ),
        "compact_geometry_plus_soft_position": (
            np.column_stack((compact_geometry, soft_prior)),
            [*COMPACT_GEOMETRY_FEATURES, "position__soft_log_prior"],
        ),
        "compact_geometry_image_plus_soft_position": (
            np.column_stack((compact_geometry, compact_image, soft_prior)),
            [
                *COMPACT_GEOMETRY_FEATURES,
                *COMPACT_IMAGE_FEATURES,
                "position__soft_log_prior",
            ],
        ),
        "compact_fixed_components": (
            np.column_stack((compact_geometry, soft_prior)),
            [*COMPACT_GEOMETRY_FEATURES, "position__soft_log_prior"],
        ),
    }
    return groups[mode]


def _fit_predict(
    train_names: list[str],
    validation_names: list[str],
    cases: dict[str, LoadedCase],
    mode: str,
    *,
    penalty: str,
) -> tuple[dict[str, int], np.ndarray, list[str]]:
    position_statistics = _training_position_statistics(train_names, cases)
    rows, targets, weights = [], [], []
    feature_names: list[str] = []
    for name in train_names:
        matrix, feature_names = _candidate_matrix(cases[name], mode, position_statistics)
        count = matrix.shape[0]
        rows.extend(matrix)
        targets.extend(int(cut == cases[name].geometry.target_cut) for cut in cases[name].geometry.candidates)
        weights.extend([1.0 / count] * count)
    kwargs = {
        "C": 0.20,
        "class_weight": "balanced",
        "max_iter": 5000,
    }
    if penalty == "l1":
        kwargs.update({"penalty": "l1", "solver": "liblinear", "C": 0.10})
    model = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))
    model.fit(
        np.asarray(rows),
        np.asarray(targets),
        logisticregression__sample_weight=np.asarray(weights),
    )
    predictions = {}
    for name in validation_names:
        matrix, _ = _candidate_matrix(cases[name], mode, position_statistics)
        scores = model.predict_proba(matrix)[:, 1]
        predictions[name] = cases[name].geometry.candidates[int(np.argmax(scores))]
    return predictions, model.named_steps["logisticregression"].coef_[0], feature_names


def _median_prediction(
    train_names: list[str],
    case: LoadedCase,
    cases: dict[str, LoadedCase],
) -> int:
    median, _ = _training_position_statistics(train_names, cases)
    candidates = np.asarray(case.geometry.candidates)
    relative = (candidates - case.geometry.low) / max(case.geometry.high - case.geometry.low, 1)
    return int(candidates[int(np.argmin(np.abs(relative - median)))])


def _fixed_compact_prediction(
    train_names: list[str],
    case: LoadedCase,
    cases: dict[str, LoadedCase],
) -> int:
    matrix, names = _candidate_matrix(
        case,
        "compact_fixed_components",
        _training_position_statistics(train_names, cases),
    )
    weights = {
        "clear_shape_delta__area": 1.0,
        "clear_shape_next_delta__area": 1.0,
        "clear_shape_delta__superior_notch_depth": 0.25,
        "clear_shape_current__multirun_side_max": 0.25,
        "transition__superior_boundary_rise_max": 0.50,
        "transition__novel_superior_side_difference": 0.25,
        "transition__slice_dice": 0.25,
        # This is intentionally soft: evidence can overcome the population prior.
        "position__soft_log_prior": 0.25,
    }
    score = matrix @ np.asarray([weights[name] for name in names])
    return case.geometry.candidates[int(np.argmax(score))]


def _metrics(predictions: dict[str, int], cases: dict[str, LoadedCase]) -> dict[str, float]:
    errors = np.asarray(
        [abs(cut - cases[name].geometry.target_cut) for name, cut in predictions.items()]
    )
    return {
        "count": int(errors.size),
        "mae_slices": float(errors.mean()),
        "exact_fraction": float((errors == 0).mean()),
        "within_1_fraction": float((errors <= 1).mean()),
        "within_2_fraction": float((errors <= 2).mean()),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = _load_cases(args.dataset_root)
    splits = json.loads((args.dataset_root / "splits_final.json").read_text())
    modes = (
        "refined_shape",
        "image_only",
        "refined_shape_plus_image",
        "clear_anatomical",
        "clear_anatomical_plus_soft_position",
        "compact_geometry",
        "compact_geometry_plus_soft_position",
        "compact_geometry_image_plus_soft_position",
    )
    predictions = {"median_relative": {}}
    predictions.update({mode: {} for mode in modes})
    predictions["clear_anatomical_sparse"] = {}
    predictions["compact_fixed_formula"] = {}
    coefficients: dict[str, list[np.ndarray]] = {mode: [] for mode in (*modes, "clear_anatomical_sparse")}
    coefficient_names: dict[str, list[str]] = {}
    atypicality = {}
    validation_fold = {}

    for fold_index, split in enumerate(splits):
        train, validation = list(split["train"]), list(split["val"])
        median, _ = _training_position_statistics(train, cases)
        for name in validation:
            geometry = cases[name].geometry
            relative_target = (geometry.target_cut - geometry.low) / max(geometry.high - geometry.low, 1)
            atypicality[name] = abs(relative_target - median)
            validation_fold[name] = fold_index
            predictions["median_relative"][name] = _median_prediction(train, cases[name], cases)
            predictions["compact_fixed_formula"][name] = _fixed_compact_prediction(
                train, cases[name], cases
            )
        for mode in modes:
            fold_predictions, fold_coefficients, names = _fit_predict(
                train, validation, cases, mode, penalty="l2"
            )
            predictions[mode].update(fold_predictions)
            coefficients[mode].append(fold_coefficients)
            coefficient_names[mode] = names
        fold_predictions, fold_coefficients, names = _fit_predict(
            train, validation, cases, "clear_anatomical", penalty="l1"
        )
        predictions["clear_anatomical_sparse"].update(fold_predictions)
        coefficients["clear_anatomical_sparse"].append(fold_coefficients)
        coefficient_names["clear_anatomical_sparse"] = names

    all_names = set(cases)
    if any(set(method_predictions) != all_names for method_predictions in predictions.values()):
        raise ValueError("five-fold predictions must cover all cases")
    metrics = {name: _metrics(values, cases) for name, values in predictions.items()}
    atypical_names = set(
        sorted(atypicality, key=atypicality.get, reverse=True)[: len(cases) // 3]
    )
    atypical_metrics = {
        method: _metrics({name: cut for name, cut in values.items() if name in atypical_names}, cases)
        for method, values in predictions.items()
    }
    coefficient_summary = {}
    for mode, arrays in coefficients.items():
        matrix = np.stack(arrays)
        names = coefficient_names[mode]
        coefficient_summary[mode] = sorted(
            [
                {
                    "feature": name,
                    "mean": float(matrix[:, index].mean()),
                    "mean_absolute": float(np.abs(matrix[:, index]).mean()),
                    "nonzero_folds": int(np.count_nonzero(np.abs(matrix[:, index]) > 1e-10)),
                }
                for index, name in enumerate(names)
            ],
            key=lambda row: row["mean_absolute"],
            reverse=True,
        )

    with (args.output_dir / "case_predictions.csv").open("w", newline="") as handle:
        fields = ["case_name", "validation_fold", "target_cut", "relative_cut_atypicality", *sorted(predictions)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(cases):
            row = {
                "case_name": name,
                "validation_fold": validation_fold[name],
                "target_cut": cases[name].geometry.target_cut,
                "relative_cut_atypicality": atypicality[name],
            }
            row.update({method: values[name] for method, values in predictions.items()})
            writer.writerow(row)

    summary = {
        "case_count": len(cases),
        "feature_source": {
            "shape": "ground-truth foreground union, without A/P labels",
            "image": "robustly normalized native T1 within/around the union ROI",
            "target": "A/P labels used only for best-fit cut and held-out scoring",
        },
        "metrics": metrics,
        "atypical_third_metrics": atypical_metrics,
        "coefficient_summary": coefficient_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ordered = sorted(metrics, key=lambda name: metrics[name]["mae_slices"])
    lines = [
        "# Refined uncal-fold descriptor audit",
        "",
        "Five-fold subject-held-out results on all 260 Task04 training masks/images.",
        "The refined geometry is mirror-invariant because the release mixes left/right",
        "hippocampus crops without reliable laterality metadata.",
        "",
        "| method | MAE | exact | within 1 | within 2 | p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        row = metrics[name]
        lines.append(
            f"| {name} | {row['mae_slices']:.3f} | {row['exact_fraction']:.1%} | "
            f"{row['within_1_fraction']:.1%} | {row['within_2_fraction']:.1%} | "
            f"{row['p90_absolute_error']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Most atypical third",
            "",
            "| method | MAE | exact | within 1 | p90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ordered:
        row = atypical_metrics[name]
        lines.append(
            f"| {name} | {row['mae_slices']:.3f} | {row['exact_fraction']:.1%} | "
            f"{row['within_1_fraction']:.1%} | {row['p90_absolute_error']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Stable coefficients in the best compact model",
            "",
            "| feature | mean coefficient | mean absolute | nonzero folds |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in coefficient_summary["compact_geometry_image_plus_soft_position"][:16]:
        lines.append(
            f"| `{row['feature']}` | {row['mean']:.3f} | {row['mean_absolute']:.3f} | "
            f"{row['nonzero_folds']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The repeatable evidence is a transition pattern, not one static foldedness",
            "number: cross-sectional area grows across the current and following slice;",
            "the superior boundary rises or changes notch configuration; and the T1 band",
            "immediately above the hippocampus becomes darker, more asymmetric, and more",
            "edge-rich. Shape-only and image-only scores have catastrophic end-slice",
            "outliers. The compact combination works only with a soft position prior, which",
            "must remain defeasible rather than becoming a hard anatomical axiom.",
            "",
            "The failed sparse and fixed-formula controls show that the current result does",
            "not justify calling any single scalar an uncal-apex detector. The compact score",
            "is the appropriate candidate for a fuzzy LTN predicate, followed by a baseline",
            "comparison on predicted rather than ground-truth foreground masks.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python -m thesis.new_constraints.uncal_fold.audit_refined",
            "```",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
