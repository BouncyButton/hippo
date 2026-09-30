#!/usr/bin/env python3
"""Audit descending hippocampal contours and their relation to segmentation errors.

The native MSD grid is axis-aligned: y increases anteriorly and z superiorly.
This script analyzes the union of the two hippocampal classes. It verifies that
each saved prediction's reference mask is an exact center-padded native label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets/Dataset101_MSD"
DEFAULT_PREDICTIONS = REPO_ROOT / (
    "experiments/uncal_fold_early_stopping_20260921/voxel_audit/"
    "baseline_seed0/error_maps"
)


def profile(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return robust coronal profiles on slices with meaningful foreground area."""
    if mask.ndim != 3 or not mask.any():
        raise ValueError("expected a nonempty 3D foreground mask")
    area = mask.sum(axis=(0, 2))
    ys = np.flatnonzero(area >= max(5, 0.1 * int(area.max())))
    if len(ys) < 3:
        raise ValueError("too few supported coronal slices")
    lower = np.empty(len(ys))
    upper = np.empty(len(ys))
    center = np.empty(len(ys))
    for index, y in enumerate(ys):
        _, zs = np.where(mask[:, y, :])
        lower[index], upper[index] = np.quantile(zs, [0.1, 0.9])
        center[index] = zs.mean()
    return {
        "y": ys,
        "area": area[ys].astype(float),
        "lower": lower,
        "upper": upper,
        "center": center,
        "height": upper - lower,
    }


def profile_metrics(values: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    """Measure directional trend and distance to a decreasing step function."""
    fitted = IsotonicRegression(increasing=False).fit_transform(ys, values)
    linear = np.polyval(np.polyfit(ys, values, 1), ys)
    differences = np.diff(values)
    return {
        "spearman_y": float(spearmanr(ys, values).statistic) if np.ptp(values) else 0.0,
        "end_minus_start_voxels": float(values[-1] - values[0]),
        "isotonic_rmse_voxels": float(np.sqrt(np.mean((values - fitted) ** 2))),
        "linear_rmse_voxels": float(np.sqrt(np.mean((values - linear) ** 2))),
        "upward_jumps_over_1_voxel": int(np.count_nonzero(differences > 1)),
        "upward_excess_voxels": float(np.maximum(differences - 1, 0).sum()),
    }


def mask_metrics(mask: np.ndarray) -> dict[str, object]:
    curves = profile(mask)
    return {
        "foreground_voxels": int(mask.sum()),
        "supported_slices": len(curves["y"]),
        "profiles": {
            name: profile_metrics(curves[name], curves["y"])
            for name in ("upper", "lower", "center", "height", "area")
        },
    }


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.1)),
        "q90": float(np.quantile(array, 0.9)),
        "mean": float(array.mean()),
    }


def correlation(x: list[float], y: list[float]) -> float:
    if not np.ptp(x) or not np.ptp(y):
        return 0.0
    return float(spearmanr(x, y).statistic)


def top_fraction_capture(scores: list[float], errors: list[int], fraction: float = 0.2) -> float:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(-np.asarray(scores))[:count]
    return float(np.asarray(errors)[order].sum() / sum(errors))


def fitted_silhouette_steps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the actual superior/inferior sagittal silhouette, including endpoints."""
    projection = mask.any(axis=0)
    ys = np.flatnonzero(projection.any(axis=1))
    zs = np.arange(mask.shape[2])
    upper = np.asarray([zs[projection[y]].max() for y in ys], dtype=float)
    lower = np.asarray([zs[projection[y]].min() for y in ys], dtype=float)
    regression = IsotonicRegression(increasing=False)
    return (
        ys,
        regression.fit_transform(ys, upper),
        regression.fit_transform(ys, lower),
    )


def step_neighborhood(
    shape: tuple[int, int, int], steps: tuple[np.ndarray, np.ndarray, np.ndarray], radius: int
) -> np.ndarray:
    """A y-z band around either fitted step, broadcast across lateral x."""
    ys, upper, lower = steps
    zs = np.arange(shape[2])
    band = np.zeros(shape[1:], dtype=bool)
    band[ys] = (np.abs(zs[None, :] - upper[:, None]) <= radius) | (
        np.abs(zs[None, :] - lower[:, None]) <= radius
    )
    return band[None, :, :]


def voxel_step_audit(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    """Cross reference/predicted step envelopes with actual voxel errors.

    Reference-derived signals are oracle measurements; they cannot be used at
    inference without an independent patient-specific step estimator.
    """
    extra = prediction & ~truth
    missed = truth & ~prediction
    error = extra | missed
    eligible = truth | prediction
    zs = np.arange(truth.shape[2])[None, :]
    truth_projection = truth.any(axis=0)
    prediction_projection = prediction.any(axis=0)
    truth_upper = np.where(truth_projection, zs, -1).max(axis=1)
    truth_lower = np.where(truth_projection, zs, truth.shape[2]).min(axis=1)
    prediction_upper = np.where(prediction_projection, zs, -1).max(axis=1)
    prediction_lower = np.where(prediction_projection, zs, truth.shape[2]).min(axis=1)
    z_grid = np.arange(truth.shape[2])[None, None, :]
    outside_truth = (z_grid > truth_upper[None, :, None]) | (z_grid < truth_lower[None, :, None])
    outside_prediction = (z_grid > prediction_upper[None, :, None]) | (
        z_grid < prediction_lower[None, :, None]
    )
    result: dict[str, object] = {
        "extra_voxels": int(extra.sum()),
        "missed_voxels": int(missed.sum()),
        "eligible_union_voxels": int(eligible.sum()),
        "oracle_exterior_extra_voxels": int((extra & outside_truth).sum()),
        "oracle_exterior_missed_voxels": int((missed & outside_prediction).sum()),
        "neighborhoods": {},
    }
    truth_steps = fitted_silhouette_steps(truth)
    prediction_steps = fitted_silhouette_steps(prediction)
    for source, steps in (("reference", truth_steps), ("prediction", prediction_steps)):
        for radius in (1, 2):
            band = step_neighborhood(truth.shape, steps, radius)
            result["neighborhoods"][f"{source}_{radius}"] = {
                "error_voxels": int((error & band).sum()),
                "eligible_voxels": int((eligible & band).sum()),
            }
    truth_y, truth_top, truth_bottom = truth_steps
    pred_y, pred_top, pred_bottom = prediction_steps
    shared = np.intersect1d(truth_y, pred_y)
    truth_indices = np.searchsorted(truth_y, shared)
    pred_indices = np.searchsorted(pred_y, shared)
    displacement = np.r_[
        np.abs(truth_top[truth_indices] - pred_top[pred_indices]),
        np.abs(truth_bottom[truth_indices] - pred_bottom[pred_indices]),
    ]
    result["matched_step_positions"] = len(displacement)
    result["matched_step_positions_within_1_voxel"] = int((displacement <= 1).sum())
    result["matched_step_positions_over_2_voxels"] = int((displacement > 2).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads((args.dataset / "splits_final.json").read_text())[args.fold]
    validation_names = set(split["val"])
    label_paths = sorted((args.dataset / "labelsTr").glob("*.nii.gz"))
    if not label_paths:
        raise ValueError("no native labels found")
    cases = []
    for path in label_paths:
        name = path.name.removesuffix(".nii.gz")
        image = nib.load(path)
        if not np.allclose(image.affine[:3, :3], np.eye(3)):
            raise ValueError(f"unexpected axis orientation or spacing: {name}")
        label = np.asarray(image.dataobj)
        if not np.isin(label, [0, 1, 2]).all():
            raise ValueError(f"invalid class ID: {name}")
        row: dict[str, object] = {
            "case": name,
            "validation_fold": args.fold if name in validation_names else None,
            "truth": mask_metrics(label > 0),
        }
        if name in validation_names:
            with np.load(args.predictions / f"{name}.npz") as archive:
                archive_gt = archive["ground_truth"]
                prediction = archive["prediction"]
            if archive_gt.shape != prediction.shape:
                raise ValueError(f"prediction shape mismatch: {name}")
            delta = np.asarray(archive_gt.shape) - np.asarray(label.shape)
            if (delta < 0).any():
                raise ValueError(f"prediction grid smaller than label: {name}")
            before = delta // 2
            expected = np.pad(label, tuple(zip(before, delta - before)))
            if not np.array_equal(expected, archive_gt):
                raise ValueError(f"saved reference differs from native label: {name}")
            if not np.isin(prediction, [0, 1, 2]).all():
                raise ValueError(f"invalid prediction class ID: {name}")
            truth_foreground = archive_gt > 0
            prediction_foreground = prediction > 0
            boundary_errors = int(np.count_nonzero(truth_foreground != prediction_foreground))
            row["prediction"] = mask_metrics(prediction_foreground)
            row["boundary_error_voxels"] = boundary_errors
            row["boundary_error_rate"] = boundary_errors / max(1, int(truth_foreground.sum()))
            row["ap_swap_voxels"] = int(np.count_nonzero(
                truth_foreground & prediction_foreground & (archive_gt != prediction)
            ))
            row["voxel_step"] = voxel_step_audit(truth_foreground, prediction_foreground)
            curves = profile(prediction_foreground)
            ys = curves["y"]
            jumping = np.diff(curves["upper"]) > 1
            flagged = np.unique(np.r_[ys[:-1][jumping], ys[1:][jumping]])
            error_by_y = (truth_foreground != prediction_foreground).sum(axis=(0, 2))
            truth_by_y = truth_foreground.sum(axis=(0, 2))
            row["local_jump"] = {
                "flagged_slices": len(flagged),
                "supported_slices": len(ys),
                "flagged_boundary_errors": int(error_by_y[flagged].sum()),
                "supported_boundary_errors": int(error_by_y[ys].sum()),
                "flagged_truth_foreground": int(truth_by_y[flagged].sum()),
                "supported_truth_foreground": int(truth_by_y[ys].sum()),
            }
        cases.append(row)

    if {row["case"] for row in cases if row["validation_fold"] == args.fold} != validation_names:
        raise ValueError("missing validation labels")
    validation = [row for row in cases if "prediction" in row]
    report: dict[str, object] = {
        "dataset": "MSD hippocampus",
        "reference_cases": len(cases),
        "prediction_cases": len(validation),
        "prediction_source": str(args.predictions),
        "definitions": {
            "axis": "native y increases anteriorly; z increases superiorly",
            "mask": "union of labels 1 and 2",
            "profile": "10th/90th z quantiles, mean z, and area on slices with area >= max(5, 10% of peak)",
            "step_fit": "decreasing isotonic regression, unconstrained number of plateaus",
            "jump": "positive adjacent-slice change exceeding 1 voxel",
            "association": "Spearman correlation; top 20% capture by score",
            "voxel_localization": (
                "exact sagittal silhouette extrema on all occupied slices, then decreasing "
                "isotonic superior/inferior steps; count errors within 1 or 2 z voxels "
                "of the steps across x, among the reference/prediction foreground union"
            ),
            "oracle_warning": "reference steps use the annotation and are unavailable at inference",
        },
        "reference_summary": {},
        "prediction_summary": {},
        "associations": {},
        "local_jump_summary": {},
        "voxel_localization": {},
        "cases": cases,
    }
    for source, rows, key in (("reference_summary", cases, "truth"),
                              ("prediction_summary", validation, "prediction")):
        metrics = report[source]
        for field in ("upper", "lower", "center", "height", "area"):
            metrics[field] = {
                metric: summary([row[key]["profiles"][field][metric] for row in rows])
                for metric in (
                    "spearman_y", "end_minus_start_voxels", "isotonic_rmse_voxels",
                    "linear_rmse_voxels", "upward_jumps_over_1_voxel",
                    "upward_excess_voxels",
                )
            }
    errors = [row["boundary_error_voxels"] for row in validation]
    rates = [row["boundary_error_rate"] for row in validation]
    scores = {
        "upper_isotonic_rmse": [row["prediction"]["profiles"]["upper"]["isotonic_rmse_voxels"] for row in validation],
        "upper_upward_excess": [row["prediction"]["profiles"]["upper"]["upward_excess_voxels"] for row in validation],
        "center_isotonic_rmse": [row["prediction"]["profiles"]["center"]["isotonic_rmse_voxels"] for row in validation],
        "lower_isotonic_rmse": [row["prediction"]["profiles"]["lower"]["isotonic_rmse_voxels"] for row in validation],
        "predicted_volume": [row["prediction"]["foreground_voxels"] for row in validation],
    }
    report["associations"] = {
        name: {
            "boundary_error_count_spearman": correlation(score, errors),
            "boundary_error_rate_spearman": correlation(score, rates),
            "top20_boundary_error_capture": top_fraction_capture(score, errors),
        }
        for name, score in scores.items()
    }
    local = {
        key: sum(row["local_jump"][key] for row in validation)
        for key in (
            "flagged_slices", "supported_slices", "flagged_boundary_errors",
            "supported_boundary_errors", "flagged_truth_foreground",
            "supported_truth_foreground",
        )
    }
    local["cases_with_flagged_slices"] = sum(row["local_jump"]["flagged_slices"] > 0 for row in validation)
    local["boundary_error_capture"] = local["flagged_boundary_errors"] / max(1, local["supported_boundary_errors"])
    local["truth_foreground_coverage"] = local["flagged_truth_foreground"] / max(1, local["supported_truth_foreground"])
    local["boundary_error_enrichment_vs_foreground"] = (
        local["boundary_error_capture"] / max(1e-12, local["truth_foreground_coverage"])
    )
    report["local_jump_summary"] = local
    voxel = {
        key: sum(row["voxel_step"][key] for row in validation)
        for key in (
            "extra_voxels", "missed_voxels", "eligible_union_voxels",
            "oracle_exterior_extra_voxels", "oracle_exterior_missed_voxels",
            "matched_step_positions", "matched_step_positions_within_1_voxel",
            "matched_step_positions_over_2_voxels",
        )
    }
    voxel["oracle_exterior_error_recall"] = (
        voxel["oracle_exterior_extra_voxels"] + voxel["oracle_exterior_missed_voxels"]
    ) / max(1, voxel["extra_voxels"] + voxel["missed_voxels"])
    voxel["neighborhoods"] = {}
    for source in ("reference", "prediction"):
        for radius in (1, 2):
            key = f"{source}_{radius}"
            error_count = sum(row["voxel_step"]["neighborhoods"][key]["error_voxels"] for row in validation)
            eligible_count = sum(row["voxel_step"]["neighborhoods"][key]["eligible_voxels"] for row in validation)
            voxel["neighborhoods"][key] = {
                "error_voxels": error_count,
                "eligible_voxels": eligible_count,
                "error_recall": error_count / max(1, voxel["extra_voxels"] + voxel["missed_voxels"]),
                "eligible_coverage": eligible_count / max(1, voxel["eligible_union_voxels"]),
                "error_fraction_among_eligible": error_count / max(1, eligible_count),
            }
    report["voxel_localization"] = voxel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
