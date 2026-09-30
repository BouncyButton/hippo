#!/usr/bin/env python3
"""Compare a model's native A/P cut with its frozen-feature probe cut."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .audit_predicted_foreground import largest_foreground_component
from .foldedness import best_fit_first_anterior_slice


ROOT = Path(__file__).resolve().parents[3]


def metrics(errors: np.ndarray) -> dict[str, object]:
    return {
        "count": int(errors.size),
        "mae_slices": float(errors.mean()),
        "exact_fraction": float((errors == 0).mean()),
        "within_1_fraction": float((errors <= 1).mean()),
        "within_2_fraction": float((errors <= 2).mean()),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
        "maximum_absolute_error": int(errors.max()),
        "error_distribution": {
            str(int(error)): int(count)
            for error, count in zip(*np.unique(errors, return_counts=True), strict=True)
        },
    }


def main() -> None:
    rows = list(
        csv.DictReader(
            (
                ROOT
                / "docs/experiments/uncal_feature_probe_20260921/per_case.csv"
            ).open(encoding="utf-8")
        )
    )
    prediction_root = (
        ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit"
    )
    folders = {
        "unaugmented": "baseline_seed0",
        "augmented": "augmentation_seed0",
    }
    output = {}
    for model_name, folder in folders.items():
        model_rows = [row for row in rows if row["model"] == model_name]
        native_errors, feature_errors = [], []
        per_case = []
        for row in model_rows:
            archive_path = (
                prediction_root / folder / "error_maps" / f"{row['case']}.npz"
            )
            with np.load(archive_path) as archive:
                prediction = archive["prediction"].astype(np.uint8)
            union, _, _ = largest_foreground_component(prediction != 0)
            cleaned_prediction = np.where(union, prediction, 0)
            native_cut, _, _ = best_fit_first_anterior_slice(cleaned_prediction)
            target = int(row["target_cut"])
            native_error = abs(native_cut - target)
            feature_error = int(row["feature_only_absolute_error"])
            native_errors.append(native_error)
            feature_errors.append(feature_error)
            per_case.append(
                {
                    "case": row["case"],
                    "target_cut": target,
                    "native_cut": native_cut,
                    "native_absolute_error": native_error,
                    "feature_cut": int(row["feature_only_cut"]),
                    "feature_absolute_error": feature_error,
                }
            )
        native = np.asarray(native_errors)
        feature = np.asarray(feature_errors)
        output[model_name] = {
            "native_segmentation_cut": metrics(native),
            "frozen_feature_cut": metrics(feature),
            "paired_feature_minus_native": {
                "feature_better_cases": int((feature < native).sum()),
                "equal_cases": int((feature == native).sum()),
                "feature_worse_cases": int((feature > native).sum()),
                "mean_error_change": float((feature - native).mean()),
            },
            "per_case": per_case,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
