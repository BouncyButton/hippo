#!/usr/bin/env python3
"""Fixed ridge heads: train boundary/swap risk on one held-out fold, test the other."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_GROUPS = {
    "union_three_views": ("union:sagittal", "union:coronal", "union:axial"),
    "classes_three_views": ("classes:sagittal", "classes:coronal", "classes:axial"),
    "all_six_profiles": ("union:sagittal", "union:coronal", "union:axial",
                         "classes:sagittal", "classes:coronal", "classes:axial"),
}


def fixed_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    center, scale = train_x.mean(axis=0), train_x.std(axis=0).clip(min=1e-8)
    x_train = (train_x - center) / scale
    x_test = (test_x - center) / scale
    target = np.log1p(train_y)
    target_mean = target.mean()
    weights = np.linalg.solve(x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
                              x_train.T @ (target - target_mean))
    return np.expm1(np.clip(x_test @ weights + target_mean, 0, 20))


def evaluate(signal: np.ndarray, error: np.ndarray) -> dict:
    count = max(1, int(np.ceil(0.2 * len(signal))))
    selected = np.argsort(-signal)[:count]
    return {
        "pearson_r": float(np.corrcoef(signal, error)[0, 1]) if np.std(signal) > 1e-10 else None,
        "top20_error_capture": float(error[selected].sum() / max(error.sum(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.fold0, args.fold1)]
    cases = [report["cases"] for report in reports]
    if not set(item["case_name"] for item in cases[0]).isdisjoint(item["case_name"] for item in cases[1]):
        raise ValueError("cross-fold patients overlap")
    output = {"schema": "semantic_constraints.cst_teacher.multiview_crossfold_boundary_risk.v1",
              "source_folds": [report["fold"] for report in reports], "directions": {},
              "warning": "Both underlying Swin folds have been studied before. No hyperparameter is selected on the target fold; this is a small-sample cross-fold QC probe, not proof of deployable calibration."}
    for training_fold, target_fold in ((0, 1), (1, 0)):
        train, test = cases[training_fold], cases[target_fold]
        direction = f"fold{training_fold}_to_fold{target_fold}"
        result = {}
        for target_name, target_field in (("boundary", "boundary_error_voxels"), ("swaps", "swap_error_voxels")):
            train_y = np.asarray([item[target_field] for item in train], dtype=np.float64)
            test_y = np.asarray([item[target_field] for item in test], dtype=np.float64)
            methods = {"unsupervised_classes_triview_mean": evaluate(
                np.asarray([item["signals"]["classes:triview_mean"] for item in test]), test_y)}
            for name, features in FEATURE_GROUPS.items():
                train_x = np.asarray([[item["signals"][feature] for feature in features] for item in train])
                test_x = np.asarray([[item["signals"][feature] for feature in features] for item in test])
                predicted = fixed_ridge_predict(train_x, train_y, test_x)
                methods[name] = evaluate(predicted, test_y)
            result[target_name] = methods
        output["directions"][direction] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
