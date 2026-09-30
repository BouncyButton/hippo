#!/usr/bin/env python3
"""Patient-out-of-fold probe of missing-versus-extra hippocampal tissue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .error_types import TYPE_NAMES
from .risk_probe import grouped_folds
from .slice_qc import Standardizer, load_patient_features


FEATURE_KINDS = ("uncertainty", "portable", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--error-types", type=Path, required=True)
    parser.add_argument("--qc-oof-csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def load_qc_scores(path: Path, names: np.ndarray, slices: int, fold: int) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    scores = np.full((len(names), slices), np.nan, dtype=np.float32)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["fold"]) != fold:
                continue
            name, position = row["case_name"], int(row["slice_index"])
            if name not in index or not 0 <= position < slices:
                raise ValueError("QC OOF row does not match target bank")
            patient = index[name]
            if np.isfinite(scores[patient, position]):
                raise ValueError("duplicate QC OOF row")
            scores[patient, position] = float(row["ridge_combined"])
    if not np.isfinite(scores).all():
        raise ValueError("QC OOF scores are incomplete")
    return scores


def crossfit_direction(
    features: np.ndarray,
    error_type: np.ndarray,
    names: np.ndarray,
    *,
    fold: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Fit only on unambiguous missing/extra slices of training patients."""
    if features.shape[:2] != error_type.shape or len(features) != len(names):
        raise ValueError("feature/target/patient alignment mismatch")
    partitions = grouped_folds(names, 5, seed + fold)
    probability = np.full(error_type.shape, np.nan, dtype=np.float32)
    split_reports = []
    for outer, held_names in enumerate(partitions):
        held = np.isin(names, held_names)
        train = ~held
        if set(names[train]) & set(names[held]):
            raise AssertionError("patient leakage")
        actionable = train[:, None] & np.isin(error_type, (1, 2))
        x = features[actionable]
        y = (error_type[actionable] == 1).astype(np.int8)
        if len(x) < 20 or len(np.unique(y)) != 2:
            raise ValueError("insufficient missing/extra examples in outer training split")
        standardizer = Standardizer.fit(x)
        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        model.fit(standardizer.transform(x), y)
        probability[held] = model.predict_proba(standardizer.transform(features[held].reshape(-1, features.shape[-1])))[:, 1].reshape((-1, features.shape[1]))
        split_reports.append({
            "outer_fold": outer,
            "train_patients": names[train].tolist(),
            "test_patients": names[held].tolist(),
            "train_missing_slices": int(y.sum()),
            "train_extra_slices": int(len(y) - y.sum()),
        })
    if not np.isfinite(probability).all():
        raise AssertionError("cross-fitting left unscored slices")
    return probability, split_reports


def directional_metrics(error_type: np.ndarray, probability: np.ndarray, selected: np.ndarray) -> dict:
    actionable = selected & np.isin(error_type, (1, 2))
    y = (error_type[actionable] == 1).astype(np.int8)
    p = probability[actionable]
    if not len(y):
        raise ValueError("selected slice subset has no directional targets")
    hard = p >= 0.5
    confident = (p <= 0.2) | (p >= 0.8)
    result = {
        "selected_slices": int(selected.sum()),
        "directional_slices": int(len(y)),
        "directional_fraction": float(len(y) / selected.sum()),
        "missing_fraction": float(y.mean()),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
        "accuracy": float(np.mean(hard == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)) if len(np.unique(y)) == 2 else None,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "confident_directional_slices": int(confident.sum()),
        "confident_directional_accuracy": float(np.mean(hard[confident] == y[confident])) if confident.any() else None,
    }
    return result


def run_study(features_path: Path, types_path: Path, qc_path: Path, *, fold: int, seed: int) -> dict:
    data = load_patient_features(features_path, require_target=True)
    with np.load(types_path, allow_pickle=False) as target_bank:
        names = np.asarray(target_bank["case_names"]).astype(str)
        error_type = np.asarray(target_bank["error_type"], dtype=np.int8)
        counts = np.asarray(target_bank["error_counts"], dtype=np.int32)
    if not np.array_equal(names, data.names) or error_type.shape != data.target.shape:
        raise ValueError("error taxonomy and feature-bank patients do not align")
    qc = load_qc_scores(qc_path, names, data.slices, fold)
    top_count = int(np.ceil(qc.size * 0.20))
    selected = np.zeros(qc.size, dtype=bool)
    selected[np.argpartition(qc.ravel(), -top_count)[-top_count:]] = True
    selected = selected.reshape(qc.shape)
    whole = np.ones_like(selected, dtype=bool)
    report = {
        "schema": "semantic_constraints.cst_teacher.error_direction_probe.v1",
        "fold": fold,
        "features": str(features_path),
        "target_bank": str(types_path),
        "patients": len(names),
        "distribution": dict(zip(TYPE_NAMES, np.bincount(error_type.ravel(), minlength=len(TYPE_NAMES)).tolist())),
        "selected_distribution": dict(zip(TYPE_NAMES, np.bincount(error_type[selected], minlength=len(TYPE_NAMES)).tolist())),
        "error_voxel_totals": counts.sum(axis=(0, 1)).tolist(),
        "models": {},
    }
    for kind in FEATURE_KINDS:
        predicted, splits = crossfit_direction(data.features[kind], error_type, names, fold=fold, seed=seed)
        report["models"][kind] = {
            "all_directional": directional_metrics(error_type, predicted, whole),
            "qc_top20_directional": directional_metrics(error_type, predicted, selected),
            "splits": splits,
        }
    return report


def main() -> None:
    args = parse_args()
    report = run_study(args.features, args.error_types, args.qc_oof_csv, fold=args.fold, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "fold": args.fold,
        "distribution": report["distribution"],
        "selected_distribution": report["selected_distribution"],
        "metrics": {name: entry["qc_top20_directional"] for name, entry in report["models"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
