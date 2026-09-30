#!/usr/bin/env python3
"""Patient-grouped CV and artifact training for CST-conditioned slice QC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .risk_probe import grouped_folds
from .slice_qc import (
    PatientFeatures,
    fit_temporal,
    load_patient_features,
    ridge_fit,
    ridge_predict,
    save_ridge,
    temporal_predict,
)
from .summarize_risk_utility import within_case_capture


ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_KINDS = ("uncertainty", "portable", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold0-features", type=Path, required=True)
    parser.add_argument("--fold1-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def select_alpha(features: np.ndarray, target: np.ndarray, seed: int) -> float:
    groups = np.arange(len(features)).astype(str)
    partitions = grouped_folds(groups, min(4, len(groups)), seed)
    errors = []
    for alpha in ALPHAS:
        losses = []
        for names in partitions:
            held_out = np.isin(groups, names)
            model = ridge_fit(features[~held_out], target[~held_out], alpha)
            prediction = ridge_predict(model, features[held_out])
            losses.append(float(np.mean((prediction - target[held_out]) ** 2)))
        errors.append(np.mean(losses))
    return ALPHAS[int(np.argmin(errors))]


def metrics(target: np.ndarray, prediction: np.ndarray, names: np.ndarray) -> dict[str, float]:
    y = target.reshape(-1)
    p = prediction.reshape(-1)
    if len(y) != len(p):
        raise ValueError("target and prediction size mismatch")
    groups = np.repeat(names, target.shape[1])
    return {
        "slice_pearson": float(np.corrcoef(y, p)[0, 1]) if p.std() > 1e-9 else 0.0,
        "slice_mae": float(np.mean(np.abs(y - p))),
        "slice_mse": float(np.mean((y - p) ** 2)),
        "top20_error_capture": top20_capture(target, prediction),
        "within_patient_top4_capture": within_case_capture(y, p, groups, 4)["mean_error_mass_capture"],
    }


def top20_capture(target: np.ndarray, prediction: np.ndarray) -> float:
    y, p = target.reshape(-1), prediction.reshape(-1)
    count = max(1, int(np.ceil(len(y) * 0.20)))
    selected = np.argpartition(p, -count)[-count:]
    return float(y[selected].sum() / y.sum())


def bootstrap_capture_difference(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    samples: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        chosen = rng.integers(0, len(target), size=len(target))
        first_capture = top20_capture(target[chosen], first[chosen])
        second_capture = top20_capture(target[chosen], second[chosen])
        differences.append(second_capture - first_capture)
    return {
        "point_difference": top20_capture(target, second) - top20_capture(target, first),
        "bootstrap_lower95": float(np.quantile(differences, 0.025)),
        "bootstrap_upper95": float(np.quantile(differences, 0.975)),
    }


def run_fold(data: PatientFeatures, *, fold: int, args: argparse.Namespace) -> dict[str, Any]:
    if data.target is None:
        raise ValueError("training requires slice targets")
    groups = data.names
    if args.outer_folds > len(groups):
        raise ValueError("outer folds exceed patient count")
    partitions = grouped_folds(groups, args.outer_folds, args.seed + fold)
    predictions = {
        "entropy": np.empty_like(data.target),
        **{f"ridge_{kind}": np.empty_like(data.target) for kind in FEATURE_KINDS},
        "temporal_portable": np.empty_like(data.target),
        "temporal_combined": np.empty_like(data.target),
    }
    predictions["entropy"] = data.features["uncertainty"][:, :, 0]
    split_reports = []
    for outer, test_names in enumerate(partitions):
        test = np.isin(groups, test_names)
        train = ~test
        train_indices = np.flatnonzero(train)
        test_indices = np.flatnonzero(test)
        if np.intersect1d(train_indices, test_indices).size:
            raise AssertionError("patient leakage across outer split")
        rng = np.random.default_rng(args.seed + fold * 100 + outer)
        shuffled = rng.permutation(len(train_indices))
        inner_count = min(len(train_indices) - 1, max(4, int(np.ceil(len(train_indices) * 0.20))))
        inner_val = shuffled[:inner_count]
        inner_train = shuffled[inner_count:]
        split_report: dict[str, Any] = {
            "outer_fold": outer,
            "train_patients": groups[train].tolist(),
            "test_patients": groups[test].tolist(),
            "inner_train_count": len(inner_train),
            "inner_validation_count": len(inner_val),
            "models": {},
        }
        for kind in FEATURE_KINDS:
            x_train, x_test = data.features[kind][train], data.features[kind][test]
            y_train = data.target[train]
            alpha = select_alpha(x_train, y_train, args.seed + fold * 100 + outer)
            ridge = ridge_fit(x_train, y_train, alpha)
            predictions[f"ridge_{kind}"][test] = ridge_predict(ridge, x_test)
            split_report["models"][f"ridge_{kind}"] = {"alpha": alpha}
            if kind in ("portable", "combined"):
                temporal, best_epoch = fit_temporal(
                    x_train,
                    y_train,
                    selection_train=inner_train,
                    selection_val=inner_val,
                    seed=args.seed + fold * 100 + outer,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    device=args.device,
                )
                predictions[f"temporal_{kind}"][test] = temporal_predict(temporal, x_test, args.device)
                split_report["models"][f"temporal_{kind}"] = {"selected_epoch": best_epoch}
        split_reports.append(split_report)
        print(json.dumps({"fold": fold, "outer": outer, "models": split_report["models"]}), flush=True)

    model_metrics = {name: metrics(data.target, value, data.names) for name, value in predictions.items()}
    comparisons = {
        name: bootstrap_capture_difference(data.target, predictions["ridge_uncertainty"], value, seed=args.seed + fold)
        for name, value in predictions.items()
        if name not in ("entropy", "ridge_uncertainty")
    }
    return {
        "fold": fold,
        "case_names": data.names.tolist(),
        "patients": data.count,
        "slices_per_patient": data.slices,
        "feature_source": str(data.source),
        "target": data.target.tolist(),
        "predictions": {key: value.tolist() for key, value in predictions.items()},
        "metrics": model_metrics,
        "capture_improvement_over_ridge_uncertainty": comparisons,
        "outer_splits": split_reports,
    }


def fit_final_artifacts(data: PatientFeatures, *, fold: int, args: argparse.Namespace) -> list[str]:
    if data.target is None:
        raise ValueError("final fitting requires targets")
    root = args.output_dir / "models" / f"fold{fold}"
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in FEATURE_KINDS:
        features = data.features[kind]
        alpha = select_alpha(features, data.target, args.seed + fold)
        model = ridge_fit(features, data.target, alpha)
        path = root / f"ridge_{kind}.npz"
        save_ridge(path, model, kind=kind, metadata={"fold": fold, "training_patients": data.names.tolist()})
        paths.append(str(path))
        if kind in ("portable", "combined"):
            indices = np.random.default_rng(args.seed + fold).permutation(data.count)
            val_count = max(4, int(np.ceil(data.count * 0.20)))
            temporal, _ = fit_temporal(
                features,
                data.target,
                selection_train=indices[val_count:],
                selection_val=indices[:val_count],
                seed=args.seed + fold,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=args.device,
            )
            temporal["kind"] = kind
            temporal["fold"] = fold
            temporal["training_patients"] = data.names.tolist()
            path = root / f"temporal_{kind}.pt"
            torch.save(temporal, path)
            paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_patient_features(path, require_target=True) for path in (args.fold0_features, args.fold1_features)]
    overlap = set(datasets[0].names) & set(datasets[1].names)
    if overlap:
        raise ValueError(f"fold validation patients overlap: {sorted(overlap)[:5]}")
    reports = []
    for fold, data in enumerate(datasets):
        reports.append(run_fold(data, fold=fold, args=args))
    artifacts = [fit_final_artifacts(data, fold=fold, args=args) for fold, data in enumerate(datasets)]
    report = {
        "schema": "semantic_constraints.cst_teacher.slice_qc_experiment.v1",
        "configuration": {"outer_folds": args.outer_folds, "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience},
        "folds": reports,
        "model_artifacts": artifacts,
        "warning": "Patient-grouped validation-head CV on two already-analysed MSD folds; not prospective external validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Learnable CST slice-quality head",
        "",
        "Five-fold patient-grouped out-of-fold prediction within each 52-patient validation set.",
        "",
        "| Fold | Model | Slice r | MAE | Top-20% error capture | Within-patient top-4 capture |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in reports:
        for name, metric in item["metrics"].items():
            lines.append(
                f"| {item['fold']} | {name} | {metric['slice_pearson']:.3f} | "
                f"{metric['slice_mae']:.4f} | {metric['top20_error_capture']:.1%} | "
                f"{metric['within_patient_top4_capture']:.1%} |"
            )
    lines.extend(("", "Saved models in `models/fold0/` and `models/fold1/`.", ""))
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        model_names = list(reports[0]["predictions"])
        writer.writerow(("fold", "case_name", "slice_index", "true_error", *model_names))
        for item in reports:
            target = np.asarray(item["target"])
            for patient_index, name in enumerate(item["case_names"]):
                for slice_index in range(item["slices_per_patient"]):
                    writer.writerow(
                        (item["fold"], name, slice_index, float(target[patient_index, slice_index]),
                         *(item["predictions"][model][patient_index][slice_index] for model in model_names))
                    )
    print((args.output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
