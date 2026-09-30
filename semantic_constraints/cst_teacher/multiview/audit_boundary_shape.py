#!/usr/bin/env python3
"""Test whether cross-view whole-mask shape priors flag boundary or A/P errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluate_predictions import prediction_map
from ..train_teacher import build_loaders
from .cut_model import VIEW_AXIS


def area_profile(label: np.ndarray, view: str, *, include_classes: bool) -> np.ndarray:
    """Fixed-coordinate, peak-normalized whole-hippocampus area profile."""
    if label.ndim != 3:
        raise ValueError("expected a 3D label")
    axis = VIEW_AXIS[view]
    reduce_axes = tuple(index for index in range(3) if index != axis)
    categories = (1, 2) if include_classes else (0,)
    profiles = []
    for category in categories:
        mask = (label > 0) if category == 0 else (label == category)
        profile = mask.sum(axis=reduce_axes).astype(np.float32)
        profiles.append(profile / max(float(profile.max()), 1.0))
    return np.concatenate(profiles)


def fit_profile_pca(training: np.ndarray, components: int = 5) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean(axis=0)
    _, _, vectors = np.linalg.svd(training - mean, full_matrices=False)
    return mean, vectors[:components]


def reconstruction_error(samples: np.ndarray, mean: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    centered = samples - mean
    residual = centered - centered @ vectors.T @ vectors
    return np.sqrt(np.mean(residual ** 2, axis=1))


def correlation(signal: np.ndarray, target: np.ndarray) -> float:
    if np.std(signal) < 1e-10 or np.std(target) < 1e-10:
        return float("nan")
    return float(np.corrcoef(signal, target)[0, 1])


def capture(signal: np.ndarray, errors: np.ndarray) -> float:
    count = max(1, int(np.ceil(len(signal) * 0.2)))
    return float(errors[np.argsort(-signal)[:count]].sum() / max(errors.sum(), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=32, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=0, batch_size=16, num_workers=0,
    )
    train_loader, val_loader = build_loaders(loader_args)
    train_labels = [np.asarray(train_loader.dataset.base_dataset[i]["label"])[0].astype(np.int8)
                    for i in range(len(train_loader.dataset.base_dataset))]
    paths = prediction_map(args.inference_dir)
    truth_labels, predicted_labels, names = [], [], []
    for i in range(len(val_loader.dataset.base_dataset)):
        case = val_loader.dataset.base_dataset[i]
        name = str(case["case_name"])
        truth_labels.append(np.asarray(case["label"])[0].astype(np.int8))
        predicted_labels.append(np.load(paths[name], allow_pickle=False).argmax(axis=0).astype(np.int8))
        names.append(name)
    boundary = np.asarray([np.count_nonzero((truth > 0) != (pred > 0))
                           for truth, pred in zip(truth_labels, predicted_labels)])
    swaps = np.asarray([np.count_nonzero((truth > 0) & (pred > 0) & (truth != pred))
                        for truth, pred in zip(truth_labels, predicted_labels)])
    truth_volume = np.asarray([np.count_nonzero(truth > 0) for truth in truth_labels])
    predicted_volume = np.asarray([np.count_nonzero(pred > 0) for pred in predicted_labels])
    boundary_rate = boundary / np.maximum(truth_volume, 1)
    report = {"schema": "semantic_constraints.cst_teacher.multiview_boundary_shape.v1",
              "fold": args.fold, "patients": len(names), "signals": {},
              "warning": "Unsupervised PCA fitted on training labels only; error correlations on previously studied validation folds are exploratory."}
    report["volume_baseline"] = {
        "boundary_r": correlation(predicted_volume, boundary),
        "boundary_rate_r": correlation(predicted_volume, boundary_rate),
        "boundary_top20_capture": capture(predicted_volume, boundary),
    }
    standardized = {"union": [], "classes": []}
    patient_signals: dict[str, np.ndarray] = {}
    for include_classes in (False, True):
        family = "classes" if include_classes else "union"
        for view in VIEW_AXIS:
            training = np.stack([area_profile(label, view, include_classes=include_classes) for label in train_labels])
            truth = np.stack([area_profile(label, view, include_classes=include_classes) for label in truth_labels])
            prediction = np.stack([area_profile(label, view, include_classes=include_classes) for label in predicted_labels])
            mean, vectors = fit_profile_pca(training)
            training_error = reconstruction_error(training, mean, vectors)
            truth_error = reconstruction_error(truth, mean, vectors)
            prediction_error = reconstruction_error(prediction, mean, vectors)
            name = f"{family}:{view}"
            report["signals"][name] = {
                "training_gt_mean_pca_error": float(training_error.mean()),
                "validation_gt_mean_pca_error": float(truth_error.mean()),
                "swin_mean_pca_error": float(prediction_error.mean()),
                "boundary_r": correlation(prediction_error, boundary),
                "boundary_rate_r": correlation(prediction_error, boundary_rate),
                "swaps_r": correlation(prediction_error, swaps),
                "boundary_top20_capture": capture(prediction_error, boundary),
                "swaps_top20_capture": capture(prediction_error, swaps),
            }
            patient_signals[name] = prediction_error
            scale = max(float(np.std(training_error)), 1e-6)
            standardized[family].append((prediction_error - np.mean(training_error)) / scale)
    for family, signals in standardized.items():
        for aggregation, signal in (("mean", np.mean(signals, axis=0)), ("max", np.max(signals, axis=0))):
            report["signals"][f"{family}:triview_{aggregation}"] = {
                "boundary_r": correlation(signal, boundary),
                "boundary_rate_r": correlation(signal, boundary_rate),
                "swaps_r": correlation(signal, swaps),
                "boundary_top20_capture": capture(signal, boundary),
                "swaps_top20_capture": capture(signal, swaps),
            }
            patient_signals[f"{family}:triview_{aggregation}"] = signal
    report["cases"] = [
        {"case_name": name, "boundary_error_voxels": int(boundary[index]),
         "swap_error_voxels": int(swaps[index]),
         "truth_foreground_voxels": int(truth_volume[index]),
         "swin_foreground_voxels": int(predicted_volume[index]),
         "signals": {key: float(value[index]) for key, value in patient_signals.items()}}
        for index, name in enumerate(names)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "signals": report["signals"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
