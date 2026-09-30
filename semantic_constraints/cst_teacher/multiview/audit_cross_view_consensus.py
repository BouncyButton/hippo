#!/usr/bin/env python3
"""Test label-free 3D interface consensus from intersecting sagittal/axial traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluate_predictions import prediction_map
from ..train_teacher import build_loaders
from .cut_model import best_cut_from_labels
from .train_cut_study import foreground_dice, relabel_by_cut


def trace_cut_candidates(prediction: np.ndarray) -> np.ndarray:
    """One interface midpoint per Y-directed column containing both labels."""
    if prediction.ndim != 3:
        raise ValueError("prediction must have shape (X,Y,Z)")
    anterior = prediction == 1
    posterior = prediction == 2
    any_both = anterior.any(axis=1) & posterior.any(axis=1)
    first_anterior = np.argmax(anterior, axis=1)
    last_posterior = prediction.shape[1] - 1 - np.argmax(posterior[:, ::-1, :], axis=1)
    midpoints = np.floor((first_anterior + last_posterior - 1) / 2).astype(np.int16)
    return midpoints[any_both]


def soft_global_cut(probabilities: np.ndarray, confidence: float = 0.0) -> int:
    """Choose the plane minimizing class probability disagreement within foreground."""
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError("probabilities must have shape (3,X,Y,Z)")
    foreground = probabilities[1] + probabilities[2]
    weight = (foreground >= confidence).astype(np.float32)
    anterior = (probabilities[1] * weight).sum(axis=(0, 2))
    posterior = (probabilities[2] * weight).sum(axis=(0, 2))
    costs = np.cumsum(anterior) + posterior.sum() - np.cumsum(posterior)
    return int(np.argmin(costs[:-1]))


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
    _, val_loader = build_loaders(loader_args)
    paths = prediction_map(args.inference_dir)
    rows = []
    for index in range(len(val_loader.dataset.base_dataset)):
        item = val_loader.dataset.base_dataset[index]
        name = str(item["case_name"])
        label = np.asarray(item["label"])[0].astype(np.int8)
        probabilities = np.load(paths[name], allow_pickle=False)
        prediction = probabilities.argmax(axis=0).astype(np.int8)
        true_cut = best_cut_from_labels(label)[0]
        swin_cut = best_cut_from_labels(prediction)[0]
        traces = trace_cut_candidates(prediction)
        cuts = {
            "swin_hard_global": swin_cut,
            "soft_global": soft_global_cut(probabilities),
            "soft_confident_08": soft_global_cut(probabilities, 0.8),
        }
        if len(traces):
            cuts["trace_median"] = int(np.rint(np.median(traces)))
            cuts["trace_trimmed_mean"] = int(np.rint(np.mean(np.sort(traces)[len(traces) // 10:len(traces) - len(traces) // 10])))
        baseline = foreground_dice(label, prediction)
        rows.append({
            "case_name": name, "true_cut_y": true_cut, "cuts": cuts,
            "trace_count": int(len(traces)),
            "trace_std": float(np.std(traces)) if len(traces) else None,
            "absolute_errors": {key: abs(value - true_cut) for key, value in cuts.items()},
            "relabel_dice_deltas": {key: foreground_dice(label, relabel_by_cut(prediction, value)) - baseline
                                    for key, value in cuts.items()},
        })
    metrics = {}
    for method in rows[0]["cuts"]:
        errors = np.asarray([row["absolute_errors"].get(method, np.nan) for row in rows])
        deltas = np.asarray([row["relabel_dice_deltas"].get(method, np.nan) for row in rows])
        metrics[method] = {
            "mae_voxels": float(np.nanmean(errors)),
            "within_one_voxel": float(np.nanmean(errors <= 1)),
            "mean_relabel_dice_delta": float(np.nanmean(deltas)),
            "patients_harmed": int(np.sum(deltas < -1e-8)),
            "patients_improved": int(np.sum(deltas > 1e-8)),
        }
    report = {
        "schema": "semantic_constraints.cst_teacher.multiview_cross_view_consensus.v1",
        "fold": args.fold, "patients": len(rows), "methods": metrics, "cases": rows,
        "warning": "Predeclared unsupervised 3D consensus rules evaluated on previously studied Swin validation folds; not a new prospective cohort.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
