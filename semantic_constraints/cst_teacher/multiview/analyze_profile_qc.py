#!/usr/bin/env python3
"""Audit whether view-specific CST profile disagreements flag Swin errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluate_predictions import prediction_map
from ..train_teacher import build_loaders
from .cut_model import VIEW_SPECS


def correlation(signal: np.ndarray, target: np.ndarray) -> float:
    if np.std(signal) < 1e-10 or np.std(target) < 1e-10:
        return float("nan")
    return float(np.corrcoef(signal, target)[0, 1])


def top_fraction_capture(signal: np.ndarray, errors: np.ndarray, fraction: float = 0.2) -> float:
    count = max(1, int(np.ceil(len(signal) * fraction)))
    return float(errors[np.argsort(-signal)[:count]].sum() / max(errors.sum(), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=32, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=0, batch_size=16, num_workers=0,
    )
    _, val_loader = build_loaders(loader_args)
    cases = [val_loader.dataset.base_dataset[index] for index in range(len(val_loader.dataset.base_dataset))]
    paths = prediction_map(args.inference_dir)
    names, boundary, swaps, total, entropy = [], [], [], [], []
    for case in cases:
        name = str(case["case_name"])
        truth = np.asarray(case["label"])[0].astype(np.int8)
        probs = np.load(paths[name], allow_pickle=False)
        pred = probs.argmax(axis=0)
        union_error = int(np.count_nonzero((pred > 0) != (truth > 0)))
        swap_error = int(np.count_nonzero((pred > 0) & (truth > 0) & (pred != truth)))
        names.append(name)
        boundary.append(union_error)
        swaps.append(swap_error)
        total.append(union_error + swap_error)
        foreground = probs[1:].sum(axis=0)
        region = (foreground > 0.05) & (foreground < 0.95)
        entropy.append(float((-(probs * np.log(probs.clip(1e-8))).sum(axis=0))[region].mean()) if region.any() else 0.0)
    boundary, swaps, total, entropy = map(np.asarray, (boundary, swaps, total, entropy))
    report = {
        "schema": "semantic_constraints.cst_teacher.multiview_profile_qc.v1",
        "fold": args.fold, "patients": len(names),
        "error_masses": {"boundary": int(boundary.sum()), "swaps": int(swaps.sum()), "total": int(total.sum())},
        "signals": {},
        "warning": "Exploratory correlations on previously studied validation folds; no threshold is trained or claimed as prospective validation.",
    }
    signals = {"swin_entropy": entropy}
    for specification_name, specification in VIEW_SPECS.items():
        path = args.profile_dir / f"{specification_name}_val_features.npz"
        features = np.load(path, allow_pickle=False)
        if features["case_names"].tolist() != names:
            raise ValueError(f"patient order mismatch in {path}")
        teacher = features["teacher_profiles"]
        swin = features["swin_profiles"]
        target = features["profile_targets"]
        offset = 0
        for view, count in specification.items():
            part = slice(offset, offset + count)
            signal_name = f"{specification_name}:{view}:disagreement"
            signals[signal_name] = np.mean(np.abs(teacher[:, part] - swin[:, part]), axis=(1, 2))
            report["signals"][signal_name] = {
                "teacher_target_profile_mae": float(np.mean(np.abs(teacher[:, part] - target[:, part]))),
                "optimistic_heldout_mean_profile_mae": float(np.mean(np.abs(
                    target[:, part] - target[:, part].mean(axis=0, keepdims=True)))),
                "swin_target_profile_mae": float(np.mean(np.abs(swin[:, part] - target[:, part]))),
                "oracle_profile_residual_vs_boundary_r": correlation(
                    np.mean(np.abs(swin[:, part] - target[:, part]), axis=(1, 2)), boundary),
            }
            offset += count
        signals[f"{specification_name}:all:disagreement"] = np.mean(np.abs(teacher - swin), axis=(1, 2))
    for name, signal in signals.items():
        item = report["signals"].setdefault(name, {})
        item.update({
            "boundary_r": correlation(signal, boundary),
            "swaps_r": correlation(signal, swaps),
            "total_r": correlation(signal, total),
            "boundary_top20_capture": top_fraction_capture(signal, boundary),
            "swaps_top20_capture": top_fraction_capture(signal, swaps),
            "total_top20_capture": top_fraction_capture(signal, total),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
