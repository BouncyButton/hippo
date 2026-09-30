#!/usr/bin/env python3
"""Bound A/P gains from a perfect global cut while preserving Swin foreground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluate_predictions import prediction_map
from ..train_teacher import build_loaders
from .cut_model import best_cut_from_labels
from .train_cut_study import foreground_dice, relabel_by_cut


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
    _, loader = build_loaders(loader_args)
    paths = prediction_map(args.inference_dir)
    cases = []
    for index in range(len(loader.dataset.base_dataset)):
        item = loader.dataset.base_dataset[index]
        name = str(item["case_name"])
        truth = np.asarray(item["label"])[0].astype(np.int8)
        probabilities = np.load(paths[name], allow_pickle=False)
        prediction = probabilities.argmax(axis=0).astype(np.int8)
        gt_cut, high, nonplanar = best_cut_from_labels(truth)
        if not high:
            raise ValueError("unexpected reversed MSD A/P orientation")
        try:
            swin_cut = best_cut_from_labels(prediction)[0]
        except ValueError:
            swin_cut = None
        oracle = relabel_by_cut(prediction, gt_cut)
        baseline_swap = (truth > 0) & (prediction > 0) & (truth != prediction)
        oracle_swap = (truth > 0) & (oracle > 0) & (truth != oracle)
        before = foreground_dice(truth, prediction)
        after = foreground_dice(truth, oracle)
        cases.append({
            "case_name": name,
            "true_cut_y": gt_cut,
            "swin_cut_y": swin_cut,
            "swin_cut_absolute_error": abs(swin_cut - gt_cut) if swin_cut is not None else None,
            "nonplanar_ground_truth_voxels": nonplanar,
            "baseline_swap_voxels": int(baseline_swap.sum()),
            "oracle_swap_voxels": int(oracle_swap.sum()),
            "swap_voxels_fixed": int(np.count_nonzero(baseline_swap & ~oracle_swap)),
            "swap_voxels_introduced": int(np.count_nonzero(~baseline_swap & oracle_swap)),
            "baseline_dice": before,
            "oracle_cut_dice": after,
            "oracle_dice_delta": after - before,
        })
    delta = np.asarray([case["oracle_dice_delta"] for case in cases])
    rng = np.random.default_rng(20260922 + args.fold)
    bootstrap = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(1000)])
    report = {
        "schema": "semantic_constraints.cst_teacher.multiview.oracle_cut_audit.v1",
        "fold": args.fold,
        "summary": {
            "patients": len(cases),
            "swin_cut_mae_voxels": float(np.mean([case["swin_cut_absolute_error"] for case in cases if case["swin_cut_absolute_error"] is not None])),
            "baseline_swap_voxels": int(sum(case["baseline_swap_voxels"] for case in cases)),
            "oracle_swap_voxels": int(sum(case["oracle_swap_voxels"] for case in cases)),
            "swap_voxels_fixed": int(sum(case["swap_voxels_fixed"] for case in cases)),
            "swap_voxels_introduced": int(sum(case["swap_voxels_introduced"] for case in cases)),
            "oracle_mean_dice_gain": float(delta.mean()),
            "oracle_with_label_selected_abstention_gain": float(np.maximum(delta, 0).mean()),
            "oracle_gain_lower95": float(np.quantile(bootstrap, 0.025)),
            "oracle_gain_upper95": float(np.quantile(bootstrap, 0.975)),
            "patients_harmed": int(np.sum(delta < -1e-8)),
            "patients_improved": int(np.sum(delta > 1e-8)),
        },
        "cases": cases,
        "warning": "The global cut is computed from the validation label. The optional-abstention value also uses validation Dice to select patients; both are oracle diagnostics, not deployable correction gains.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
