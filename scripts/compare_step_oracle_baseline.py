#!/usr/bin/env python3
"""Compare a frozen SwinUNETR baseline with label-only step-function oracles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_step_function_oracle import audit_case, centered_64, dice


def baseline_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    true_presence = truth.any(axis=2)
    pred_presence = prediction.any(axis=2)
    common = true_presence & pred_presence
    true_lower = truth.argmax(axis=2)
    true_upper = 63 - truth[:, :, ::-1].argmax(axis=2)
    pred_lower = prediction.argmax(axis=2)
    pred_upper = 63 - prediction[:, :, ::-1].argmax(axis=2)
    edges = np.concatenate((
        np.abs(true_lower[common] - pred_lower[common]),
        np.abs(true_upper[common] - pred_upper[common]),
    ))
    return {
        "baseline_union_dice": dice(truth, prediction),
        "baseline_xor_voxels": int(np.count_nonzero(truth ^ prediction)),
        "baseline_presence_precision": float(common.sum() / pred_presence.sum()),
        "baseline_presence_recall": float(common.sum() / true_presence.sum()),
        "baseline_false_positive_columns": int((pred_presence & ~true_presence).sum()),
        "baseline_false_negative_columns": int((true_presence & ~pred_presence).sum()),
        "baseline_edge_mae_common_columns": float(edges.mean()),
        "baseline_edge_within_one_common_columns": float(np.mean(edges <= 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=(4, 8, 12, 16, 24))
    args = parser.parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    split = json.loads((args.dataset / "splits_final.json").read_text())[0]
    expected = set(split["val"])
    df = pd.read_pickle(args.dataset / "msd_hippocampus_full.pkl", compression="gzip")
    records = df.set_index("subject_id")
    saved = {p.stem for p in args.masks.glob("hippocampus_*.npz")}
    if saved != expected:
        raise RuntimeError(f"Mask cases differ from fold-0 validation: missing={sorted(expected-saved)}, extra={sorted(saved-expected)}")
    rows = []
    for name in sorted(expected):
        label = np.asarray(records.loc[name, "label_data"])
        truth = centered_64(label)
        with np.load(args.masks / f"{name}.npz") as data:
            prediction = data["prediction"] > 0
        if prediction.shape != truth.shape:
            raise RuntimeError(f"Shape mismatch for {name}: {prediction.shape}")
        common = baseline_metrics(truth, prediction)
        for oracle in audit_case(label, budgets):
            rows.append({"case": name, **common, **{f"oracle_{key}": value for key, value in oracle.items()}})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "case_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = [row for row in rows if row["oracle_k"] == budgets[0]]
    report = {
        "cohort": "MSD fold-0 outer validation, previously studied; exploratory",
        "cases": len(cases),
        "baseline": {
            key: float(np.mean([row[key] for row in cases]))
            for key in common if key not in {"baseline_false_positive_columns", "baseline_false_negative_columns", "baseline_xor_voxels"}
        },
        "baseline_totals": {
            key: int(sum(row[key] for row in cases))
            for key in ("baseline_false_positive_columns", "baseline_false_negative_columns", "baseline_xor_voxels")
        },
        "oracle": {},
    }
    for budget in budgets:
        subset = [row for row in rows if row["oracle_k"] == budget]
        report["oracle"][str(budget)] = {
            "mean_union_dice": float(np.mean([row["oracle_dice"] for row in subset])),
            "min_union_dice": float(np.min([row["oracle_dice"] for row in subset])),
            "mean_edge_mae_voxels": float(np.mean([row["oracle_edge_mae_voxels"] for row in subset])),
            "mean_run_endpoint_edge_mae_voxels": float(np.mean([row["oracle_run_endpoint_edge_mae_voxels"] for row in subset])),
            "mean_plateaus_per_run": float(sum(row["oracle_plateaus"] for row in subset) / sum(row["oracle_runs"] for row in subset)),
            "total_xor_voxels": int(sum(row["oracle_xor_voxels"] for row in subset)),
            "mean_full_interval_dice": float(np.mean([row["oracle_full_interval_dice"] for row in subset])),
        }
    (args.output / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
