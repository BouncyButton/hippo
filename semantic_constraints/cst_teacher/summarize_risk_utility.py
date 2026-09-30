#!/usr/bin/env python3
"""Summarize selective-QC utility from cross-fitted CST risk predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-sets", nargs="+", default=("uncertainty", "combined"))
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    parser.add_argument("--within-case-slices", type=int, default=4)
    return parser.parse_args()


def selective_metrics(target: np.ndarray, prediction: np.ndarray, fractions: list[float]) -> list[dict[str, Any]]:
    output = []
    for fraction in fractions:
        count = max(1, int(np.ceil(len(target) * fraction)))
        order = np.argsort(prediction)
        flagged, retained = order[-count:], order[:-count]
        output.append(
            {
                "flagged_fraction": fraction,
                "flagged_count": count,
                "error_mass_capture": float(target[flagged].sum() / target.sum()),
                "flagged_mean_error": float(target[flagged].mean()),
                "retained_mean_error": float(target[retained].mean()),
            }
        )
    return output


def within_case_capture(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    selected_slices: int,
) -> dict[str, float]:
    values = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        count = min(selected_slices, len(indices))
        selected = indices[np.argsort(prediction[indices])[-count:]]
        total = float(target[indices].sum())
        values.append(0.0 if total == 0 else float(target[selected].sum() / total))
    return {
        "selected_slices_per_case": selected_slices,
        "mean_error_mass_capture": float(np.mean(values)),
        "median_error_mass_capture": float(np.median(values)),
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.risk_report.read_text(encoding="utf-8"))
    arrays = np.load(args.features, allow_pickle=True)
    slice_target = arrays["slice_target"]
    slice_groups = arrays["slice_groups"]
    case_target = arrays["case_target"]
    results = {}
    for feature in args.feature_sets:
        slice_prediction = np.median(
            np.asarray([seed["slice"][feature]["predictions"] for seed in report["seed_reports"]]),
            axis=0,
        )
        case_prediction = np.median(
            np.asarray([seed["case"][feature]["predictions"] for seed in report["seed_reports"]]),
            axis=0,
        )
        results[feature] = {
            "slice_pearson": float(np.corrcoef(slice_target, slice_prediction)[0, 1]),
            "case_pearson": float(np.corrcoef(case_target, case_prediction)[0, 1]),
            "global_slice_selection": selective_metrics(slice_target, slice_prediction, args.fractions),
            "within_case_selection": within_case_capture(
                slice_target,
                slice_prediction,
                slice_groups,
                args.within_case_slices,
            ),
        }
    output = {
        "schema": "semantic_constraints.cst_teacher.risk_utility.v1",
        "source_report": str(args.risk_report),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [
        "# CST selective-QC utility",
        "",
        "| Features | Slice r | Top 5% capture | Top 10% capture | Top 20% capture | Within-patient top-4 capture |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, item in results.items():
        captures = [entry["error_mass_capture"] for entry in item["global_slice_selection"]]
        lines.append(
            f"| {feature} | {item['slice_pearson']:.3f} | {captures[0]:.1%} | {captures[1]:.1%} | "
            f"{captures[2]:.1%} | {item['within_case_selection']['mean_error_mass_capture']:.1%} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
