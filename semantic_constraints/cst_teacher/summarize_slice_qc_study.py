#!/usr/bin/env python3
"""Aggregate the predeclared three-seed learnable slice-QC comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from .run_slice_qc_experiment import bootstrap_capture_difference, metrics as score_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [json.loads((args.run_root / f"seed_{seed}" / "report.json").read_text()) for seed in args.seeds]
    metrics = {}
    ensembles = {}
    for fold in (0, 1):
        metrics[str(fold)] = {}
        reference = reports[0]["folds"][fold]
        target = np.asarray(reference["target"], dtype=np.float32)
        names = np.asarray(reference["case_names"])
        ensembles[str(fold)] = {}
        model_names = reports[0]["folds"][fold]["metrics"]
        for model in model_names:
            metrics[str(fold)][model] = {}
            for key in ("slice_pearson", "slice_mae", "top20_error_capture", "within_patient_top4_capture"):
                values = [report["folds"][fold]["metrics"][model][key] for report in reports]
                metrics[str(fold)][model][key] = {
                    "mean": mean(values),
                    "std": pstdev(values),
                    "per_seed": values,
                }
            for report in reports[1:]:
                other = report["folds"][fold]
                if other["case_names"] != reference["case_names"] or not np.allclose(other["target"], target):
                    raise ValueError("teacher seeds have inconsistent patient order or targets")
            average = np.mean(
                [np.asarray(report["folds"][fold]["predictions"][model]) for report in reports], axis=0
            )
            ensembles[str(fold)][model] = score_metrics(target, average, names)
        uncertainty = np.mean(
            [np.asarray(report["folds"][fold]["predictions"]["ridge_uncertainty"]) for report in reports],
            axis=0,
        )
        for model in ("ridge_portable", "ridge_combined", "temporal_portable", "temporal_combined"):
            average = np.mean(
                [np.asarray(report["folds"][fold]["predictions"][model]) for report in reports], axis=0
            )
            ensembles[str(fold)][model]["capture_gain_vs_uncertainty"] = bootstrap_capture_difference(
                target, uncertainty, average, seed=20260921 + fold
            )
    report = {
        "schema": "semantic_constraints.cst_teacher.slice_qc_study.v1",
        "seeds": args.seeds,
        "metrics": metrics,
        "matched_teacher_ensembles": ensembles,
        "decision_rule": (
            "Prefer the lowest-complexity model whose top-20% error capture improves over "
            "ridge_uncertainty on both folds across all three CST seeds; do not tune on fold 1."
        ),
        "warning": "These two folds were previously analysed; no prospective holdout claim.",
    }
    (args.run_root / "study_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Three-seed learnable CST slice-QC study",
        "",
        "Patient-grouped 5-fold CV within each 52-patient Swin validation split.",
        "",
        "| Fold | Model | Slice r | Top-20% capture | Within-patient top-4 capture |",
        "|---:|---|---:|---:|---:|",
    ]
    for fold in (0, 1):
        for model, item in metrics[str(fold)].items():
            correlation = item["slice_pearson"]
            capture = item["top20_error_capture"]
            within = item["within_patient_top4_capture"]
            lines.append(
                f"| {fold} | {model} | {correlation['mean']:.3f} ± {correlation['std']:.3f} | "
                f"{capture['mean']:.1%} ± {capture['std']:.1%} | "
                f"{within['mean']:.1%} ± {within['std']:.1%} |"
            )
    lines.extend(("", report["decision_rule"], ""))
    lines.extend(
        (
            "## Matched three-teacher ensemble",
            "",
            "| Fold | Model | Slice r | Top-20% capture | Within-patient top-4 capture |",
            "|---:|---|---:|---:|---:|",
        )
    )
    for fold in (0, 1):
        for model, item in ensembles[str(fold)].items():
            lines.append(
                f"| {fold} | {model} | {item['slice_pearson']:.3f} | "
                f"{item['top20_error_capture']:.1%} | {item['within_patient_top4_capture']:.1%} |"
            )
    lines.append("")
    (args.run_root / "study_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.run_root / "study_summary.md").read_text())


if __name__ == "__main__":
    main()
