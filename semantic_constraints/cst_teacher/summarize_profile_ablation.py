#!/usr/bin/env python3
"""Aggregate profile-focused CST ablations across variants and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser.parse_args()


def finite_mean(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return mean(filtered) if filtered else None


def main() -> None:
    args = parse_args()
    variants: list[dict[str, Any]] = []
    for variant in args.variants:
        runs = []
        for seed in args.seeds:
            root = args.run_root / f"{variant}_seed_{seed}"
            history = json.loads((root / "history.json").read_text(encoding="utf-8"))
            evaluation = json.loads((root / "prediction_evaluation.json").read_text(encoding="utf-8"))
            best = min(history, key=lambda record: record["validation"]["loss"])
            runs.append(
                {
                    "seed": seed,
                    "best_epoch": best["epoch"],
                    "validation_profile_mae": best["validation"]["profile_absolute_error"],
                    "prediction_profile_violation": evaluation["profile_summary"]["mean_profile_violation"],
                    "profile_vs_dice_error": evaluation["profile_summary"]["violation_vs_dice_error_pearson"],
                    "teacher_profile_mae": evaluation["profile_summary"]["teacher_profile_mae"],
                }
            )
        correlations = [run["profile_vs_dice_error"] for run in runs]
        teacher_errors = [run["teacher_profile_mae"] for run in runs]
        stable_positive = all(value is not None and value >= 0.20 for value in correlations)
        useful_accuracy = mean(teacher_errors) <= 0.10
        variants.append(
            {
                "variant": variant,
                "runs": runs,
                "mean_teacher_profile_mae": mean(teacher_errors),
                "std_teacher_profile_mae": pstdev(teacher_errors),
                "mean_profile_vs_dice_error": finite_mean(correlations),
                "stable_positive_error_signal": stable_positive,
                "useful_profile_accuracy": useful_accuracy,
                "candidate": stable_positive and useful_accuracy,
            }
        )

    report = {
        "schema": "semantic_constraints.cst_teacher.profile_ablation.v1",
        "seeds": args.seeds,
        "variants": variants,
        "warning": "Discovery-fold ablation only; any candidate still requires counterfactual repair.",
    }
    json_path = args.run_root / "profile_ablation_summary.json"
    md_path = args.run_root / "profile_ablation_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CST profile-head ablation",
        "",
        "| Variant | Teacher profile MAE | Profile/error r | Stable candidate |",
        "|---|---:|---:|---:|",
    ]
    for item in variants:
        correlation = item["mean_profile_vs_dice_error"]
        correlation_text = "nan" if correlation is None else f"{correlation:.3f}"
        lines.append(
            f"| {item['variant']} | {item['mean_teacher_profile_mae']:.4f} ± "
            f"{item['std_teacher_profile_mae']:.4f} | {correlation_text} | {item['candidate']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
