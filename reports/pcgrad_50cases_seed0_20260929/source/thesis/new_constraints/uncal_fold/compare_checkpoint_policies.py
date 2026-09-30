#!/usr/bin/env python3
"""Compare the historical fixed-50 and canonical early-stopped cut audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OLD = REPOSITORY_ROOT / "docs" / "experiments" / "uncal_foldedness_predicted_foreground_20260921"
NEW = REPOSITORY_ROOT / "docs" / "experiments" / "uncal_foldedness_predicted_foreground_early_stopping_20260921"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed50-dir", type=Path, default=OLD)
    parser.add_argument("--early-stopping-dir", type=Path, default=NEW)
    return parser.parse_args()


def paired(errors_first: np.ndarray, errors_second: np.ndarray) -> dict[str, float | int]:
    difference = errors_second - errors_first
    return {
        "second_better": int(np.count_nonzero(difference < 0)),
        "equal": int(np.count_nonzero(difference == 0)),
        "second_worse": int(np.count_nonzero(difference > 0)),
        "mean_change": float(difference.mean()),
        "wilcoxon_two_sided_p_value_exploratory": float(
            wilcoxon(errors_second, errors_first).pvalue
        ),
    }


def main() -> None:
    args = parse_args()
    old_summary = json.loads((args.fixed50_dir / "summary.json").read_text())
    new_summary = json.loads((args.early_stopping_dir / "summary.json").read_text())
    old_cases = pd.read_csv(args.fixed50_dir / "case_results.csv")
    new_cases = pd.read_csv(args.early_stopping_dir / "case_results.csv")
    cases = old_cases.merge(
        new_cases,
        on=["case_name", "target_cut"],
        suffixes=("_fixed50", "_early_stopping"),
        validate="one_to_one",
    )

    policy_comparison = {}
    for arm in ("unaugmented", "augmented"):
        fixed = cases[f"{arm}_error_fixed50"].to_numpy()
        early = cases[f"{arm}_error_early_stopping"].to_numpy()
        comparison = paired(fixed, early)
        comparison["cut_agreement_fraction"] = float(
            np.mean(
                cases[f"{arm}_cut_fixed50"]
                == cases[f"{arm}_cut_early_stopping"]
            )
        )
        policy_comparison[arm] = comparison

    outlier_index = int(cases["augmented_error_early_stopping"].idxmax())
    outlier = cases.loc[outlier_index]
    without_outlier = cases.index != outlier_index
    robust_augmented = {
        "excluded_case": str(outlier["case_name"]),
        "excluded_error": int(outlier["augmented_error_early_stopping"]),
        "early_stopping_unaugmented_mae_without_case": float(
            cases.loc[without_outlier, "unaugmented_error_early_stopping"].mean()
        ),
        "fixed50_mae_without_case": float(
            cases.loc[without_outlier, "augmented_error_fixed50"].mean()
        ),
        "early_stopping_mae_without_case": float(
            cases.loc[without_outlier, "augmented_error_early_stopping"].mean()
        ),
    }
    summary = {
        "schema": "uncal_fold.checkpoint_policy_comparison.v1",
        "locator_coefficients_identical": (
            old_summary["frozen_locator_coefficients"]
            == new_summary["frozen_locator_coefficients"]
        ),
        "fixed50_cut_metrics": old_summary["cut_metrics"],
        "early_stopping_cut_metrics": new_summary["cut_metrics"],
        "fixed50_prediction_support": old_summary["prediction_support"],
        "early_stopping_prediction_support": new_summary["prediction_support"],
        "paired_early_stopping_vs_fixed50": policy_comparison,
        "early_stopping_augmented_vs_unaugmented": new_summary[
            "paired_augmented_vs_unaugmented"
        ],
        "largest_early_stopping_augmented_outlier": {
            "case_name": str(outlier["case_name"]),
            "target_cut": int(outlier["target_cut"]),
            "predicted_cut": int(outlier["augmented_cut_early_stopping"]),
            "absolute_error": int(outlier["augmented_error_early_stopping"]),
            "foreground_dice": float(outlier["augmented_foreground_dice_early_stopping"]),
        },
        "augmented_sensitivity_without_largest_outlier": robust_augmented,
    }
    (args.early_stopping_dir / "checkpoint_policy_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    old_metrics = old_summary["cut_metrics"]
    new_metrics = new_summary["cut_metrics"]
    lines = [
        "# Uncal-cut result under matched early stopping",
        "",
        "The descriptor and its fitted coefficients are identical between audits. Only",
        "the segmentation checkpoints and their predicted foreground masks changed.",
        "",
        "| model foreground | policy | MAE | exact | within 1 | within 2 | p90 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("unaugmented_prediction", "Unaugmented"),
        ("augmented_prediction", "Augmented"),
    ):
        for policy, metrics in (
            ("Fixed 50", old_metrics[arm]),
            ("Early-stopped best", new_metrics[arm]),
        ):
            lines.append(
                f"| {label} | {policy} | {metrics['mae_slices']:.3f} | "
                f"{metrics['exact_fraction']:.1%} | {metrics['within_1_fraction']:.1%} | "
                f"{metrics['within_2_fraction']:.1%} | {metrics['p90_absolute_error']:.1f} | "
                f"{metrics['maximum_absolute_error']} |"
            )
    paired_aug = new_summary["paired_augmented_vs_unaugmented"]
    lines.extend(
        [
            "",
            "## Matched early-stopping comparison",
            "",
            f"Augmentation is better/equal/worse in "
            f"`{paired_aug['second_better_cases']}/{paired_aug['equal_cases']}/"
            f"{paired_aug['second_worse_cases']}` cases (paired exploratory "
            f"p=`{paired_aug['wilcoxon_two_sided_p_value_exploratory']:.4g}`).",
            "It improves exact and within-one localization, but one catastrophic endpoint",
            f"selection in `{outlier['case_name']}` chooses slice "
            f"`{int(outlier['augmented_cut_early_stopping'])}` instead of "
            f"`{int(outlier['target_cut'])}`. That single `{int(outlier['augmented_error_early_stopping'])}`-slice",
            "error inflates the augmented MAE. It cannot be removed or tuned away after",
            "looking at validation labels; it is evidence that the soft position prior is",
            "not a sufficient safety guard.",
            "",
            "Without that case (reported only as a sensitivity analysis), augmented",
            f"early-stopping MAE is `{robust_augmented['early_stopping_mae_without_case']:.3f}`",
            f"versus `{robust_augmented['early_stopping_unaugmented_mae_without_case']:.3f}`",
            "for the matched non-augmented checkpoint and",
            f"`{robust_augmented['fixed50_mae_without_case']:.3f}` for the historical",
            "augmented fixed-50 checkpoint. The primary all-52 result remains the table above.",
            "",
            "## Scientific conclusion",
            "",
            "The anatomical evidence discovered earlier does not change: area transition,",
            "superior-boundary rise/notch change, asymmetric new superior tissue, and the",
            "superior T1 band retain exactly the same fitted weights. Early stopping and",
            "augmentation improve the quality of the foreground supplied to those descriptors,",
            "but the locator remains vulnerable to a high-scoring endpoint. It should remain",
            "a soft, quality-gated LTN predicate rather than a hard cut rule.",
        ]
    )
    (args.early_stopping_dir / "CHECKPOINT_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
