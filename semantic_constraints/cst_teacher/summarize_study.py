#!/usr/bin/env python3
"""Aggregate a multi-seed CST teacher study and surface reproducible clues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from .analyze_embeddings import _adjusted_rand_index
from .descriptors import DESCRIPTOR_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "standard_deviation": float(pstdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def optional_stats(values: list[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {"available_seeds": len(finite), **(stats(finite) if finite else {})}


def load_seed(run_root: Path, seed: int) -> dict[str, Any]:
    directory = run_root / f"seed_{seed}"
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    cluster = json.loads((directory / "embedding_analysis/cluster_report.json").read_text(encoding="utf-8"))
    evaluation = json.loads((directory / "prediction_evaluation.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "embedding_analysis/embeddings_and_clusters.npz")
    best = min(history, key=lambda record: record["validation"]["loss"])
    return {
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best["epoch"],
        "best_validation": best["validation"],
        "cluster": cluster,
        "evaluation": evaluation,
        "train_case_names": arrays["train_case_names"],
        "validation_case_names": arrays["validation_case_names"],
        "train_assignments": arrays["train_assignments"],
        "validation_assignments": arrays["validation_assignments"],
    }


def aligned_ari(first: dict[str, Any], second: dict[str, Any], split: str) -> float:
    first_names = first[f"{split}_case_names"].astype(str)
    second_names = second[f"{split}_case_names"].astype(str)
    first_map = dict(zip(first_names, first[f"{split}_assignments"], strict=True))
    second_map = dict(zip(second_names, second[f"{split}_assignments"], strict=True))
    names = sorted(first_map.keys() & second_map.keys())
    return _adjusted_rand_index(
        np.asarray([first_map[name] for name in names]),
        np.asarray([second_map[name] for name in names]),
    )


def aggregate(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    descriptor_summary = {}
    clues = []
    for descriptor in DESCRIPTOR_NAMES:
        entries = [seed["evaluation"]["descriptor_summary"][descriptor] for seed in seeds]
        correlations = [entry["violation_vs_dice_error_pearson"] for entry in entries]
        available = [value for value in correlations if value is not None]
        coverage = [entry["teacher_interval_coverage"] for entry in entries]
        violation_rate = [entry["prediction_violation_rate"] for entry in entries]
        stable_positive = len(available) == len(seeds) and min(available) >= 0.20
        calibrated = 0.65 <= mean(coverage) <= 0.95
        nonvacuous = 0.05 <= mean(violation_rate) <= 0.80
        descriptor_summary[descriptor] = {
            "teacher_median_mae": stats([entry["teacher_median_mae"] for entry in entries]),
            "teacher_interval_coverage": stats(coverage),
            "prediction_violation_rate": stats(violation_rate),
            "violation_vs_dice_error_pearson": optional_stats(correlations),
            "violation_vs_descriptor_error_pearson": optional_stats(
                [entry["violation_vs_descriptor_error_pearson"] for entry in entries]
            ),
            "screen": {
                "calibrated_roughly_to_nominal_80_percent": calibrated,
                "nonvacuous_prediction_violations": nonvacuous,
                "stable_positive_error_signal": stable_positive,
            },
        }
        if calibrated and nonvacuous and stable_positive:
            clues.append(
                {
                    "candidate": f"conditional_{descriptor}",
                    "reason": "calibrated, non-vacuous, and positively aligned with Dice error in every seed",
                }
            )

    profile_correlations = [
        seed["evaluation"]["profile_summary"]["violation_vs_dice_error_pearson"] for seed in seeds
    ]
    anomaly_correlations = [
        seed["evaluation"]["anomaly_summary"]["score_vs_dice_error_pearson"] for seed in seeds
    ]
    if all(value is not None and value >= 0.20 for value in profile_correlations):
        clues.append({"candidate": "coronal_profile", "reason": "positive Dice-error signal in every seed"})
    if all(value is not None and value >= 0.20 for value in anomaly_correlations):
        clues.append({"candidate": "mask_anomaly", "reason": "positive Dice-error signal in every seed"})

    pairwise = []
    for first_index, first in enumerate(seeds):
        for second in seeds[first_index + 1 :]:
            pairwise.append(
                {
                    "seeds": [first["seed"], second["seed"]],
                    "train_ari": aligned_ari(first, second, "train"),
                    "validation_ari": aligned_ari(first, second, "validation"),
                }
            )
    selected_counts = [seed["cluster"]["selection"]["selected_clusters"] for seed in seeds]
    bootstrap_ari = [seed["cluster"]["bootstrap_stability"]["mean_adjusted_rand_index"] for seed in seeds]
    stable_clusters = (
        len(set(selected_counts)) == 1
        and min(bootstrap_ari) >= 0.60
        and min(item["validation_ari"] for item in pairwise) >= 0.50
    )
    if stable_clusters:
        clues.append(
            {
                "candidate": "embedding_morphology_clusters",
                "reason": "same cluster count with stable bootstrap and cross-seed validation assignments",
            }
        )

    return {
        "schema": "semantic_constraints.cst_teacher.study_summary.v1",
        "seeds": [seed["seed"] for seed in seeds],
        "per_seed": [
            {
                "seed": seed["seed"],
                "epochs_completed": seed["epochs_completed"],
                "best_epoch": seed["best_epoch"],
                "best_validation": seed["best_validation"],
            }
            for seed in seeds
        ],
        "descriptors": descriptor_summary,
        "profile": {
            "teacher_profile_mae": stats(
                [seed["evaluation"]["profile_summary"]["teacher_profile_mae"] for seed in seeds]
            ),
            "violation_vs_dice_error_pearson": optional_stats(profile_correlations),
        },
        "anomaly": {
            "validation_detection_accuracy": stats(
                [seed["best_validation"]["anomaly_accuracy"] for seed in seeds]
            ),
            "score_vs_dice_error_pearson": optional_stats(anomaly_correlations),
        },
        "clusters": {
            "selected_counts": selected_counts,
            "bootstrap_mean_ari": bootstrap_ari,
            "pairwise_assignment_ari": pairwise,
            "passes_stability_screen": stable_clusters,
        },
        "candidate_clues": clues,
        "warning": "Screens are discovery heuristics. A candidate still requires counterfactual repair and matched training.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# CST teacher multi-seed discovery summary", "", "## Candidate clues", ""]
    if report["candidate_clues"]:
        for clue in report["candidate_clues"]:
            lines.append(f"- `{clue['candidate']}`: {clue['reason']}.")
    else:
        lines.append("No candidate passed all automatic discovery screens.")
    lines.extend(["", "## Descriptor screens", ""])
    for name, data in report["descriptors"].items():
        screen = data["screen"]
        lines.append(
            f"- `{name}`: coverage={data['teacher_interval_coverage']['mean']:.3f}, "
            f"violation rate={data['prediction_violation_rate']['mean']:.3f}, "
            f"error correlation={data['violation_vs_dice_error_pearson'].get('mean', float('nan')):.3f}, "
            f"screen={all(screen.values())}."
        )
    lines.extend(
        [
            "",
            "## Other signals",
            "",
            f"- Profile/error correlation: {report['profile']['violation_vs_dice_error_pearson'].get('mean', float('nan')):.3f}.",
            f"- Anomaly/error correlation: {report['anomaly']['score_vs_dice_error_pearson'].get('mean', float('nan')):.3f}.",
            f"- Stable morphology clusters: {report['clusters']['passes_stability_screen']}.",
            "",
            report["warning"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seeds = [load_seed(args.run_root, seed) for seed in args.seeds]
    report = aggregate(seeds)
    output_json = args.output_json or args.run_root / "study_summary.json"
    output_md = args.output_md or args.run_root / "study_summary.md"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

