#!/usr/bin/env python3
"""Cross-fit signed profile corrections and test them on frozen Swin outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_constraints.cst_teacher.descriptors import sampled_slice_profiles  # noqa: E402
from semantic_constraints.cst_teacher.evaluate_predictions import (  # noqa: E402
    foreground_dice,
    prediction_map,
    validation_label_map,
)
from semantic_constraints.cst_teacher.profile_counterfactual import repair, soft_dice  # noqa: E402
from semantic_constraints.cst_teacher.risk_probe import (  # noqa: E402
    correlation,
    extract_seed_features,
    ranks,
    repeated_nested_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=REPO_ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl",
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=REPO_ROOT / "datasets/Dataset101_MSD/splits_final.json",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0))
    parser.add_argument("--repair-modes", nargs="+", default=("slice_ap_bias", "slice_fg_ap_bias"))
    parser.add_argument("--repair-gammas", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2))
    parser.add_argument("--repair-steps", type=int, default=100)
    parser.add_argument("--repair-learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def correction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(target - prediction)
    zero_absolute = np.abs(target)
    active = np.abs(target) >= 0.01
    per_class_pearson = [correlation(target[:, index], prediction[:, index]) for index in range(2)]
    per_class_spearman = [
        correlation(ranks(target[:, index]), ranks(prediction[:, index])) for index in range(2)
    ]
    return {
        "mae": float(absolute.mean()),
        "zero_baseline_mae": float(zero_absolute.mean()),
        "mae_improvement_vs_zero": float(zero_absolute.mean() - absolute.mean()),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "pearson_per_class": per_class_pearson,
        "mean_pearson": mean(value for value in per_class_pearson if value is not None),
        "spearman_per_class": per_class_spearman,
        "mean_spearman": mean(value for value in per_class_spearman if value is not None),
        "sign_accuracy_active": float(np.mean(np.sign(target[active]) == np.sign(prediction[active]))),
        "active_entries": int(active.sum()),
    }


def crossfit_corrections(features: dict[str, Any], args: argparse.Namespace, seed: int) -> dict[str, Any]:
    target = features["slice_correction_target"]
    groups = features["slice_groups"]
    results = {}
    for feature_index, (feature_name, x) in enumerate(features["slice_features"].items()):
        class_predictions = []
        for class_index in range(2):
            prediction, _ = repeated_nested_predictions(
                x,
                target[:, class_index],
                groups,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                repeats=args.repeats,
                alphas=list(args.alphas),
                seed=seed + 101 * feature_index + 1009 * class_index,
            )
            class_predictions.append(prediction)
        prediction = np.stack(class_predictions, axis=1)
        results[feature_name] = {
            "metrics": correction_metrics(target, prediction),
            "prediction": prediction,
        }
    teacher_prediction = features["slice_teacher_correction"]
    results["raw_teacher_residual"] = {
        "metrics": correction_metrics(target, teacher_prediction),
        "prediction": teacher_prediction,
    }
    return results


def load_repair_cases(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    from semantic_constraints.cst_teacher.evaluate_predictions import load_teachers
    from semantic_constraints.cst_teacher.train_teacher import build_loaders

    _, _, payload = load_teachers(checkpoint, device)
    config = payload["config"]
    args.fold = int(config["fold"])
    args.spatial_size = tuple(config["spatial_size"])
    args.set_size = int(config["set_size"])
    args.slab_depth = int(config["slab_depth"])
    args.inplane_size = int(config.get("inplane_size", 32))
    args.cache_dataset = True
    args.max_train_cases = 0
    args.max_val_cases = 0
    _, loader = build_loaders(args)
    paths = prediction_map(args.inference_dir)
    labels = validation_label_map(loader)
    cases = []
    for batch in loader:
        positions = batch["positions"][0]
        for name in [str(value) for value in batch["case_name"]]:
            probability = torch.from_numpy(np.load(paths[name])).float().unsqueeze(0)
            cases.append(
                {
                    "case_name": name,
                    "probabilities": probability,
                    "label": labels[name].unsqueeze(0),
                    "positions": positions.clone(),
                    "prediction_profile": sampled_slice_profiles(probability, positions)[0],
                }
            )
    return cases


def evaluate_repairs(
    cases: list[dict[str, Any]],
    correction: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    correction_by_case = correction.reshape(len(cases), -1, 2)
    original_dice = np.asarray(
        [float(foreground_dice(case["probabilities"], case["label"])[0]) for case in cases]
    )
    worst = set(np.argsort(original_dice)[:10].tolist())
    configurations = []
    for mode in args.repair_modes:
        for gamma in args.repair_gammas:
            rows = []
            for index, case in enumerate(cases):
                original = case["probabilities"].to(device)
                label = case["label"].to(device)
                positions = case["positions"].to(device)
                desired = (
                    case["prediction_profile"].to(device) + original.new_tensor(correction_by_case[index])
                ).clamp(0.0, 1.0).unsqueeze(0)
                repaired, diagnostics = repair(
                    original,
                    positions,
                    desired,
                    mode=mode,
                    gamma=gamma,
                    steps=args.repair_steps,
                    learning_rate=args.repair_learning_rate,
                )
                hard_before = foreground_dice(original, label)[0]
                hard_after = foreground_dice(repaired, label)[0]
                soft_before = soft_dice(original, label)[0]
                soft_after = soft_dice(repaired, label)[0]
                rows.append(
                    {
                        "case_name": case["case_name"],
                        "original_hard_dice": float(hard_before),
                        "hard_dice_delta": float(hard_after - hard_before),
                        "soft_dice_delta": float(soft_after - soft_before),
                        **diagnostics,
                    }
                )
            hard = np.asarray([row["hard_dice_delta"] for row in rows])
            soft = np.asarray([row["soft_dice_delta"] for row in rows])
            configurations.append(
                {
                    "summary": {
                        "mode": mode,
                        "gamma": gamma,
                        "mean_hard_dice_delta": float(hard.mean()),
                        "mean_soft_dice_delta": float(soft.mean()),
                        "hard_improvement_rate": float(np.mean(hard > 1e-6)),
                        "hard_harm_rate": float(np.mean(hard < -1e-6)),
                        "worst10_mean_hard_dice_delta": float(hard[list(worst)].mean()),
                        "worst10_mean_soft_dice_delta": float(soft[list(worst)].mean()),
                        "mean_initial_profile_error": float(
                            np.mean([row["initial_profile_error"] for row in rows])
                        ),
                        "mean_final_profile_error": float(
                            np.mean([row["final_profile_error"] for row in rows])
                        ),
                    },
                    "per_case": rows,
                }
            )
            print(json.dumps(configurations[-1]["summary"], sort_keys=True), flush=True)
    return configurations


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-fitted CST signed residual correction",
        "",
        "## Slice correction prediction",
        "",
        "| Features | MAE | Δ MAE vs zero | Pearson | Sign accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["aggregate_prediction_metrics"].items():
        lines.append(
            f"| {name} | {metrics['mae']['mean']:.4f} ± {metrics['mae']['std']:.4f} | "
            f"{metrics['mae_improvement_vs_zero']['mean']:+.4f} | "
            f"{metrics['mean_pearson']['mean']:.3f} ± {metrics['mean_pearson']['std']:.3f} | "
            f"{metrics['sign_accuracy_active']['mean']:.1%} |"
        )
    lines.extend(
        (
            "",
            "## Ensemble OOF correction repair",
            "",
            "| Mode | Gamma | Hard Dice Δ | Soft Dice Δ | Worst-10 hard Δ | Harm |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for config in report["repair_configurations"]:
        item = config["summary"]
        lines.append(
            "| {mode} | {gamma:g} | {mean_hard_dice_delta:+.6f} | {mean_soft_dice_delta:+.6f} | "
            "{worst10_mean_hard_dice_delta:+.6f} | {hard_harm_rate:.1%} |".format(**item)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    from semantic_constraints.cst_teacher.train_teacher import resolve_device, seed_everything

    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports, ensemble_predictions = [], []
    reference = None
    for seed_index, checkpoint in enumerate(args.checkpoints):
        print(json.dumps({"stage": "crossfit", "seed": seed_index}), flush=True)
        features = extract_seed_features(checkpoint, args, device)
        if reference is None:
            reference = features
        elif not np.array_equal(reference["slice_groups"], features["slice_groups"]):
            raise ValueError("inconsistent slice order across teachers")
        results = crossfit_corrections(features, args, args.seed + 10000 * seed_index)
        for name, result in results.items():
            print(json.dumps({"seed": seed_index, "features": name, **result["metrics"]}), flush=True)
        ensemble_predictions.append(results["combined"]["prediction"])
        seed_reports.append(
            {
                "seed_index": seed_index,
                "checkpoint": str(checkpoint.resolve()),
                "metrics": {name: result["metrics"] for name, result in results.items()},
            }
        )

    names = seed_reports[0]["metrics"].keys()
    aggregate = {}
    for name in names:
        aggregate[name] = {}
        for metric in ("mae", "mae_improvement_vs_zero", "mean_pearson", "mean_spearman", "sign_accuracy_active"):
            values = [report["metrics"][name][metric] for report in seed_reports]
            aggregate[name][metric] = {"mean": mean(values), "std": pstdev(values), "per_seed": values}

    correction = np.median(np.stack(ensemble_predictions), axis=0)
    cases = load_repair_cases(args.checkpoints[0], args, device)
    repair_configurations = evaluate_repairs(cases, correction, args, device)
    report = {
        "schema": "semantic_constraints.cst_teacher.residual_correction.v1",
        "config": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "repeats": args.repeats,
            "ensemble": "median of three independently cross-fitted dense CST teachers",
        },
        "seed_reports": seed_reports,
        "aggregate_prediction_metrics": aggregate,
        "repair_configurations": repair_configurations,
        "warning": "Fold-0 discovery experiment. Freeze choices before evaluation on another fold.",
    }
    path = args.output_dir / "residual_correction_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "residual_correction_report.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
