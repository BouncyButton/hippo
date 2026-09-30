#!/usr/bin/env python3
"""Quantify Swin train/validation gaps and late-epoch degradation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for fold in (0, 1):
        parser.add_argument(f"--fold{fold}-train-metrics", type=Path, required=True)
        parser.add_argument(f"--fold{fold}-val-metrics", type=Path, required=True)
        parser.add_argument(f"--fold{fold}-train-log", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["metrics"].items()}


def parse_curve(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    epochs, losses, validation = [], [], []
    pending_epoch = None
    for line in lines:
        epoch_match = re.search(r"Epoch (\d+)/(\d+), Loss: ([0-9.eE+-]+)", line)
        if epoch_match:
            pending_epoch = int(epoch_match.group(1))
            epochs.append(pending_epoch)
            losses.append(float(epoch_match.group(3)))
            continue
        dice_match = re.search(r"Val Dice \(hard\): ([0-9.eE+-]+)", line)
        if dice_match and pending_epoch is not None:
            validation.append((pending_epoch, float(dice_match.group(1))))
            pending_epoch = None
    if not losses or not validation:
        raise ValueError(f"could not parse training curve from {path}")
    peak_epoch, peak_dice = max(validation, key=lambda item: item[1])
    final_epoch, final_dice = validation[-1]
    return {
        "epochs": len(epochs),
        "initial_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "peak_validation_dice": peak_dice,
        "peak_validation_epoch": peak_epoch,
        "final_logged_validation_dice": final_dice,
        "final_logged_epoch": final_epoch,
        "peak_to_final_dice_drop": peak_dice - final_dice,
    }


def severity(dice_gap: float, peak_drop: float) -> str:
    if dice_gap > 0.10 or peak_drop > 0.03:
        return "severe"
    if dice_gap > 0.05 or peak_drop > 0.015:
        return "concerning"
    if dice_gap > 0.02 or peak_drop > 0.005:
        return "mild"
    return "low"


def main() -> None:
    args = parse_args()
    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    folds = []
    for fold in (0, 1):
        train_path = getattr(args, f"fold{fold}_train_metrics")
        val_path = getattr(args, f"fold{fold}_val_metrics")
        log_path = getattr(args, f"fold{fold}_train_log")
        train = load_metrics(train_path)
        validation = load_metrics(val_path)
        curve = parse_curve(log_path)
        split = splits[fold]
        overlap = sorted(set(split["train"]) & set(split["val"]))
        dice_gap = train["dice_hard"] - validation["dice_hard"]
        result = {
            "fold": fold,
            "train_cases": len(split["train"]),
            "validation_cases": len(split["val"]),
            "patient_overlap": overlap,
            "train_metrics": train,
            "validation_metrics": validation,
            "hard_dice_generalization_gap": dice_gap,
            "hd95_generalization_gap": validation["hd95"] - train["hd95"],
            "curve": curve,
            "heuristic_severity": severity(dice_gap, curve["peak_to_final_dice_drop"]),
        }
        folds.append(result)

    report = {
        "schema": "semantic_constraints.cst_teacher.swin_overfit_audit.v1",
        "folds": folds,
        "thresholds": {
            "severe": "Dice gap > 0.10 or peak-to-final drop > 0.03",
            "concerning": "Dice gap > 0.05 or peak-to-final drop > 0.015",
            "mild": "Dice gap > 0.02 or peak-to-final drop > 0.005",
            "low": "otherwise",
        },
        "warning": "Severity thresholds are diagnostic heuristics, not clinical acceptance criteria.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# SwinUNETR overfitting audit",
        "",
        "| Fold | Train Dice | Validation Dice | Gap | Peak→final drop | Train HD95 | Validation HD95 | Severity |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in folds:
        lines.append(
            f"| {item['fold']} | {item['train_metrics']['dice_hard']:.4f} | "
            f"{item['validation_metrics']['dice_hard']:.4f} | "
            f"{item['hard_dice_generalization_gap']:+.4f} | "
            f"{item['curve']['peak_to_final_dice_drop']:.4f} | "
            f"{item['train_metrics']['hd95']:.3f} | {item['validation_metrics']['hd95']:.3f} | "
            f"{item['heuristic_severity']} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
