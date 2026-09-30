#!/usr/bin/env python3
"""Summarize matched Swin best-checkpoint and early-stopped-checkpoint metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = []
    folds = []
    for fold in (0, 1):
        training = load_json(args.run_root / "models" / f"MSD_fold{fold}" / "training_summary.json")
        fold_result = {"fold": fold, "training": training, "checkpoints": {}}
        for checkpoint in ("best", "stopped"):
            metrics = {}
            for split in ("train", "val"):
                path = args.run_root / "inference" / f"fold{fold}_{checkpoint}_{split}" / "metrics_summary.json"
                metrics[split] = load_json(path)["metrics"]
            gap = float(metrics["train"]["dice_hard"] - metrics["val"]["dice_hard"])
            hd95_gap = float(metrics["val"]["hd95"] - metrics["train"]["hd95"])
            fold_result["checkpoints"][checkpoint] = {
                "train_metrics": metrics["train"],
                "validation_metrics": metrics["val"],
                "hard_dice_generalization_gap": gap,
                "hd95_generalization_gap": hd95_gap,
            }
            rows.append(
                {
                    "fold": fold,
                    "checkpoint": checkpoint,
                    "epoch": training["best_epoch"] if checkpoint == "best" else training["completed_epochs"],
                    "train_dice": metrics["train"]["dice_hard"],
                    "validation_dice": metrics["val"]["dice_hard"],
                    "gap": gap,
                    "train_hd95": metrics["train"]["hd95"],
                    "validation_hd95": metrics["val"]["hd95"],
                }
            )
        folds.append(fold_result)

    report = {
        "schema": "semantic_constraints.cst_teacher.swin_early_stopping.v1",
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# SwinUNETR early-stopping study",
        "",
        "| Fold | Checkpoint | Epoch | Train Dice | Validation Dice | Gap | Train HD95 | Validation HD95 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['checkpoint']} | {row['epoch']} | "
            f"{row['train_dice']:.4f} | {row['validation_dice']:.4f} | {row['gap']:+.4f} | "
            f"{row['train_hd95']:.3f} | {row['validation_hd95']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
