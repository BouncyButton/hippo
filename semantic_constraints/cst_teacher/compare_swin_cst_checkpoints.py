#!/usr/bin/env python3
"""Compare frozen CST diagnostics on matched old and early-stopped Swin folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-fold0-teachers", type=Path, required=True)
    parser.add_argument("--old-fold0-risk", type=Path, required=True)
    parser.add_argument("--old-fold0-utility", type=Path, required=True)
    parser.add_argument("--old-fold1-teachers", type=Path, required=True)
    parser.add_argument("--old-fold1-risk", type=Path, required=True)
    parser.add_argument("--old-fold1-utility", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def capture(utility: dict, feature: str, fraction: float) -> float:
    rows = utility["results"][feature]["global_slice_selection"]
    return next(row["error_mass_capture"] for row in rows if abs(row["flagged_fraction"] - fraction) < 1e-8)


def compact(teacher_dir: Path, risk_dir: Path, utility_path: Path, *, old: bool = False) -> dict:
    seed_dir = lambda seed: f"dense32_smooth_seed_{seed}" if old else f"seed_{seed}"
    evaluations = [read(teacher_dir / seed_dir(seed) / "prediction_evaluation.json") for seed in range(3)]
    risk = read(risk_dir / "risk_probe_report.json")
    utility = read(utility_path)
    names = [row["case_name"] for row in evaluations[0]["per_case"]]
    for item in evaluations[1:]:
        if [row["case_name"] for row in item["per_case"]] != names:
            raise ValueError("CST seeds evaluated different cases or case order")
    consensus = []
    for index, name in enumerate(names):
        votes = sum(
            item["per_case"][index]["descriptors"]["anterior_volume_fraction"]["prediction_normalized_violation"] > 0
            for item in evaluations
        )
        if votes >= 2:
            consensus.append(name)
    dice_by_name = {row["case_name"]: row["foreground_dice"] for row in evaluations[0]["per_case"]}
    unflagged = [name for name in names if name not in consensus]
    return {
        "case_names": names,
        "mean_cst_foreground_dice": mean(item["mean_foreground_dice"] for item in evaluations),
        "anterior_consensus_2of3": {
            "cases": consensus,
            "mean_flagged_dice": mean(dice_by_name[name] for name in consensus) if consensus else None,
            "mean_unflagged_dice": mean(dice_by_name[name] for name in unflagged) if unflagged else None,
        },
        "anterior_violation_error_r": [
            item["descriptor_summary"]["anterior_volume_fraction"]["violation_vs_dice_error_pearson"]
            for item in evaluations
        ],
        "profile_violation_error_r": [
            item["profile_summary"]["violation_vs_dice_error_pearson"] for item in evaluations
        ],
        "slice_r": {
            feature: utility["results"][feature]["slice_pearson"]
            for feature in ("uncertainty", "combined")
        },
        "top_20pct_error_capture": {
            feature: capture(utility, feature, 0.20)
            for feature in ("uncertainty", "combined")
        },
        "within_patient_top4_error_capture": {
            feature: utility["results"][feature]["within_case_selection"]["mean_error_mass_capture"]
            for feature in ("uncertainty", "combined")
        },
        "nested_slice_r": {
            feature: risk["aggregate"]["slice"][feature]["pearson"]["mean"]
            for feature in ("uncertainty", "combined")
        },
    }


def main() -> None:
    args = parse_args()
    folds = []
    for fold in (0, 1):
        old_teacher = getattr(args, f"old_fold{fold}_teachers")
        old_risk = getattr(args, f"old_fold{fold}_risk")
        old_utility = getattr(args, f"old_fold{fold}_utility")
        new_dir = args.new_root / f"fold{fold}"
        old = compact(old_teacher, old_risk, old_utility, old=True)
        new = compact(new_dir / "teacher", new_dir / "risk", new_dir / "risk_utility.json")
        if old["case_names"] != new["case_names"]:
            raise ValueError(f"fold {fold} cases or order differ between checkpoints")
        folds.append({"fold": fold, "cases": len(old["case_names"]), "epoch50": old, "early_best": new})

    report = {"schema": "semantic_constraints.cst_teacher.swin_cst_checkpoint_comparison.v1", "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CST signals: epoch-50 versus early-stopped best Swin",
        "",
        "| Fold | Checkpoint | CST Dice | Anterior violation/error r (3 seeds) | Profile violation/error r (3 seeds) | Slice r uncertainty → combined | Top-20% capture uncertainty → combined |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for item in folds:
        for name, key in (("epoch 50", "epoch50"), ("early best", "early_best")):
            data = item[key]
            anterior = ", ".join("nan" if v is None else f"{v:+.3f}" for v in data["anterior_violation_error_r"])
            profile = ", ".join("nan" if v is None else f"{v:+.3f}" for v in data["profile_violation_error_r"])
            lines.append(
                f"| {item['fold']} | {name} | {data['mean_cst_foreground_dice']:.4f} | "
                f"{anterior} | {profile} | {data['slice_r']['uncertainty']:.3f} → "
                f"{data['slice_r']['combined']:.3f} | "
                f"{data['top_20pct_error_capture']['uncertainty']:.1%} → "
                f"{data['top_20pct_error_capture']['combined']:.1%} |"
            )
    lines.extend(("", "## Anterior-volume consensus flags", ""))
    for item in folds:
        for name, key in (("epoch 50", "epoch50"), ("early best", "early_best")):
            flags = item[key]["anterior_consensus_2of3"]
            flagged = flags["mean_flagged_dice"]
            lines.append(
                f"- Fold {item['fold']} {name}: {len(flags['cases'])} flags "
                f"({', '.join(flags['cases']) or 'none'}); "
                f"flagged Dice {'n/a' if flagged is None else f'{flagged:.4f}'} versus "
                f"unflagged Dice {flags['mean_unflagged_dice']:.4f}."
            )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
