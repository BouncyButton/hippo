#!/usr/bin/env python3
"""Aggregate three-seed patient-held-out error-direction probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for fold in (0, 1):
        items = [json.loads((args.run_root / f"seed_{seed}_fold{fold}.json").read_text(encoding="utf-8")) for seed in (0, 1, 2)]
        if not all(item["distribution"] == items[0]["distribution"] for item in items):
            raise ValueError("error taxonomy changed across teacher seeds")
        reports[fold] = items
    lines = [
        "# CST error-direction probe",
        "",
        "Patient-grouped 5-fold CV. Conditional target: missing versus extra tissue",
        "on slices where one of those errors contributes at least 60% of >=5 wrong voxels.",
        "This is a discovery analysis, not a correction result or prospective test.",
        "",
        "| Fold | QC top-20% composition: minimal/missing/extra/swap/mixed | Model | Directional accuracy | Majority baseline | AUC | Confident coverage | Confident accuracy |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for fold, items in reports.items():
        first = items[0]
        selected = first["selected_distribution"]
        composition = "/".join(str(selected[name]) for name in ("minimal", "missing", "extra", "ap_swap", "mixed"))
        for kind in ("uncertainty", "portable", "combined"):
            metrics = [item["models"][kind]["qc_top20_directional"] for item in items]
            def show(key: str) -> str:
                values = np.asarray([metric[key] for metric in metrics], dtype=float)
                return f"{values.mean():.3f} ± {values.std():.3f}"
            coverage = np.asarray([metric["confident_directional_slices"] / metric["directional_slices"] for metric in metrics])
            confident_values = [metric["confident_directional_accuracy"] for metric in metrics]
            confident = "n/a" if any(value is None for value in confident_values) else f"{np.mean(confident_values):.3f} ± {np.std(confident_values):.3f}"
            lines.append(
                f"| {fold} | {composition} | {kind} | {show('accuracy')} | "
                f"{show('majority_accuracy')} | {show('auc')} | "
                f"{coverage.mean():.3f} ± {coverage.std():.3f} | {confident} |"
            )
    lines += ["", "Do not launch direction-based correction unless the directional head beats the majority and uncertainty baselines on both folds, and a patient-bootstrap lower bound is positive.", ""]
    destination = args.run_root / "summary.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
