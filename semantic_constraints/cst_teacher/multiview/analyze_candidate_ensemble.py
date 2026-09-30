#!/usr/bin/env python3
"""Predeclared rank-consensus of separately trained view-specific cut scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ENSEMBLES = {
    "three_single_views": ("coronal32", "sagittal32", "axial32"),
    "two_triview_budgets": ("triview32", "triview96"),
    "all_five_views": ("coronal32", "sagittal32", "axial32", "triview32", "triview96"),
}


def rank_consensus(score_sets: list[np.ndarray]) -> int:
    ranks = [np.argsort(np.argsort(scores)) for scores in score_sets]
    return int(np.argmax(np.mean(ranks, axis=0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    models = {item["view"]: item["patients"] for item in source["models"]}
    names = [row["case_name"] for row in models["coronal32"]]
    if any([row["case_name"] for row in patients] != names for patients in models.values()):
        raise ValueError("patient orders differ across view rankers")
    report = {"schema": "semantic_constraints.cst_teacher.multiview_candidate_ensemble.v1",
              "fold": source["fold"], "ensembles": {}}
    for ensemble, views in ENSEMBLES.items():
        rows = []
        for index, name in enumerate(names):
            base = models[views[0]][index]
            position = rank_consensus([np.asarray(models[view][index]["scores"]) for view in views])
            selected = int(base["swin_cut_y"] + position - 5)
            rows.append({"case_name": name, "selected_cut_y": selected,
                         "true_cut_y": base["true_cut_y"], "swin_cut_y": base["swin_cut_y"]})
        baseline = np.asarray([abs(row["swin_cut_y"] - row["true_cut_y"]) for row in rows])
        selected = np.asarray([abs(row["selected_cut_y"] - row["true_cut_y"]) for row in rows])
        report["ensembles"][ensemble] = {"views": views,
            "swin_mae": float(baseline.mean()), "ensemble_mae": float(selected.mean()),
            "patients_better": int(np.sum(selected < baseline)),
            "patients_worse": int(np.sum(selected > baseline)), "patients": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: {field: value for field, value in item.items() if field != "patients"}
                      for key, item in report["ensembles"].items()}, indent=2))


if __name__ == "__main__":
    main()
