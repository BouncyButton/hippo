"""Reconstruct Tables 2 and 3 from final-checkpoint validation Dice."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from constraint_gap import ARMS, ROOT, cohort, comparison_rows


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["model"], row["objective"]].append(row)
    for (dataset, model, objective), cells in groups.items():
        pairs = {(r["fold"], r["seed"]) for r in cells}
        if pairs != cohort(dataset, model) or len(cells) != len(pairs):
            raise ValueError(f"Incomplete or duplicated cohort for {dataset} {model} {objective}")
        if any(r["gain_pp"] is None for r in cells):
            raise ValueError(f"Missing final-checkpoint score for {dataset} {model} {objective}")
        folds = defaultdict(list)
        for row in cells:
            folds[row["fold"]].append(row)
        result = dict(dataset=dataset, model=model, objective=objective)
        for field in ("baseline_pct", "objective_pct", "gain_pp"):
            means = [mean(r[field] for r in fold) for fold in folds.values()]
            result[field] = mean(means)
            result[field + "_fold_sd"] = stdev(means)
        result["improved_pairs"] = sum(r["gain_pp"] > 0 for r in cells)
        yield result


def manuscript_rows(rows):
    """Use the paper's highest-mean rule. Preserve exact ties."""
    groups = defaultdict(list)
    for row in summaries(rows):
        groups[row["dataset"], row["model"]].append(row)
    for group, arms in groups.items():
        if {r["objective"] for r in arms} != set(ARMS):
            raise ValueError(f"Incomplete objective set for {group}")
        maximum = max(r["objective_pct"] for r in arms)
        yield from (r for r in arms if r["objective_pct"] == maximum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "controls.csv").open() as stream:
        rows = list(manuscript_rows(comparison_rows(csv.DictReader(stream), ROOT)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
