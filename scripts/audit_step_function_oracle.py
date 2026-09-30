#!/usr/bin/env python3
"""Measure the best compact sagittal step representation of annotated masks.

For each occupied run along y in each sagittal x slice, dynamic programming
minimizes foreground voxel symmetric difference using up to K constant z
intervals. Occupancy is kept exact. The resulting error is a label-only lower
bound for this representation, before any prediction head is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_K = (1, 2, 3, 4, 6, 8, 12, 16, 24)


def centered_64(label: np.ndarray) -> np.ndarray:
    """Match SpatialPadd(64), CenterSpatialCropd(64), DivisiblePadd(32)."""
    result = np.zeros((64, 64, 64), dtype=bool)
    source = []
    target = []
    for size in label.shape:
        if size >= 64:
            start = (size - 64) // 2
            source.append(slice(start, start + 64))
            target.append(slice(0, 64))
        else:
            start = (64 - size) // 2
            source.append(slice(0, size))
            target.append(slice(start, start + size))
    result[tuple(target)] = np.asarray(label[tuple(source)]) > 0
    return result


def occupied_runs(occupied: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(occupied.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def best_interval(counts: np.ndarray, length: int) -> tuple[int, int, int]:
    """The nonempty interval with minimum XOR against `length` binary columns."""
    gains = 2 * counts.astype(np.int32) - length
    prefix = np.empty(len(gains) + 1, dtype=np.int32)
    prefix[0] = 0
    np.cumsum(gains, out=prefix[1:])
    gains_by_end = prefix[1:] - np.minimum.accumulate(prefix[:-1])
    end = int(np.argmax(gains_by_end)) + 1
    begin = int(np.argmin(prefix[:end]))
    return begin, end - 1, int(gains_by_end[end - 1])


def fit_run(columns: np.ndarray, budgets: tuple[int, ...]) -> dict[int, list[tuple[int, int, int, int]]]:
    """Return optimal [y_start,y_end), [z_lower,z_upper] plateaus for each K."""
    n, depth = columns.shape
    counts = np.zeros(depth, dtype=np.int16)
    costs = np.full((n, n + 1), np.inf)
    lower = np.zeros((n, n + 1), dtype=np.int16)
    upper = np.zeros((n, n + 1), dtype=np.int16)
    for begin in range(n):
        counts.fill(0)
        total_true = 0
        for end in range(begin + 1, n + 1):
            counts += columns[end - 1]
            total_true += int(columns[end - 1].sum())
            lo, hi, gain = best_interval(counts, end - begin)
            costs[begin, end] = total_true - gain
            lower[begin, end] = lo
            upper[begin, end] = hi

    max_steps = min(max(budgets), n)
    dp = np.full((max_steps + 1, n + 1), np.inf)
    previous = np.zeros((max_steps + 1, n + 1), dtype=np.int16)
    dp[0, 0] = 0
    for steps in range(1, max_steps + 1):
        for end in range(steps, n + 1):
            candidates = dp[steps - 1, steps - 1 : end] + costs[steps - 1 : end, end]
            offset = int(np.argmin(candidates))
            previous[steps, end] = steps - 1 + offset
            dp[steps, end] = candidates[offset]

    result = {}
    for budget in budgets:
        steps = min(budget, n)
        end = n
        plateaus = []
        while steps:
            begin = int(previous[steps, end])
            plateaus.append((begin, end, int(lower[begin, end]), int(upper[begin, end])))
            end = begin
            steps -= 1
        merged = []
        for begin, stop, lo, hi in plateaus[::-1]:
            if merged and merged[-1][1] == begin and merged[-1][2:] == (lo, hi):
                merged[-1] = (merged[-1][0], stop, lo, hi)
            else:
                merged.append((begin, stop, lo, hi))
        result[budget] = merged
    return result


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + np.count_nonzero(b)))


def audit_case(label: np.ndarray, budgets: tuple[int, ...]) -> list[dict]:
    truth = centered_64(label)
    occupied = truth.any(axis=2)
    endpoint_columns = np.zeros_like(occupied)
    recon = {k: np.zeros_like(truth) for k in budgets}
    full_interval = np.zeros_like(truth)
    run_count = 0
    run_lengths = []
    occupied_columns = int(occupied.sum())
    plateau_counts = {k: 0 for k in budgets}
    for x in range(64):
        for y0, y1 in occupied_runs(occupied[x]):
            run_count += 1
            run_lengths.append(y1 - y0)
            endpoint_columns[x, y0] = True
            endpoint_columns[x, y1 - 1] = True
            columns = truth[x, y0:y1, :].astype(np.int16)
            z0 = columns.argmax(axis=1)
            z1 = 63 - columns[:, ::-1].argmax(axis=1)
            for offset, (lo, hi) in enumerate(zip(z0, z1)):
                full_interval[x, y0 + offset, lo : hi + 1] = True
            fits = fit_run(columns, budgets)
            for k, plateaus in fits.items():
                plateau_counts[k] += len(plateaus)
                for begin, end, lo, hi in plateaus:
                    recon[k][x, y0 + begin : y0 + end, lo : hi + 1] = True

    lower_true = truth.argmax(axis=2)
    upper_true = 63 - truth[:, :, ::-1].argmax(axis=2)
    rows = []
    for k in budgets:
        mask = recon[k]
        lower_fit = mask.argmax(axis=2)
        upper_fit = 63 - mask[:, :, ::-1].argmax(axis=2)
        edge_errors = np.concatenate(
            (np.abs(lower_fit[occupied] - lower_true[occupied]),
             np.abs(upper_fit[occupied] - upper_true[occupied]))
        )
        endpoint_errors = np.concatenate(
            (np.abs(lower_fit[endpoint_columns] - lower_true[endpoint_columns]),
             np.abs(upper_fit[endpoint_columns] - upper_true[endpoint_columns]))
        )
        rows.append({
            "k": k,
            "dice": dice(mask, truth),
            "full_interval_dice": dice(full_interval, truth),
            "xor_voxels": int(np.count_nonzero(mask ^ truth)),
            "false_positive_voxels": int(np.count_nonzero(mask & ~truth)),
            "false_negative_voxels": int(np.count_nonzero(truth & ~mask)),
            "edge_mae_voxels": float(edge_errors.mean()),
            "edge_p95_voxels": float(np.percentile(edge_errors, 95)),
            "edge_exact_fraction": float(np.mean(edge_errors == 0)),
            "edge_within_one_fraction": float(np.mean(edge_errors <= 1)),
            "run_endpoint_edge_mae_voxels": float(endpoint_errors.mean()),
            "run_endpoint_within_one_fraction": float(np.mean(endpoint_errors <= 1)),
            "runs": run_count,
            "median_run_length": float(np.median(run_lengths)),
            "max_run_length": int(max(run_lengths)),
            "plateaus": plateau_counts[k],
            "occupied_columns": occupied_columns,
            "foreground_voxels": int(truth.sum()),
            "interval_hole_voxels": int(np.count_nonzero(full_interval & ~truth)),
        })
    return rows


def summarize(rows: list[dict], budgets: tuple[int, ...]) -> dict:
    result = {}
    for cohort in ("train", "inner_val", "all_fold0_train"):
        cohort_rows = [r for r in rows if r["cohort"] == cohort or (cohort == "all_fold0_train")]
        result[cohort] = {}
        for k in budgets:
            items = [r for r in cohort_rows if r["k"] == k]
            result[cohort][str(k)] = {
                "cases": len(items),
                "mean_dice": float(np.mean([r["dice"] for r in items])),
                "median_dice": float(np.median([r["dice"] for r in items])),
                "min_dice": float(np.min([r["dice"] for r in items])),
                "mean_edge_mae_voxels": float(np.mean([r["edge_mae_voxels"] for r in items])),
                "mean_edge_within_one_fraction": float(np.mean([r["edge_within_one_fraction"] for r in items])),
                "mean_run_endpoint_edge_mae_voxels": float(np.mean([r["run_endpoint_edge_mae_voxels"] for r in items])),
                "mean_run_endpoint_within_one_fraction": float(np.mean([r["run_endpoint_within_one_fraction"] for r in items])),
                "total_xor_voxels": int(sum(r["xor_voxels"] for r in items)),
                "mean_plateaus_per_run": float(sum(r["plateaus"] for r in items) / sum(r["runs"] for r in items)),
                "mean_full_interval_dice": float(np.mean([r["full_interval_dice"] for r in items])),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_K)
    args = parser.parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    split = json.loads((args.dataset / "splits_final.json").read_text())[0]
    fold_train = sorted(split["train"])
    rng = np.random.default_rng(0)
    rng.shuffle(fold_train)
    inner_val = set(fold_train[:42])
    fold_train = set(fold_train)
    df = pd.read_pickle(args.dataset / "msd_hippocampus_full.pkl", compression="gzip")
    records = df.loc[df.subject_id.isin(fold_train)].sort_values("subject_id")
    if args.max_cases is not None:
        records = records.iloc[: args.max_cases]
    rows = []
    for index, (_, record) in enumerate(records.iterrows(), 1):
        name = str(record.subject_id)
        case_rows = audit_case(np.asarray(record.label_data), budgets)
        for row in case_rows:
            row.update(case=name, cohort="inner_val" if name in inner_val else "train")
        rows.extend(case_rows)
        if index % 20 == 0:
            print(f"Audited {index}/{len(records)} cases", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, budgets)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
