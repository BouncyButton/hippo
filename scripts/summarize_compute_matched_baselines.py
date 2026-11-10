#!/usr/bin/env python3
"""Verify all 27 compute-matched runs and summarize selected validation Dice."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

MODELS = ("unet", "swinunetr", "nnunet")
ARTIFACTS = {
    "config_sha256": "config.json",
    "checkpoint_best_sha256": "checkpoint_best.pt",
    "checkpoint_latest_sha256": "checkpoint_latest.pt",
    "metrics_sha256": "metrics.csv",
    "budget_ledger_sha256": "budget_ledger.json",
    "validation_best_sha256": "validation_best.json",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def aggregate(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "n": len(values)}


def summarize(root: Path, allow_incomplete=False, *, protocol_dir=None, models=None):
    protocol_dir = protocol_dir or root / "source/experiments/compute_matched_baselines_20261003"
    models = tuple(models) if models is not None else MODELS
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError("Expected distinct supported model names")
    protocol_path = protocol_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    splits = json.loads((protocol_dir / "splits_10patients.json").read_text())
    if digest(protocol_dir / "splits_10patients.json") != protocol["splits_sha256"]:
        raise ValueError("Campaign splits changed")
    if digest(protocol_dir / "nnunet_architecture.json") != protocol["nnunet_architecture_sha256"]:
        raise ValueError("Campaign nnU-Net topology changed")
    rows, missing = [], []
    expected_count = len(models) * len(protocol["folds"]) * len(protocol["seeds"])
    for model in models:
        for fold in protocol["folds"]:
            for seed in protocol["seeds"]:
                run = root / model / f"fold{fold}" / f"seed{seed}" / "run"
                completion = run / "completion.json"
                identity = (model, fold, seed)
                if not completion.exists():
                    missing.append(identity)
                    continue
                done = json.loads(completion.read_text())
                if done["status"] != "complete" or tuple(done[k] for k in ("model", "fold", "seed")) != identity:
                    raise ValueError(f"Invalid production completion: {run}")
                for key, name in ARTIFACTS.items():
                    if done[key] != digest(run / name):
                        raise ValueError(f"Changed artifact {name}: {run}")
                config = json.loads((run / "config.json").read_text())
                if tuple(config[k] for k in ("model", "fold", "seed")) != identity or config["smoke_test"]:
                    raise ValueError(f"Invalid run configuration: {run}")
                if config["protocol_sha256"] != digest(protocol_path) or config["dataset_sha256"] != protocol["pkl_sha256"]:
                    raise ValueError(f"Run did not use the frozen protocol/data: {run}")
                if config["script_sha256"] != digest(root / "source/scripts/train_compute_matched_baselines.py"):
                    raise ValueError(f"Run did not use campaign training source: {run}")
                if config["train_cases"] != splits[fold]["train"] or config["validation_cases"] != splits[fold]["val"]:
                    raise ValueError(f"Patient subset differs: {run}")
                cost = protocol["models"][model]["flops_per_update"]
                budget = protocol["budget_flops"]
                attempts = done["training_attempts_charged"]
                if (config["flops_per_update"] != cost or config["budget_flops"] != budget
                        or config["max_steps"] != budget // cost or attempts != budget // cost
                        or not 0 <= done["selected_step"] <= done["updates"] <= attempts
                        or done["discarded_attempts"] != attempts - done["updates"]
                        or done["consumed_training_flops"] != attempts * cost
                        or done["unused_training_flops"] != budget - attempts * cost
                        or attempts * cost > budget or (attempts + 1) * cost <= budget):
                    raise ValueError(f"Training-budget accounting differs: {run}")
                ledger = json.loads((run / "budget_ledger.json").read_text())
                if (ledger["charged_attempts"] != attempts or ledger["charged_training_flops"] != attempts * cost
                        or ledger["config_sha256"] != done["config_sha256"]):
                    raise ValueError(f"Budget ledger differs: {run}")
                selected = json.loads((run / "validation_best.json").read_text())
                if [r["case_name"] for r in selected["cases"]] != splits[fold]["val"]:
                    raise ValueError(f"Selected validation cases differ: {run}")
                score = selected["summary"]["macro_hard_dice"]
                if not math.isfinite(score) or not 0 <= score <= 1 or score != done["best_validation_hard_dice"]:
                    raise ValueError(f"Selected Dice score differs: {run}")
                with (run / "metrics.csv").open(newline="") as stream:
                    history = list(csv.DictReader(stream))
                milestone_set = {math.ceil(k * attempts / protocol["validation_milestones"])
                                 for k in range(1, protocol["validation_milestones"] + 1)}
                if (not history or int(history[-1]["charged_attempts"]) != attempts
                        or len(history) > protocol["validation_milestones"]
                        or any(int(r["charged_attempts"]) not in milestone_set for r in history)
                        or max(float(r["macro_hard_dice"]) for r in history) != score):
                    raise ValueError(f"Validation selection history differs: {run}")
                rows.append({"model": model, "fold": fold, "seed": seed, "updates": done["updates"],
                             "charged_attempts": attempts, "discarded_attempts": done["discarded_attempts"],
                             "patient_passes": done["updates"] / 10, "selected_step": done["selected_step"],
                             "consumed_gflops": done["consumed_training_flops"] / 1e9,
                             **selected["summary"]})
    if missing:
        if not allow_incomplete:
            raise ValueError(f"Missing {len(missing)} of {expected_count} runs: {missing}")
        print(f"Verified {len(rows)}/{expected_count} completed runs; aggregate waits for all runs.", flush=True)
        return None
    by_model = {}
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        by_model[model] = {
            "selected_validation_macro_hard_dice": aggregate([r["macro_hard_dice"] for r in subset]),
            "folds": {str(f): aggregate([r["macro_hard_dice"] for r in subset if r["fold"] == f])
                      for f in protocol["folds"]},
            "runs_with_discarded_attempts": sum(r["discarded_attempts"] > 0 for r in subset),
        }
    summary = {"runs": rows, "models": by_model, "budget_gflops_per_run": protocol["budget_flops"] / 1e9,
               "notes": ["Dice is selected on validation data; this is not independent held-out test performance.",
                         "Standard deviations describe runs; overlapping folds and repeated seeds are not independent datasets.",
                         "Interrupted attempts consume allowance; inspect discarded_attempts before comparing trajectories."]}
    atomic_text(root / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(root / "summary.csv", buffer.getvalue())
    markdown = ["# Compute-matched baseline validation results", "",
                f"Selected validation macro hard Dice, {len(protocol['folds']) * len(protocol['seeds'])} runs per model "
                f"({len(protocol['folds'])} folds × {len(protocol['seeds'])} seeds).", "",
                "| Model | Mean Dice | Sample SD | Interrupted runs |", "| --- | ---: | ---: | ---: |"]
    for model, stats in by_model.items():
        score = stats["selected_validation_macro_hard_dice"]
        markdown.append(f"| {model} | {score['mean']:.4f} | {score['sample_std']:.4f} | {stats['runs_with_discarded_attempts']} |")
    markdown += ["", *summary["notes"], ""]
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(by_model, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    args = parser.parse_args()
    with (args.root / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summarize(args.root, args.allow_incomplete, protocol_dir=args.protocol_dir, models=args.models)


if __name__ == "__main__":
    main()
