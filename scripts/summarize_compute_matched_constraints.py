#!/usr/bin/env python3
"""Verify a calibrated compute campaign and publish validation results."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

import train_compute_matched_baselines as base
from train_compute_matched_constraints import ARTIFACTS, DEFAULT_PROTOCOL


def summarize(root, allow_incomplete=False):
    protocol_dir = root / "source" / DEFAULT_PROTOCOL.relative_to(base.REPO)
    protocol_path = protocol_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    splits = json.loads((protocol_dir / "splits_10patients.json").read_text())
    if base.digest(protocol_dir / "splits_10patients.json") != protocol["splits_sha256"]:
        raise ValueError("Campaign splits changed")
    manifest = json.loads((root / "source_manifest.json").read_text())
    if any(base.digest(root / "source" / name) != expected for name, expected in manifest.items()):
        raise ValueError("Campaign source changed")
    rows, missing = [], []
    for model, settings in protocol["models"].items():
        for arm, schedule in settings["objectives"].items():
            for fold in protocol["folds"]:
                for seed in protocol["seeds"]:
                    run = root / model / arm / f"fold{fold}" / f"seed{seed}" / "run"
                    identity = (model, arm, fold, seed)
                    if not (run / "completion.json").exists():
                        missing.append(identity)
                        continue
                    done = json.loads((run / "completion.json").read_text())
                    config = json.loads((run / "config.json").read_text())
                    if done["status"] != "complete" or config["smoke_test"] or tuple(done[k] for k in ("model", "objective", "fold", "seed")) != identity:
                        raise ValueError(f"Invalid run identity: {run}")
                    for key, filename in ARTIFACTS.items():
                        if done[key] != base.digest(run / filename):
                            raise ValueError(f"Changed artifact {filename}: {run}")
                    if (config["schedule"] != schedule or config["max_steps"] != schedule["max_steps"]
                            or config["budget_flops"] != protocol["budget_flops"]
                            or config["protocol_sha256"] != base.digest(protocol_path)
                            or config["dataset_sha256"] != protocol["pkl_sha256"]
                            or config["train_cases"] != splits[fold]["train"]
                            or config["validation_cases"] != splits[fold]["val"]):
                        raise ValueError(f"Configuration differs: {run}")
                    for key, filename in (("script_sha256", "train_compute_matched_constraints.py"),
                                          ("objective_source_sha256", "compute_constraint_objectives.py"),
                                          ("shared_source_sha256", "train_compute_matched_baselines.py")):
                        if config[key] != base.digest(root / "source/scripts" / filename):
                            raise ValueError(f"Training source differs: {run}")
                    if config["sdf_source_sha256"] != base.digest(root / "source/thesis/new_constraints/signed_distance.py"):
                        raise ValueError(f"Signed-distance source differs: {run}")
                    ledger = json.loads((run / "budget_ledger.json").read_text())
                    attempts = ledger["charged_attempts"]
                    prefix = min(attempts, schedule["prefix_updates"])
                    training_cost = prefix * schedule["baseline_flops_per_update"] + (attempts - prefix) * schedule["flops_per_update"]
                    calibration_cost = ledger["calibration_attempts"] * schedule["calibration_flops"]
                    total = training_cost + calibration_cost
                    if (ledger["config_sha256"] != done["config_sha256"] or ledger["training_flops"] != training_cost
                            or ledger["calibration_flops"] != calibration_cost or done["consumed_total_flops"] != total
                            or done["consumed_training_flops"] != training_cost or done["calibration_flops"] != calibration_cost
                            or done["training_attempts_charged"] != attempts or attempts > schedule["max_steps"]
                            or not 0 < done["updates"] <= attempts or done["discarded_attempts"] != attempts - done["updates"]
                            or done["unused_flops"] != protocol["budget_flops"] - total
                            or not 0 <= done["unused_flops"] < schedule["flops_per_update"]):
                        raise ValueError(f"Budget differs: {run}")
                    calibration = json.loads((run / "calibration.json").read_text())
                    if arm != "dice":
                        if (calibration["status"] != "complete" or ledger["calibration_attempts"] < 1
                                or calibration["charged_flops"] != schedule["calibration_flops"]
                                or [r["case_name"] for r in calibration["cases"]] != splits[fold]["train"]):
                            raise ValueError(f"Training-only calibration differs: {run}")
                        for name, weight in calibration["weights"].items():
                            stats = calibration["statistics"][name]
                            if not math.isfinite(weight) or weight <= 0 or stats["weighted_median_ratio"] > .1 + 1e-8 or stats["weighted_p95_ratio"] > .5 + 1e-8:
                                raise ValueError(f"Calibration limits differ: {run}")
                    elif ledger["calibration_attempts"] or calibration["status"] != "not_required":
                        raise ValueError(f"Unexpected baseline calibration: {run}")
                    selected = json.loads((run / "validation_best.json").read_text())
                    if [r["case_name"] for r in selected["cases"]] != splits[fold]["val"]:
                        raise ValueError(f"Validation cases differ: {run}")
                    score = selected["summary"]["macro_hard_dice"]
                    with (run / "metrics.csv").open() as stream:
                        history = list(csv.DictReader(stream))
                    if (not math.isfinite(score) or not 0 <= score <= 1 or score != done["best_validation_hard_dice"]
                            or max(float(r["macro_hard_dice"]) for r in history) != score
                            or int(history[-1]["charged_attempts"]) != attempts
                            or not 0 < done["selected_step"] <= done["updates"]):
                        raise ValueError(f"Checkpoint selection differs: {run}")
                    rows.append({"model": model, "objective": arm, "fold": fold, "seed": seed,
                                 "updates": done["updates"], "patient_pass_epochs": done["updates"] / 10,
                                 "selected_step": done["selected_step"], "total_gflops": total / 1e9,
                                 "calibration_gflops": calibration_cost / 1e9,
                                 "discarded_attempts": done["discarded_attempts"],
                                 "calibration_retries": max(0, ledger["calibration_attempts"] - 1),
                                 **selected["summary"]})
    if missing:
        if not allow_incomplete:
            raise ValueError(f"Missing {len(missing)} runs")
        print(f"Verified {len(rows)} completed runs; {len(missing)} pending", flush=True)
        return None
    groups = {}
    for model, settings in protocol["models"].items():
        for arm in settings["objectives"]:
            values = [r["macro_hard_dice"] for r in rows if r["model"] == model and r["objective"] == arm]
            groups[model + "/" + arm] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else 0,
                                        "n": len(values)}
    summary = {"runs": rows, "groups": groups, "budget_gflops_per_run": protocol["budget_flops"] / 1e9,
               "notes": ["Selected validation scores, not independent held-out test performance.",
                         "Fold/seed standard deviations are descriptive; runs are not independent datasets.",
                         "Warm-start training, calibration and all attempted updates are charged to each run."]}
    base.atomic_json(root / "summary.json", summary)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "summary.csv").write_text(buffer.getvalue())
    lines = ["# Calibrated compute-matched validation results", "", "| Model/objective | Mean Dice | SD | Runs |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {v['mean']:.4f} | {v['sample_std']:.4f} | {v['n']} |" for name, v in groups.items()]
    lines += ["", *summary["notes"], ""]
    (root / "summary.md").write_text("\n".join(lines))
    print(json.dumps(groups, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.allow_incomplete)
