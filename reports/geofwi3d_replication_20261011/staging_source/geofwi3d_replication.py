"""Frozen, quota-aware three-fold/three-seed replication of the GeoFWI3D pilot."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import getpass
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from geofwi3d_data import FRACTIONS, sha256, validate_splits, write_json

METHODS = ("baseline", "bands", "sdf", "bands_sdf", "bands_edge", "bands_sdf_edge")
GIB = 1024**3
RETAINED = ("run_config.json", "history.csv", "validation_best.json", "validation_per_image.csv")


def fold_splits(manifest: dict) -> list[dict]:
    """Keep the pilot fold exactly; partition its complement into four strata-balanced folds."""
    cases = manifest["cases"]
    original = manifest["splits"]
    validate_splits(original, cases)
    rng = np.random.default_rng(0)
    validation = [list(original["validation"])] + [[] for _ in range(4)]
    for fault in (0, 1):
        ids = sorted(k for k in original["train_pool"] if cases[k]["fault_present"] == fault)
        rng.shuffle(ids)
        for fold, group in enumerate(np.array_split(ids, 4), 1):
            validation[fold].extend(group.tolist())
    if sum(map(len, validation)) != len(cases) or len(set().union(*map(set, validation))) != len(cases):
        raise ValueError("Validation partitions are not disjoint and complete")
    splits = [copy.deepcopy(original)]
    for fold in range(1, 5):
        held_out = set(validation[fold])
        rng = np.random.default_rng(np.random.SeedSequence([0, fold, 32091]))
        groups = []
        for fault in (0, 1):
            group = sorted(k for k in cases if k not in held_out and cases[k]["fault_present"] == fault)
            rng.shuffle(group)
            groups.append(group)
        order, used = [], [0, 0]
        total = sum(map(len, groups))
        for position in range(total):
            available = [i for i in range(2) if used[i] < len(groups[i])]
            i = max(available, key=lambda i: (position + 1) * len(groups[i]) / total - used[i])
            order.append(groups[i][used[i]])
            used[i] += 1
        current = copy.deepcopy(original)
        current.update(train_pool=order, validation=sorted(held_out),
                       training_subsets={name: order[:max(1, int(math.floor(total * fraction + .5)))]
                                         for name, fraction in FRACTIONS.items()})
        validate_splits(current, cases)
        splits.append(current)
    return splits


def quota_available(output: str) -> tuple[float, float | None]:
    """Use the tightest finite space and inode allowance across applicable rows."""
    remaining, inodes = [], []
    units = {"KiB": 1024, "MiB": 1024**2, "GiB": GIB, "TiB": 1024**4, "B": 1}
    found_user = False
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[2] not in {"user", "group"}:
            continue
        found_user |= fields[2] == "user"
        match = re.fullmatch(r"([\d.]+)(KiB|MiB|GiB|TiB|B)/(.+)", fields[4])
        if not match:
            raise RuntimeError("Cannot parse applicable BeeGFS space quota")
        if match[3] not in {"∞", "unlimited"}:
            limit = re.fullmatch(r"([\d.]+)(KiB|MiB|GiB|TiB|B)", match[3])
            if not limit:
                raise RuntimeError("Cannot parse BeeGFS quota limit")
            remaining.append(float(limit[1]) * units[limit[2]] - float(match[1]) * units[match[2]])
        inode_match = re.fullmatch(r"([\d.]+)([kM]?)/(.+)", fields[5])
        if not inode_match:
            raise RuntimeError("Cannot parse applicable inode quota")
        if inode_match[3] not in {"∞", "unlimited"}:
            limit = re.fullmatch(r"([\d.]+)([kM]?)", inode_match[3])
            if not limit:
                raise RuntimeError("Cannot parse inode quota limit")
            scale = {"": 1, "k": 1000, "M": 1000000}
            inodes.append(float(limit[1]) * scale[limit[2]] - float(inode_match[1]) * scale[inode_match[2]])
    if not found_user or not remaining:
        raise RuntimeError("Cannot establish finite user quota")
    return min(remaining), min(inodes) if inodes else None


def space_check(campaign: Path, phase: str) -> dict:
    raw = subprocess.check_output(["beegfs", "quota", "list-usage", "--uids", "current",
                                   "--gids", "current", "--pool", "pool_home"], text=True)
    free, inodes = quota_available(raw)
    minimum = 1.25 * GIB
    record = {"phase": phase, "checked_at_unix": time.time(), "quota_raw": raw,
              "free_quota_gib": free / GIB, "minimum_free_gib": minimum / GIB,
              "remaining_inodes": inodes, "filesystem_free_gib": shutil.disk_usage(campaign).free / GIB,
              "retention": "JSON/CSV only; complete checkpoint verification on node-local scratch"}
    write_json(campaign / "quota_checks" / f"{phase}.json", record)
    if free < minimum or record["filesystem_free_gib"] < 1.25 or (inodes is not None and inodes < 2000):
        raise RuntimeError("Insufficient BeeGFS reserve; no submission/training permitted")
    return record


def verify_run(directory: Path, manifest: dict, method: str, seed: int, checkpoints: bool) -> dict:
    completion = json.loads((directory / "completion.json").read_text())
    if completion["state"] != "complete" or completion["method"] != method or completion["seed"] != seed:
        raise RuntimeError("Unexpected completion identity")
    for filename, expected in completion["artifact_sha256"].items():
        if filename.endswith(".pt") and not checkpoints:
            continue
        if sha256(directory / filename) != expected:
            raise RuntimeError(f"Changed artifact: {directory / filename}")
    config = json.loads((directory / "run_config.json").read_text())
    if config["training_case_ids"] != manifest["splits"]["training_subsets"]["5pct"]:
        raise RuntimeError("Training IDs differ from frozen fold")
    if config["validation_case_ids"] != manifest["splits"]["validation"]:
        raise RuntimeError("Validation IDs differ from frozen fold")
    with (directory / "validation_per_image.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = config["validation_case_ids"]
    if len(rows) != len(ids) or {r["case_id"] for r in rows} != set(ids):
        raise RuntimeError("Missing or duplicated validation case")
    for row in rows:
        score = 2 * int(row["intersection"]) / (int(row["predicted_voxels"]) + int(row["target_voxels"]))
        if not math.isclose(score, float(row["dice"]), abs_tol=1e-12):
            raise RuntimeError("Dice does not match voxel counts")
    if not math.isclose(statistics.mean(float(r["dice"]) for r in rows),
                        completion["validation"]["mean_dice"], abs_tol=1e-12):
        raise RuntimeError("Mean Dice does not match per-case records")
    if completion["stopped_update"] != 500 or config["protocol"]["model_seed"] != seed:
        raise RuntimeError("Run does not match the 500-update/seed policy")
    return completion


def verify(campaign: Path, dataset: bool = True) -> dict:
    frozen = json.loads((campaign / "source_manifest.json").read_text())
    for name, expected in frozen["sha256"].items():
        if sha256(campaign / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    plan = json.loads((campaign / "plan.json").read_text())
    for name, expected in plan["dependencies"].items():
        if version(name) != expected:
            raise RuntimeError(f"Framework version changed: {name}")
    root = Path(plan["dataset_root"])
    original = json.loads((root / "manifest.json").read_text())
    if sha256(root / "manifest.json") != plan["original_manifest_sha256"]:
        raise RuntimeError("Original dataset manifest changed")
    if dataset:
        for case in original["cases"].values():
            for kind in ("image", "label"):
                if sha256(root / case[kind]) != case[kind + "_sha256"]:
                    raise RuntimeError("Shared dataset file changed")
    manifests = []
    for fold in range(3):
        view = campaign / "folds" / f"fold_{fold}"
        manifest = json.loads((view / "manifest.json").read_text())
        validate_splits(manifest["splits"], manifest["cases"])
        if manifest["cases"] != original["cases"]:
            raise RuntimeError("Fold changed underlying images/labels")
        for name in ("imagesTr", "labelsTr", "imagesVal", "labelsVal"):
            if (view / name).resolve() != (root / name).resolve():
                raise RuntimeError("Fold view is not linked to the verified shared dataset")
        manifests.append(manifest)
    if manifests[0] != original:
        raise RuntimeError("Pilot fold changed")
    for i in range(3):
        for j in range(i):
            if set(manifests[i]["splits"]["validation"]) & set(manifests[j]["splits"]["validation"]):
                raise RuntimeError("Validation folds overlap")
    return plan


def prepare(args) -> None:
    campaign, root = args.campaign.resolve(), args.dataset_root.resolve()
    if campaign.exists():
        raise RuntimeError("Use a new campaign directory")
    reference = args.reference_four.resolve()
    source = reference / "source"
    frozen = json.loads((reference / "source_manifest.json").read_text())
    for name, expected in frozen["sha256"].items():
        if sha256(reference / name) != expected:
            raise RuntimeError("Pilot frozen input changed")
    original = json.loads((root / "manifest.json").read_text())
    if sha256(root / "manifest.json") != sha256(reference / "manifest.json"):
        raise RuntimeError("Dataset differs from pilot")
    protocol = json.loads((reference / "protocol.json").read_text())
    if protocol["max_updates"] != 500 or protocol["scheduler_updates"] != 5000:
        raise RuntimeError("Unexpected reference budget")
    deps = json.loads((reference / "dependencies.json").read_text())
    for name, expected in deps.items():
        if version(name) != expected:
            raise RuntimeError("Pilot environment changed")
    splits = fold_splits(original)
    campaign.mkdir(parents=True)
    runtime = campaign / "source"
    runtime.mkdir()
    for name in ("train_geofwi3d_swinunetr.py", "geofwi3d_data.py", "geofwi3d_losses.py"):
        shutil.copy2(source / name, runtime / name)
    for name in ("geofwi3d_replication.py", "run_geofwi3d_replication.sbatch"):
        shutil.copy2(Path(__file__).with_name(name), runtime / name)
    for fold in range(3):
        view = campaign / "folds" / f"fold_{fold}"
        view.mkdir(parents=True)
        manifest = copy.deepcopy(original)
        manifest["splits"] = splits[fold]
        if fold == 0:
            shutil.copy2(root / "manifest.json", view / "manifest.json")
        else:
            write_json(view / "manifest.json", manifest)
        for name in ("imagesTr", "labelsTr", "imagesVal", "labelsVal"):
            (view / name).symlink_to(root / name, target_is_directory=True)
    write_json(campaign / "validation_partitions.json", {
        "policy": "Five disjoint stratified partitions, anchored on original validation; use folds 0,1,2",
        "validation": {str(i): s["validation"] for i, s in enumerate(splits)}})
    locations = {"baseline": args.reference_baseline.resolve() / "results/5pct_baseline",
                 "bands": args.reference_bands.resolve() / "results/5pct_bands"}
    locations.update({method: reference / "results" / f"5pct_{method}" for method in METHODS[2:]})
    refs = []
    for method in METHODS:
        old = locations[method]
        rec = verify_run(old, original, method, 0, checkpoints=True)
        old_protocol = json.loads((old / "run_config.json").read_text())["protocol"]
        for key, value in old_protocol.items():
            if key != "methods" and protocol[key] != value:
                raise RuntimeError(f"Control recipe differs: {method}, {key}")
        dest = campaign / "references" / method
        dest.mkdir(parents=True)
        for filename in (*RETAINED, "completion.json"):
            shutil.copy2(old / filename, dest / filename)
        refs.append({"fold": 0, "seed": 0, "method": method, "directory": str(old),
                     "frozen_record_directory": str(dest.relative_to(campaign)),
                     "mean_dice": rec["validation"]["mean_dice"]})
    protocol["methods"] = list(METHODS)
    write_json(campaign / "protocol.json", protocol)
    tasks = [{"fold": fold, "seed": seed, "method": method,
              "run_id": f"fold{fold}_seed{seed}_{method}"}
             for fold in range(3) for seed in range(3) for method in METHODS
             if (fold, seed) != (0, 0)]
    plan = {"dataset_root": str(root), "original_manifest_sha256": sha256(root / "manifest.json"),
            "dependencies": deps, "python": sys.executable, "methods": list(METHODS),
            "folds": [0, 1, 2], "seeds": [0, 1, 2], "new_runs": tasks,
            "references": refs, "fold_training_subset_policy": "fixed 19-case subset per fold across seeds/methods",
            "checkpoint_retention": "temporary full best/latest on node-local scratch; archive JSON/CSV after all hashes and Dice verified; discard new checkpoints",
            "estimated_retained_logical_mib": 50, "mirrored_quota_allowance_mib": 100,
            "minimum_free_quota_gib": 1.25, "minimum_scratch_gib": 4,
            "notes": "Original fold and budget are exploratory/validation-informed. No test scores. No loss or budget retuning."}
    write_json(campaign / "plan.json", plan)
    files = [p for p in campaign.rglob("*") if p.is_file() and not p.is_symlink()]
    write_json(campaign / "source_manifest.json", {"sha256": {
        str(p.relative_to(campaign)): sha256(p) for p in sorted(files)}})
    verify(campaign)
    print(json.dumps({"prepared": str(campaign), "new_runs": len(tasks),
                      "reused_runs": len(refs), "fold_sizes": [(len(s["training_subsets"]["5pct"]), len(s["validation"])) for s in splits[:3]]}))


def aggregate(campaign: Path, complete: bool) -> dict:
    plan = json.loads((campaign / "plan.json").read_text())
    entries = []
    for reference in plan["references"]:
        directory = campaign / reference["frozen_record_directory"]
        rec = json.loads((directory / "completion.json").read_text())
        entries.append({**{k: reference[k] for k in ("fold", "seed", "method")},
                        "mean_dice": rec["validation"]["mean_dice"], "origin": "reused pilot"})
    for task in plan["new_runs"]:
        directory = campaign / "results" / task["run_id"]
        if not (directory / "completion.json").exists():
            continue
        manifest = json.loads((campaign / "folds" / f"fold_{task['fold']}" / "manifest.json").read_text())
        rec = verify_run(directory, manifest, task["method"], task["seed"], checkpoints=False)
        entries.append({**{k: task[k] for k in ("fold", "seed", "method")},
                        "mean_dice": rec["validation"]["mean_dice"], "origin": "new run"})
    if complete and len(entries) != 54:
        raise RuntimeError("Campaign completion requires all 54 fold/seed/method cells")
    cells = {(r["fold"], r["seed"], r["method"]): r["mean_dice"] for r in entries}
    for row in entries:
        base = cells.get((row["fold"], row["seed"], "baseline"))
        row["gain_vs_baseline_pp"] = 100 * (row["mean_dice"] - base) if base is not None else None
    overall, by_fold = {}, {}
    for method in METHODS:
        selected = [r for r in entries if r["method"] == method]
        scores = [r["mean_dice"] for r in selected]
        gains = [r["gain_vs_baseline_pp"] for r in selected if r["gain_vs_baseline_pp"] is not None]
        overall[method] = {"cells": len(scores), "mean_dice": statistics.mean(scores),
                           "sample_std_dice": statistics.stdev(scores) if len(scores) > 1 else None,
                           "mean_paired_gain_pp": statistics.mean(gains) if gains else None,
                           "sample_std_paired_gain_pp": statistics.stdev(gains) if len(gains) > 1 else None,
                           "note": "Descriptive fold/seed variation; overlapping training pools, not nine independent datasets"}
        for fold in range(3):
            values = [r["mean_dice"] for r in selected if r["fold"] == fold]
            paired = [r["gain_vs_baseline_pp"] for r in selected
                      if r["fold"] == fold and r["gain_vs_baseline_pp"] is not None]
            if values:
                by_fold[f"fold{fold}_{method}"] = {"seeds": len(values), "mean_dice": statistics.mean(values),
                                                   "sample_std_dice": statistics.stdev(values) if len(values) > 1 else None,
                                                   "mean_paired_gain_pp": statistics.mean(paired) if paired else None,
                                                   "sample_std_paired_gain_pp": statistics.stdev(paired) if len(paired) > 1 else None}
    results = campaign / "results"
    results.mkdir(exist_ok=True)
    with (results / "per_run_dice.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    record = {"state": "complete" if complete else "running", "cells_completed": len(entries),
              "new_runs_completed": len(entries) - 6, "expected_cells": 54,
              "overall": overall, "per_fold_three_seeds": by_fold, "runs": entries,
              "scope": "selected validation Dice; no test evaluation"}
    write_json(results / "summary.json", record)
    if complete:
        lines = ["# GeoFWI3D three-fold / three-seed replication", "",
                 "54 matched cells: six reused pilot results plus 48 new runs. Each run uses 19 training volumes and 500 updates. Validation folds contain 96 volumes each.", "",
                 "| Added objective | Mean validation Dice | SD over 9 fold/seed cells | Paired gain vs baseline |",
                 "|---|---:|---:|---:|"]
        for method, metrics in overall.items():
            lines.append(f"| {method} | {100*metrics['mean_dice']:.2f}% | {100*metrics['sample_std_dice']:.2f} pp | {metrics['mean_paired_gain_pp']:+.2f} pp |")
        lines += ["", "All checkpoints were verified before exporting results. New checkpoints were temporary node-local artifacts and are not retained. Original pilot checkpoints are unchanged.",
                  "", "These are selected validation results. The original budget was informed by pilot validation. The SD describes variation, not a confidence interval; training pools overlap across folds. Boundary-specific metrics and test performance are not claimed.",
                  "", "See summary.json for each fold's three-seed mean/SD, and per_run_dice.csv for every matched result.", ""]
        (results / "RESULTS.md").write_text("\n".join(lines))
    return record


def run(campaign: Path) -> None:
    plan = verify(campaign)
    import torch
    from train_geofwi3d_swinunetr import checked_protocol, train_run
    results = campaign / "results"
    results.mkdir(exist_ok=True)
    active = None
    scratch = None
    try:
        space_check(campaign, "allocation_start")
        parent = Path(tempfile.gettempdir()).resolve()
        if parent.stat().st_dev == campaign.stat().st_dev or shutil.disk_usage(parent).free < 4 * GIB:
            raise RuntimeError("Need at least 4 GiB node-local scratch outside BeeGFS")
        scratch = Path(tempfile.mkdtemp(prefix="geofwi3d_replication_", dir=parent))
        write_json(campaign / "scratch_check.json", {"path": str(scratch), "device": scratch.stat().st_dev,
                   "beegfs_device": campaign.stat().st_dev, "free_gib": shutil.disk_usage(scratch).free / GIB,
                   "minimum_gib": 4, "host": __import__("socket").gethostname()})
        protocol = json.loads((campaign / "protocol.json").read_text())
        checked_protocol(protocol)
        torch.set_num_threads(protocol["cpu_threads"])
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This matched campaign requires CUDA bfloat16")
        device = torch.device("cuda")
        for task in plan["new_runs"]:
            active = task
            view = campaign / "folds" / f"fold_{task['fold']}"
            manifest = json.loads((view / "manifest.json").read_text())
            target = results / task["run_id"]
            if (target / "completion.json").exists():
                verify_run(target, manifest, task["method"], task["seed"], checkpoints=False)
                continue
            space_check(campaign, task["run_id"])
            if shutil.disk_usage(scratch).free < 4 * GIB:
                raise RuntimeError("Insufficient scratch reserve before next run")
            write_json(results / "status.json", {"state": "running", "active": task,
                       "scratch_run": str(scratch / task["run_id"]),
                       "completed_new_runs": sum((results / r["run_id"] / "completion.json").exists() for r in plan["new_runs"])})
            current = copy.deepcopy(protocol)
            current["model_seed"] = task["seed"]
            temporary = scratch / task["run_id"]
            train_run(view, manifest, current, temporary, "5pct", task["method"], device)
            verify_run(temporary, manifest, task["method"], task["seed"], checkpoints=True)
            target.mkdir(exist_ok=True)
            for filename in RETAINED:
                shutil.copy2(temporary / filename, target / filename)
            write_json(target / "retention.json", {
                "all_completion_hashes_verified_before_export": True,
                "checkpoint_retention": "discard after successful best-checkpoint Dice reproduction and full artifact hash verification",
                "best_pt_sha256": sha256(temporary / "best.pt"), "latest_pt_sha256": sha256(temporary / "latest.pt"),
                "scratch_host": __import__("socket").gethostname()})
            # The completion marker is exported last; only complete archives are resumed/skipped.
            shutil.copy2(temporary / "completion.json", target / "completion.json")
            verify_run(target, manifest, task["method"], task["seed"], checkpoints=False)
            shutil.rmtree(temporary)  # Only this new run's private temporary directory.
            aggregate(campaign, complete=False)
        final = aggregate(campaign, complete=True)
        write_json(results / "status.json", {"state": "complete", "new_runs_completed": 48, "total_cells": 54})
        space_check(campaign, "allocation_end")
        shutil.rmtree(scratch)
        print(json.dumps({"state": "complete", "overall": final["overall"]}), flush=True)
    except Exception as error:
        record = {"state": "failed", "error": str(error), "active": active,
                  "scratch": str(scratch), "traceback": traceback.format_exc()}
        print(json.dumps(record), file=sys.stderr, flush=True)
        write_json(results / "failure.json", record)
        raise


def submit(campaign: Path) -> None:
    plan = verify(campaign)
    with (campaign / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (campaign / "submission.json").exists():
            print((campaign / "submission.json").read_text())
            return
        quota = space_check(campaign, "pre_submit")
        raw = subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%T|%j"], text=True).strip()
        rows = [r.split("|", 2) for r in raw.splitlines()]
        running = [r for r in rows if r[1] in {"RUNNING", "COMPLETING", "CONFIGURING"}]
        if len(running) > 1 or len(rows) != len(running):
            write_json(campaign / "queue_blocked.json", {"queue": rows, "submitted": False})
            raise RuntimeError("No queued slot under one-running/one-queued policy")
        args = ["sbatch", "--parsable", f"--output={campaign}/slurm-%j.out", f"--error={campaign}/slurm-%j.err"]
        dependency = "afterany:" + running[0][0] if running else None
        if dependency:
            if not running[0][0].isdigit():
                raise RuntimeError("Unexpected dependency identity")
            args.append("--dependency=" + dependency)
        args += [str(campaign / "source/run_geofwi3d_replication.sbatch"), str(campaign), plan["python"]]
        job = subprocess.check_output(args, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Unexpected sbatch response")
        receipt = {"submitted": True, "job_id": int(job), "queue_before": rows, "dependency": dependency,
                   "single_sbatch_job": True, "new_runs": 48, "reused_runs": 6,
                   "submitted_at_unix": time.time(), "quota_before": quota, "scheduler_arguments": args}
        write_json(campaign / "submission.json", receipt)
        print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--dataset-root", type=Path, required=True)
    for name in ("reference-four", "reference-baseline", "reference-bands"):
        preparation.add_argument("--" + name, type=Path, required=True)
    for command in ("prepare", "verify", "run", "submit"):
        sub = preparation if command == "prepare" else commands.add_parser(command)
        sub.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "verify":
        print(json.dumps({"verified": True, "new_runs": len(verify(args.campaign)["new_runs"])}))
    elif args.command == "run":
        run(args.campaign.resolve())
    else:
        submit(args.campaign.resolve())


if __name__ == "__main__":
    main()
