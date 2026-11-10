"""Constraint campaign with cluster-only archives and Slurm continuation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import baseline_runtime as runtime

CAP, MODELS = runtime.CAP, runtime.MODELS
read, write, sha = runtime.read, runtime.write, runtime.sha
NAME = "hippo-geofwi-constraints"
ARMS = ("dice_bands", "dice_sdf", "dice_bands_sdf", "dice_bands_edge",
        "dice_bands_edge_sdf", "dice_depth", "dice_bands_depth", "dice_edge_depth",
        "dice_sdf_depth", "dice_bands_edge_depth", "dice_bands_sdf_depth",
        "dice_edge_sdf_depth", "dice_bands_edge_sdf_depth")
original_validate = runtime.validate_run


def key(t):
    return t["model"], t["fold"], t["seed"], t["arm"]


def validate_run(stage, task):
    row = original_validate(stage, task)
    out = stage / "runs" / task["id"]
    p = read(out / "protocol.json")
    done, cal = read(out / "completion.json"), read(out / "calibration.json")
    schedule = p["models"][p["trainer_alias"]]["objectives"][task["arm"]]
    assert done["objective"] == task["arm"] and done["calibration_attempts"] == 1
    assert done["calibration_flops"] == schedule["calibration_flops"]
    assert done["consumed_training_flops"] == schedule["consumed_training_flops"]
    assert [r["case_name"] for r in cal["cases"]] == read(stage/"splits.json")[task["fold"]]["train"]
    assert set(cal["weights"]) == set(task["arm"].split("_")[1:])
    assert all(math.isfinite(w) and w > 0 for w in cal["weights"].values())
    assert cal["target_ratio"] == p["calibration_target_ratio"] and cal["p95_cap"] == p["calibration_p95_cap"]
    return row


def verify(stage, data=True):
    for name, expected in read(stage / "manifest.json").items():
        assert sha(stage / name) == expected, name
    plan = read(stage / "plan.json")
    expected = {(m, f, s, a) for m in MODELS for f in range(3) for s in range(3) for a in ARMS}
    cells = plan["tasks"] + plan["reused_rows"]
    assert len(cells) == len(expected) == 468
    assert {key(t) for t in cells} == expected
    assert plan["cap"] == CAP and plan["workers"] == 2
    assert len(read(stage / "baseline_results.json")["rows"]) == 36
    splits = read(stage / "splits.json")
    assert len(splits) == 3 and len(set().union(*(set(s["val"]) for s in splits))) == 288
    for split in splits:
        assert len(set(split["train"])) == 19 and len(set(split["val"])) == 96
        assert not set(split["train"]) & set(split["val"])
    for t in plan["tasks"]:
        folder = stage / "protocols" / t["id"]
        p = read(folder / "protocol.json")
        assert p["objective"] == t["arm"] != "dice"
        for filename, field in (("dataset.json", "pkl_sha256"), ("splits_10patients.json", "splits_sha256"),
                                ("nnunet_architecture.json", "nnunet_architecture_sha256"),
                                ("profile.json", "profile_sha256")):
            assert sha(folder / filename) == p[field]
        assert p["splits_sha256"] == sha(stage / "splits.json")
        assert p["model_identity"] == t["model"] and p["seeds"] == [t["seed"]] and p["folds"] == [t["fold"]]
        assert p["budget_flops"] == CAP and p["batch_size"] == 1 and not p["early_stopping"]
        s = p["models"][p["trainer_alias"]]["objectives"][t["arm"]]
        assert s["consumed_total_flops"] <= CAP < s["consumed_total_flops"] + s["flops_per_update"]
        assert s["consumed_total_flops"] == s["consumed_training_flops"] + s["calibration_flops"]
        assert s["max_steps"] == t["updates"] and s["calibration_flops"] > 0
        assert s["prefix_updates"] + s["constraint_warmup_updates"] < s["max_steps"]
    if data:
        for name, expected in read(stage / "data_manifest.json").items():
            assert sha(stage / name) == expected, name
    return plan


class ClusterArchive:
    """Latest explicit user instruction: archive on cluster, no laptop gate."""
    def __init__(self, stage):
        self.stage = stage
        sys.path.insert(0, str(stage / "bundles/unetr/source"))
        from scripts.cross_dataset_campaign import storage
        self.original = storage
        self.verify_archive = storage.verify_archive

    def prepare(self, output, task):
        manifest = self.original.prepare(output, task)
        destination = self.stage / "archives" / task["id"]
        destination.mkdir(parents=True, exist_ok=True)
        for name, info in manifest["metadata"].items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == info["sha256"]
            else:
                temp = target.with_name(target.name + ".copying")
                shutil.copyfile(output / name, temp)
                with temp.open("rb") as stream:
                    os.fsync(stream.fileno())
                temp.replace(target)
        shutil.copyfile(output / "archive_manifest.json", destination / "archive_manifest.json")
        self.verify_archive(destination, manifest)
        write(output / "archive_acknowledgment.json", dict(
            status="verified", manifest_sha256=sha(output / "archive_manifest.json"),
            source_manifest_sha256=sha(self.stage / "manifest.json"), source_verified=True,
            durable_archive=str(destination), laptop_required=False,
            authorization="Explicit user request: keep results on cluster and run while laptop is off"))
        return manifest

    def retire(self, output):
        manifest = read(output / "archive_manifest.json")
        ack = read(output / "archive_acknowledgment.json")
        destination = Path(ack["durable_archive"])
        assert destination.parent.resolve() == (self.stage / "archives").resolve()
        assert ack["manifest_sha256"] == sha(output / "archive_manifest.json")
        self.verify_archive(destination, manifest)
        self.verify_archive(output, manifest)
        journal = output / "cleanup.json"
        for item in manifest["deletion_candidates"]:
            path = output / item["name"]
            assert path.parent.resolve() == output.resolve() and not path.is_symlink()
            if path.exists():
                assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
            else:
                assert journal.exists()
        receipt = dict(status="in_progress", deleted=manifest["deletion_candidates"],
                       durable_archive=str(destination), released_file_bytes=sum(
                           i["bytes"] for i in manifest["deletion_candidates"]), time=time.time())
        write(journal, receipt)
        for item in manifest["deletion_candidates"]:
            (output / item["name"]).unlink(missing_ok=True)
        receipt["status"] = "complete"
        write(journal, receipt)
        write(destination / "cleanup.json", receipt)
        return receipt


def summarize(stage, status):
    plan = read(stage / "plan.json")
    rows = list(plan["reused_rows"])
    for t in plan["tasks"]:
        out = stage / "runs" / t["id"]
        if (out / "cleanup.json").exists() and read(out / "cleanup.json")["status"] == "complete":
            rows.append(runtime.validate_run(stage, t))
    assert len({key(r) for r in rows}) == len(rows)
    assert all(math.isfinite(r["final_dice"]) for r in rows)
    baselines = {(r["model"], r["fold"], r["seed"]): r for r in read(stage / "baseline_results.json")["rows"]}
    for r in rows:
        r["baseline_dice"] = baselines[key(r)[:3]]["final_dice"]
        r["delta_dice_percentage_points"] = 100 * (r["final_dice"] - r["baseline_dice"])
    complete = len(rows) == 468
    doc = dict(status="complete" if complete else status, rows=rows, required_cells=468,
               complete_matrix=complete, all_above_50=complete and all(r["final_dice"] > .5 for r in rows),
               baselines_reused=36, new_baselines=0, laptop_required=False, updated_at=time.time())
    write(stage / "status.json", doc)
    if complete:
        write(stage / "FINISHED.json", doc)
    return doc


def submit(stage, continuation=False):
    verify(stage)
    assert read(stage / "cpu_contract_checks.json")["status"] == "passed"
    for child in ("unetr", "basic"):
        assert read(stage / f"cpu_audit_{child}.json")["status"] == "passed"
    predecessor = os.environ.get("SLURM_JOB_ID") if continuation else None
    token = predecessor or "initial"
    folder = stage / "submissions" / token
    folder.mkdir(parents=True, exist_ok=True)
    with (stage / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = folder / "receipt.json"
        if receipt.exists():
            return read(receipt)
        if (folder / "intent.json").exists():
            raise RuntimeError("Unknown submission outcome; reconcile queue/accounting before retry")
        rows = [line.split("|") for line in subprocess.check_output(
            ["squeue", "-h", "-u", str(os.getuid()), "-o", "%i|%T|%j"], text=True).splitlines()]
        assert not any(r[1] == "PENDING" for r in rows), rows
        assert len(rows) <= 1 and all(r[1] in ("RUNNING", "COMPLETING") for r in rows), rows
        if continuation:
            assert predecessor and read(stage / "status.json")["status"] == "needs_resume"
            assert not rows or rows[0][0] == predecessor, rows
        usage = runtime.quota()
        assert usage["free_gib"] >= 24, usage
        cmd = ["sbatch", "--parsable", f"--chdir={stage}",
               f"--output={stage}/slurm-%j.out", f"--error={stage}/slurm-%j.err"]
        if rows:
            cmd.append("--dependency=afterany:" + rows[0][0])
        cmd += [str(stage / "run.sbatch"), str(stage)]
        write(folder / "intent.json", dict(command=cmd, time=time.time()))
        job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        assert job.isdigit(), job
        result = dict(status="submitted", job_id=job, predecessor=predecessor, command=cmd,
                      quota=usage, manifest_sha256=sha(stage / "manifest.json"), time=time.time())
        write(receipt, result)
        write(stage / "submission.json", result)
        return result


def run(stage):
    # No new task starts after 22h; normal cells take minutes. Slurm's signal
    # remains a checkpoint fallback. The next allocation is submitted here.
    job = os.environ["SLURM_JOB_ID"]
    continuation_probe = subprocess.run(
        ["sbatch", "--test-only", "--dependency=afterany:" + job, str(stage/"run.sbatch"), str(stage)],
        text=True, capture_output=True)
    write(stage/"continuation_preflight"/f"{job}.json",
          dict(status="passed" if continuation_probe.returncode==0 else "failed",
               exit_code=continuation_probe.returncode, stdout=continuation_probe.stdout,
               stderr=continuation_probe.stderr, submitted_job=False))
    if continuation_probe.returncode:
        raise RuntimeError("Slurm continuation unavailable from the compute node")
    os.environ["HIPPO_ALLOCATION_DEADLINE"] = str(time.time() + 22 * 3600)
    code = runtime.run(stage)
    # srun can report 75 when one lane pauses and the other fails with 1.
    # A technical failure must never be disguised as a clean continuation.
    if list(stage.glob(f"failure_{job}_*.json")):
        summarize(stage, "failed")
        return 1
    if code == 75 and not (stage / "STOP").exists():
        submit(stage, continuation=True)
        return 0
    return code


runtime.verify = verify
runtime.validate_run = validate_run
runtime.summarize = summarize
runtime.storage = ClusterArchive


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("verify", "submit", "run", "lane", "status"))
    ap.add_argument("--stage", type=Path, required=True)
    a = ap.parse_args()
    stage = a.stage.resolve()
    try:
        if a.command == "verify":
            print(json.dumps(dict(status="passed", tasks=len(verify(stage)["tasks"]))))
        elif a.command == "submit":
            print(json.dumps(submit(stage)))
        elif a.command == "status":
            print(json.dumps(summarize(stage, "inspected")))
        elif a.command == "lane":
            sys.exit(runtime.lane(stage))
        else:
            sys.exit(run(stage))
    except Exception:
        write(stage / f"failure_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_PROCID', 'main')}.json",
              dict(status="failed", traceback=traceback.format_exc(), time=time.time()))
        raise
