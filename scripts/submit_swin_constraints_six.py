"""Verify the frozen six-arm SwinUNETR payload and submit it once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.getuid() == 1395
    stage = Path(__file__).resolve().parents[1]
    receipt = stage / "submission.json"
    with (stage / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            print(receipt.read_text())
            return
        for name, sha in json.loads((stage / "payload_manifest.json").read_text()).items():
            assert digest(stage / name) == sha, name
        for name, sha in json.loads((stage / "runtime_origin_manifest.json").read_text()).items():
            assert digest(stage / "frozen_runtime" / name) == sha, name
        directory = stage / "experiments/compute_matched_swin_six_20261004"
        protocol_path = directory / "protocol.json"
        p = json.loads(protocol_path.read_text())
        arms = ["dice", "dice_bands", "dice_sdf", "dice_bands_sdf", "dice_bands_edge", "dice_bands_edge_sdf"]
        assert p["budget_flops"] == 12207124453303 and p["folds"] == p["seeds"] == [0]
        assert list(p["models"]) == ["swinunetr"]
        assert list(p["models"]["swinunetr"]["objectives"]) == arms
        assert p["constraint_warmup_updates"] == 3
        for arm, schedule in p["models"]["swinunetr"]["objectives"].items():
            prefix = 0 if arm == "dice" else 3
            assert schedule["prefix_updates"] == prefix
            cost = (prefix * schedule["baseline_flops_per_update"] + schedule["calibration_flops"]
                    + (schedule["max_steps"] - prefix) * schedule["flops_per_update"])
            assert cost == schedule["consumed_flops"] <= p["budget_flops"] < cost + schedule["flops_per_update"]
        check = json.loads((stage / "verification.json").read_text())
        assert check["preflight_passed"] and check["protocol_sha256"] == digest(protocol_path)
        data = Path("/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl")
        assert digest(data) == p["pkl_sha256"]
        quota = subprocess.run(["beegfs", "quota", "list-usage", "--uids", "current", "--pool", "pool_home"],
                               capture_output=True, text=True, check=True).stdout
        limits = re.search(r"(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB", quota)
        if not limits or float(limits[2]) - float(limits[1]) < 5:
            raise RuntimeError("Insufficient free quota: " + quota)
        queue = subprocess.run(["squeue", "-u", "3160552", "-h", "-o", "%i|%T|%j"],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        states = [line.split("|")[1] for line in queue if line.strip()]
        if states.count("PENDING") >= 1 or len(states) > 1:
            raise RuntimeError("One-running/one-queued limit: " + repr(queue))
        sbatch = stage / "scripts/run_swin_constraints_six.sbatch"
        response = subprocess.run(["sbatch", "--parsable", str(sbatch)], cwd=stage,
                                  capture_output=True, text=True, check=True)
        job = int(response.stdout.strip().split(";")[0])
        record = {"job_id": job, "submitted_at_unix": time.time(), "stage": str(stage),
                  "run_root": f"/mnt/beegfsstudents/home/3160552/compute_matched_swin_six_{job}",
                  "protocol_sha256": digest(protocol_path), "sbatch_sha256": digest(sbatch),
                  "budget_flops_per_run": p["budget_flops"], "models": ["swinunetr"],
                  "folds": [0], "seeds": [0], "objectives": arms,
                  "queue_before_submission": queue, "quota_before_submission": quota}
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(receipt)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
