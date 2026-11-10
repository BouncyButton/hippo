"""Verify a frozen GeoFWI3D campaign; submit only explicitly and with an empty queue."""
from __future__ import annotations

import argparse
import fcntl
import getpass
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

from geofwi3d_data import sha256, validate_splits, write_json


def verify(campaign: Path) -> dict:
    sources = json.loads((campaign / "source_manifest.json").read_text())
    for name, expected in sources["sha256"].items():
        if sha256(campaign / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    for name, expected in json.loads((campaign / "dependencies.json").read_text()).items():
        if version(name) != expected:
            raise RuntimeError(f"Dependency changed: {name}")
    config = json.loads((campaign / "campaign.json").read_text())
    root = Path(config["dataset_root"])
    manifest = json.loads((campaign / "manifest.json").read_text())
    if sha256(root / "manifest.json") != sha256(campaign / "manifest.json"):
        raise RuntimeError("Dataset manifest changed after campaign freeze")
    validate_splits(manifest["splits"], manifest["cases"])
    preflight = json.loads((campaign / "preflight.json").read_text())
    if preflight["state"] != "passed" or preflight["manifest_sha256"] != sha256(root / "manifest.json"):
        raise RuntimeError("Preflight is missing or stale")
    for case in manifest["cases"].values():
        for field in ["image", "label"]:
            if sha256(root / case[field]) != case[field + "_sha256"]:
                raise RuntimeError(f"Dataset file changed: {case[field]}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--submit", action="store_true", help="Actually submit; otherwise verification only")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    config = verify(campaign)
    if not args.submit:
        print(json.dumps({"state": "verified", "submitted": False, "campaign": str(campaign)}))
        return
    with (campaign / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (campaign / "submission.json").exists():
            print((campaign / "submission.json").read_text())
            return
        queue = subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %j %T %R"], text=True).strip()
        write_json(campaign / "queue_check.json", {"checked_at_unix": time.time(), "queue": queue})
        if queue:
            print(json.dumps({"submitted": False, "reason": "user queue is not empty", "queue": queue}))
            return
        if not Path(config["python"]).is_file():
            raise RuntimeError("Frozen Python interpreter is unavailable on this host")
        response = subprocess.check_output(["sbatch", "--parsable", f"--output={campaign}/slurm-%j.out",
                    f"--error={campaign}/slurm-%j.err", str(campaign / "source/run_geofwi3d_swinunetr.sbatch"),
                    str(campaign), config["python"]], text=True).strip()
        job_id = response.split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected scheduler response: {response}")
        result = {"submitted": True, "job_id": int(job_id), "queue_was_empty": True,
                  "submitted_at_unix": time.time(), "campaign": str(campaign)}
        write_json(campaign / "submission.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
