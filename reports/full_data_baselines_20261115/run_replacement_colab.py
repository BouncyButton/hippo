"""Run the missing evaluations as foreground notebook work on a replacement VM."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from continue_colab import FOLDER, session
from prepare_colab_continuation import prepare_weights


def saved_keys(drive, dataset):
    prefix = dataset + "_"
    result = set()
    token = None
    while True:
        response = drive.files().list(
            q=f"'{FOLDER}' in parents and trashed=false", pageSize=1000, pageToken=token,
            fields="nextPageToken,incompleteSearch,files(name)",
        ).execute(num_retries=5)
        if response.get("incompleteSearch"):
            raise RuntimeError("Drive returned an incomplete result listing")
        for file in response["files"]:
            name = file["name"]
            if name.startswith(prefix) and name.endswith("_results.zip"):
                result.add(name.removesuffix("_results.zip"))
        token = response.get("nextPageToken")
        if not token:
            return result


def run(dataset):
    base = Path("/content")
    root = base / ("full_data_" + dataset)
    medical = base / ("full_data_" + dataset + "_medical")
    with (base / (dataset + "_replacement.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The setup command validates both input ZIPs before reporting readiness.
        setup_log = base / "prepare_replacement.log"
        while not setup_log.exists() or "READY_FOR_SCRATCH" not in setup_log.read_text(errors="replace"):
            time.sleep(10)
        drive = session()
        prepare_weights(dataset)
        expected = set(json.loads((root / "cells.json").read_text()))
        expected.update(p.stem for p in (medical / "protocols").glob("*.json"))
        assert len(expected) == 45
        # The original pilot evidence is handled separately from interrupted work.
        expected.remove(dataset + "_nnunet_f0_s0")
        for workers in (2, 1):
            missing = expected - saved_keys(drive, dataset)
            if not missing:
                break
            print(f"{dataset}: continuing {len(missing)} missing evaluations", flush=True)
            subprocess.run([sys.executable, "-u", str(base / "continue_colab.py"),
                            dataset, "--hold-pilot", "--scratch-workers", str(workers)], check=True)
        missing = expected - saved_keys(drive, dataset)
        if missing:
            raise RuntimeError(f"{dataset} needs inspection for {sorted(missing)}; see continuation.jsonl")
        print(f"{dataset}: all continuation archives saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ADNI","HFH","MSD"])
    run(parser.parse_args().dataset)
