"""Recover per-case diagnostics in a separately charged, independently labeled run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from continue_colab import archive, existing, publish, session


def run(ds):
    key = f"{ds}_nnunet_f0_s0"
    repeat = key + "_recovery_replicate"
    bundle = Path("/content") / ("full_data_" + ds)
    root = Path("/content/pilot_replicates") / ds
    out = root / "work" / key
    out.mkdir(parents=True, exist_ok=True)
    with (root / "replicate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        drive = session()
        for name in (key + "_results.zip", repeat + "_results.zip"):
            if existing(drive, name):
                print("Already saved", name, flush=True)
                return
        original = json.loads((Path(__file__).with_name("original_colab_pilot_receipts.json")).read_text())["runs"][key]
        assert original["target_updates"] == 90 and original["last_verified_update"] == 90
        for name in ("cells.json", "protocols"):
            local = root / name
            if not local.exists():
                local.symlink_to(bundle / name)
        record = dict(key=key, replicate=repeat, purpose="separately charged per-case recovery replicate",
                      original_validation_dice=original["final_dice"], original_zip_sha256=original["runtime_archive_sha256"],
                      original_runtime=original["job_or_runtime"], original_updates=90, replicate_target_updates=90,
                      runtime=socket.gethostname())
        metadata = out / "replicate.json"
        if metadata.exists():
            assert json.loads(metadata.read_text()) == record
        else:
            metadata.write_text(json.dumps(record, indent=2) + "\n")
        source = bundle / "source"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [
            bundle / "compatibility", source, source / "scripts", source / "baselines/nnUNet"])),
            PYTHONUNBUFFERED="1", nnUNet_compile="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
            WANDB_MODE="disabled")
        if not (out / "completion.json").exists():
            print("Starting", repeat, flush=True)
            with (out / "training.log").open("a") as log:
                subprocess.run([str(bundle / "venv/bin/python"), str(bundle / "worker.py"), key,
                                "--bundle", str(bundle), "--work", str(root / "work")],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result, dice, coverage = archive(key, bundle, root, False)
        path = root / (repeat + "_results.zip")
        result.replace(path)
        proof = publish(drive, repeat, path)
        done = json.loads((out / "completion.json").read_text())
        assert done["updates"] == 90
        print(json.dumps(dict(replicate=repeat, updates=done["updates"], charged_flops=done["consumed_training_flops"],
                              dice=dice, coverage=coverage, sha256=proof["sha256Checksum"],
                              id=proof["id"], verified_at=time.time())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("MSD", "ADNI"))
    run(parser.parse_args().dataset)
