"""Collect one finished MNI adapter run, verify the archive, then retire its checkpoint."""
import os
import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
REPO = CAMPAIGN.parents[1]
sys.path.insert(0, str(REPO))
from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.cross_dataset_campaign.storage import verify_archive

HOST = os.environ.get("HIPPO_CLUSTER_HOST", "hippo-cluster")
ROOT = (Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())))
PARENT = "mni_medsam3_n5_f0_s0_400_20261110"
STAGE = ROOT / CAMPAIGN.name


def remote_python(program, *args):
    command = shlex.join(["env", f"PYTHONPATH={ROOT/PARENT/'source'}",
                          "python3", "-c", program, *map(str, args)])
    subprocess.run(["ssh", "-o", "ConnectTimeout=15", HOST, command], check=True)


def collect(key):
    manifest = read(CAMPAIGN / "manifest.json")
    row = manifest[key]
    assert row["dataset"] == "MNI" and row["model"] == "medsam3"
    parent = REPO / "reports" / PARENT / "stage"
    source_hashes = CAMPAIGN / "mni_pilot/parent_inputs.sha256"
    for line in source_hashes.read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(parent / name) != expected:
            raise ValueError(f"Original source or dataset differs: {name}")
    out = STAGE / "runs" / key
    archive = CAMPAIGN / "archive" / key
    archive.mkdir(parents=True, exist_ok=True)
    task = dict(id=key, dataset="MNI", kind="foundation", fold=row["partition"], seed=row["seed"])
    # The pilot's scheduler log is outside its run directory. Its job must
    # have exited before including the log or the monitor CSV in the manifest.
    job = row["job_or_runtime"].split(":")[1]
    check = """
import json, shutil, sys, subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import prepare
out, job = Path(sys.argv[1]), sys.argv[2]
if not (out/'completion.json').is_file():
    raise SystemExit('Training or evaluation still in progress')
if (out/'gpu.csv').exists() and not (out/'archive_manifest.json').exists():
    if job in subprocess.check_output(['squeue','--me','-h','-o','%i'],text=True).split():
        raise SystemExit('Pilot is closing its output files')
    shutil.copyfile(out.parents[1]/('slurm-'+job+'.out'),out/'training.log')
prepare(out,json.loads(sys.argv[3]))
"""
    remote_python(check, out, job, json.dumps(task))
    subprocess.run(["rsync", "-rt", "--exclude=*.pt", "--exclude=*.tmp", "--exclude=*.nii.gz",
                    f"{HOST}:{out}/", str(archive) + "/"], check=True)
    receipt = read(archive / "archive_manifest.json")
    verify_archive(archive, receipt)
    protocol, done = (read(archive / name) for name in ["protocol.json", "completion.json"])
    split = read(CAMPAIGN / "splits.json")[row["split"]]
    assert protocol["train_cases"] == split["train"] and protocol["evaluation_cases"] == split["validation"]
    assert done["fold"] == row["partition"] and done["seed"] == row["seed"] and done["updates"] == row["target_updates"]
    assert done["evaluation_sha256"] == digest(archive / "evaluation.json")
    schedule = read(archive / "slab_schedule.json")
    assert len(schedule) == done["updates"] * protocol["accumulation"]
    sampled = {case for case, anchor in schedule}
    assert sampled <= set(split["train"])
    evaluation = read(archive / "evaluation.json")["records"]
    val = [r for r in evaluation if r["split"] == "validation"]
    assert len(val) == len(split["validation"]) and all("foreground_fn" in r and "foreground_fp" in r for r in evaluation)
    assert math.isclose(sum(r["macro_hard_dice"] for r in val)/len(val), done["validation_dice"], abs_tol=1e-12)
    ack = dict(status="verified", source_verified=True, durable_archive=str(archive),
               manifest_sha256=digest(archive / "archive_manifest.json"), source_archive=str(parent / "source"),
               parent_inputs_sha256=digest(source_hashes))
    write(archive / "archive_acknowledgment.json", ack)
    remote_python("""
import json,sys
from pathlib import Path
from scripts.cross_dataset_campaign.common import write, digest
from scripts.cross_dataset_campaign.storage import retire
out=Path(sys.argv[1]); ack=json.loads(sys.argv[2])
assert digest(out/'archive_manifest.json')==ack['manifest_sha256']
write(out/'archive_acknowledgment.json',ack)
retire(out)
""", out, json.dumps(ack))
    subprocess.run(["rsync", "-rt", f"{HOST}:{out}/cleanup.json", str(archive)], check=True)
    assert read(archive / "cleanup.json")["status"] == "complete"
    manifest = read(CAMPAIGN / "manifest.json")
    manifest[key].update(status="verified_complete", final_dice=done["validation_dice"],
                         unique_crops_sampled=len(sampled), unique_subjects_sampled=len({c.rsplit('_',1)[0] for c in sampled}),
                         last_verified_update=done["updates"], archive=str(archive.relative_to(REPO)))
    write(CAMPAIGN / "manifest.json", manifest)
    subprocess.run([sys.executable, str(CAMPAIGN / "comparisons.py")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key")
    collect(parser.parse_args().key)
