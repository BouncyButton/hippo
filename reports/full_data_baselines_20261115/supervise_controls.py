"""Foreground Colab supervisor for the corrected scratch and medical runs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from continue_colab import FOLDER, session


def busy(lock):
    if not lock.parent.is_dir():
        return False
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def published(drive, dataset):
    names = set()
    token = None
    while True:
        page = drive.files().list(q=f"'{FOLDER}' in parents and trashed=false and name contains '{dataset}_'",
            fields="nextPageToken,files(name)", pageSize=1000, pageToken=token).execute(num_retries=4)
        names.update(f["name"] for f in page["files"])
        token = page.get("nextPageToken")
        if not token:
            return names


def main(ds):
    root = Path("/content")
    bundle = root / ("full_data_" + ds)
    matched = root / ("matched_" + ds)
    medical = root / ("covered_medical_" + ds)
    env = dict(os.environ, PYTHONPATH=str(root))
    latest = {}
    children = {}
    failures = dict(scratch=0, medical=0, patch=0, development=0)
    scratch = {k for k,r in json.loads((matched/"jobs.json").read_text()).items() if r["feasible"]}
    medical_keys = {f.stem for f in (root/("full_data_"+ds+"_medical")/"protocols").glob("*.json")}
    assert len(medical_keys) == 9
    queues = {
        "scratch": (matched / "runner.lock", [str(bundle/"venv/bin/python"), "-u",
            str(matched/"run_matched_controls.py"), "--original", str(bundle), "--cloud"]),
        "medical": (medical / "queue.lock", [sys.executable, "-u", str(root/"covered_medical.py"), ds]),
    }
    while True:
        try:
            drive = session()
            names = published(drive, ds)
            done = {k for k in scratch if k+"_paper_matched_results.zip" in names}
            # Reuse requires reading the published protocol and actual training schedule.
            reused = set()
            for logfile in [root/f"covered_{ds}.log", root/f"supervised_medical_{ds}.log"]:
                if logfile.is_file():
                    for line in logfile.read_text(errors="replace").splitlines():
                        if line.startswith("REUSED "):
                            reused.add(line.split()[1])
            medical_done = {k for k in medical_keys if (k+"_coverage_control_results.zip" in names or
                            (k in reused and k+"_results.zip" in names))}
            missing = {"scratch": sorted(scratch-done), "medical": sorted(medical_keys-medical_done)}
            state = dict(scratch_archived=len(done),scratch_total=len(scratch),
                         medical_archived=len(medical_done),medical_total=len(medical_keys))
            for role, suffix in (("patch","_training_patch_results.zip"),("development","_lr_development_results.zip")):
                folder = root / (role + "_" + ds)
                if not (folder / "jobs.json").is_file():
                    continue
                planned = set(json.loads((folder / "jobs.json").read_text()))
                missing[role] = sorted(k for k in planned if k + suffix not in names)
                state[role + "_archived"] = len(planned) - len(missing[role])
                state[role + "_total"] = len(planned)
                queues[role] = (folder / "runner.lock", [str(bundle/"venv/bin/python"), "-u",
                    str(folder/"run_matched_controls.py"), "--original", str(bundle), "--cloud"])
            if latest != state:
                print(json.dumps(state), flush=True)
                latest = state
            if not any(missing.values()):
                return
            for role,keys in missing.items():
                if not keys:
                    continue
                if role == "development" and missing["scratch"]:
                    continue
                lock, command = queues[role]
                child = children.get(role)
                if child is not None:
                    result = child.poll()
                    if result is None:
                        continue
                    if result:
                        failures[role] += 1
                        print(role, "worker exit", result, "see", root/f"supervised_{role}_{ds}.log", flush=True)
                    del children[role]
                if busy(lock) or failures[role] >= 3:
                    continue
                with (root/f"supervised_{role}_{ds}.log").open("a") as stream:
                    children[role] = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
                print("Started", role, "with", len(keys), "remaining", flush=True)
        except (OSError, RuntimeError) as exc:
            print("Status check:", type(exc).__name__, str(exc)[:150], flush=True)
        time.sleep(45)


if __name__ == "__main__":
    main(sys.argv[1])
