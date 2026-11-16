"""Continue full-data Colab evaluations and publish verified compact archives."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from launch_scratch_colab import ACCOUNT, FOLDER, RuntimeCredentials, read, sha


def session():
    drive = build("drive", "v3", credentials=RuntimeCredentials(), cache_discovery=False)
    identity = drive.about().get(fields="user(emailAddress)").execute(num_retries=3)
    folder = drive.files().get(fileId=FOLDER, fields="trashed,owners(emailAddress)").execute(num_retries=3)
    assert identity["user"]["emailAddress"] == folder["owners"][0]["emailAddress"] == ACCOUNT
    assert not folder["trashed"]
    return drive


def existing(drive, name):
    files = drive.files().list(q=f"'{FOLDER}' in parents and trashed=false and name='{name}'",
        fields="files(id,size,sha256Checksum)", pageSize=3).execute(num_retries=4)["files"]
    if len(files) > 1:
        raise RuntimeError(f"Ambiguous archive for {name}")
    return files[0] if files else None


def publish(drive, key, path):
    proof = existing(drive, path.name)
    if not proof:
        media = MediaFileUpload(str(path), mimetype="application/zip", resumable=True)
        proof = drive.files().create(body={"name":path.name, "parents":[FOLDER]}, media_body=media,
            fields="id,size,sha256Checksum").execute(num_retries=5)
    payload = drive.files().get_media(fileId=proof["id"]).execute(num_retries=4)
    assert len(payload) == path.stat().st_size and hashlib.sha256(payload).hexdigest() == sha(path)
    print("ARCHIVED", key, proof["id"], sha(path), flush=True)
    return proof


def validate_output(key, root, medical):
    work = root/"work"/key
    complete = read(work/"completion.json")
    assert complete["status"] == "complete"
    if medical:
        protocol = read(root/"protocols"/f"{key}.json")
        assert read(work/"protocol.json") == dict(protocol, arm="dice")
        assert complete["updates"] == protocol["updates"] and complete["fold"] == protocol["fold"]
        assert complete["seed"] == protocol["seed"] and complete["evaluation_sha256"] == sha(work/"evaluation.json")
        assert sha(work/"checkpoint_latest.pt") == complete["checkpoint_sha256"]
        records = read(work/"evaluation.json")["records"]
        for label, names in [("training_support", protocol["train_cases"]), ("validation", protocol["evaluation_cases"])]:
            selected = [r for r in records if r["split"] == label]
            assert sorted(r["case_name"] for r in selected) == sorted(names)
        assert all("foreground_fp" in r and "foreground_fn" in r and math.isfinite(r["macro_hard_dice"]) for r in records)
        schedule = read(work/"slab_schedule.json")
        assert len(schedule) == protocol["updates"] * protocol["accumulation"]
        coverage = len({case for case, anchor in schedule})
        return complete["validation_dice"], coverage
    row = read(root/"cells.json")[key]
    config = read(work/"config.json")
    assert complete["updates"] == row["target_updates"] and not complete["discarded_attempts"]
    assert complete["consumed_training_flops"] <= row["budget_flops"]
    for name in ("best","latest"):
        assert sha(work/f"checkpoint_{name}.pt") == complete[f"checkpoint_{name}_sha256"]
    for filename, names in [("training_final.json",config["train_cases"]),("validation_latest.json",config["validation_cases"])]:
        records = read(work/filename)["cases"]
        assert sorted(r["case_name"] for r in records) == sorted(names)
        assert all("confusion" in r and "foreground_fp" in r and "foreground_fn" in r for r in records)
    dice = read(work/"validation_latest.json")["summary"]["macro_hard_dice"]
    assert math.isfinite(dice)
    return dice, len(read(work/"coverage.json")["sampled_crops"])


def archive(key, bundle, root, medical):
    work = root/"work"/key
    dice, sampled = validate_output(key, root, medical)
    target = root/(key+"_results.zip")
    if not target.exists():
        with zipfile.ZipFile(target.with_suffix(".partial"), "w", zipfile.ZIP_DEFLATED) as z:
            for item in sorted(work.iterdir()):
                if item.is_file() and item.suffix in {".json", ".jsonl", ".log", ".csv", ".txt"}:
                    z.write(item, item.name)
            if medical:
                z.write(root/"protocols"/f"{key}.json","protocols/input.json")
                z.write(root/"medical_worker.py", "medical_worker.py")
            else:
                for item in sorted((root/"protocols"/key).iterdir()):
                    if item.is_file():
                        z.write(item,"protocols/"+item.name)
                z.write(bundle/"worker.py", "worker.py")
            z.write(bundle/"checksums.json","input_checksums.json")
        target.with_suffix(".partial").replace(target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and json.loads(z.read("completion.json")) == read(work/"completion.json")
    return target, dice, sampled


def evaluate(key, bundle, medical_root, env):
    medical = "_medsam3_" in key
    root = medical_root if medical else bundle
    work = root/"work"/key
    work.mkdir(parents=True, exist_ok=True)
    drive = session()
    remote = existing(drive, key+"_results.zip")
    if remote:
        return {"id":remote["id"], "status":"already_archived","sha256":remote["sha256Checksum"]}
    py = str(bundle/"venv/bin/python")
    if not (work/"completion.json").exists():
        command = [py,str(root/"medical_worker.py"),str(root/"protocols"/(key+".json")),str(root/"work"),str(root/"recovery")] if medical else [
            py,str(bundle/"worker.py"),key,"--bundle",str(bundle),"--work",str(bundle/"work")]
        # Wait for any pilot explicitly launched in this VM before admitting its cell.
        lock = (work/"worker.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_UN)
        if not (work/"completion.json").exists():
            print("START", key, socket.gethostname(), flush=True)
            with (work/"training.log").open("a") as log, (work/"gpu.csv").open("a") as gpu:
                monitor = subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total",
                                            "--format=csv","-l","10"],stdout=gpu)
                try:
                    proc = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    if proc.returncode:
                        raise RuntimeError((work/"training.log").read_text(errors="replace")[-1100:])
                finally:
                    monitor.terminate()
                    monitor.wait()
    target, dice, sampled = archive(key,bundle,root,medical)
    proof = publish(drive,key,target)
    return dict(status="verified_complete", final_dice=dice, unique_crops_sampled=sampled,
                updates=read(work/"completion.json")["updates"], id=proof["id"], sha256=sha(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["HFH","ADNI","MSD"])
    parser.add_argument("--scratch-workers", type=int, default=2)
    parser.add_argument("--hold-pilot",action="store_true")
    args = parser.parse_args()
    bundle=Path("/content/full_data_"+args.dataset)
    medical=Path(str(bundle)+"_medical")
    guard=(bundle/"continuation.lock").open("a")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],text=True).strip()
    assert len(gpu.splitlines())==1 and any(k in gpu for k in ("L4","A100"))
    source=bundle/"source"
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[bundle/"compatibility",source,source/"scripts",source/"baselines/nnUNet"])),
        PYTHONUNBUFFERED="1",nnUNet_compile="false",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",WANDB_MODE="disabled")
    scratch=list(read(bundle/"cells.json"))
    if args.hold_pilot:
        scratch.remove(args.dataset+"_nnunet_f0_s0")
    medical_keys=sorted(p.stem for p in (medical/"protocols").glob("*.json"))
    assert len(medical_keys)==9
    journal=bundle/"continuation.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.scratch_workers) as sp, concurrent.futures.ThreadPoolExecutor(max_workers=1) as mp:
        futures={pool.submit(evaluate,k,bundle,medical,env):k for pool,keys in [(sp,scratch),(mp,medical_keys)] for k in keys}
        for future in concurrent.futures.as_completed(futures):
            key=futures[future]
            try:
                record=future.result()
            except Exception as err:
                record=dict(status="needs_inspection",reason=str(err)[:1200])
            record.update(key=key,runtime=socket.gethostname(),gpu=gpu,time=time.time())
            with journal.open("a") as f:
                f.write(json.dumps(record,allow_nan=False)+"\n")
            print(json.dumps(record,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
