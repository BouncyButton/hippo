"""Run and archive one scratch cell on the current Google Colab VM."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

from google.auth.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ACCOUNT = os.environ.get('HIPPO_STORAGE_EMAIL', '')
FOLDER = os.environ.get('HIPPO_FULL_DATA_DRIVE_ROOT', '')


class RuntimeCredentials(Credentials):
    def refresh(self, request):
        self.token = subprocess.check_output(["gcloud","auth","print-access-token",
                                              "--account="+ACCOUNT,"--quiet"],text=True).strip()


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    tmp.replace(p)


def sha(p, algorithm="sha256"):
    with p.open("rb") as f:
        return hashlib.file_digest(f,algorithm).hexdigest()


def main(key, bundle):
    drive = build("drive","v3",credentials=RuntimeCredentials(),cache_discovery=False)
    identity = drive.about().get(fields="user(emailAddress),storageQuota").execute()
    owner = drive.files().get(fileId=FOLDER,fields="name,owners(emailAddress),trashed").execute()
    if identity["user"]["emailAddress"] != ACCOUNT or owner["trashed"] or owner["owners"][0]["emailAddress"] != ACCOUNT:
        raise RuntimeError("Unexpected durable storage identity or folder owner")
    rows = read(bundle/"cells.json")
    row = rows[key]
    assert row["model"] != "medsam3"
    work = bundle/"work"
    out = work/key
    out.mkdir(exist_ok=True,parents=True)
    guard=(out/"supervisor.lock").open("a")
    fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)

    def existing(name):
        result=drive.files().list(q=f"'{FOLDER}' in parents and trashed=false and name='{name}'",
                                  fields="files(id,name,md5Checksum,size)",pageSize=3).execute(num_retries=5)["files"]
        if len(result)>1:
            raise ValueError("Duplicate durable files for "+name)
        return result[0] if result else None

    def upload(name, path):
        record=existing(name)
        media=MediaFileUpload(str(path),mimetype="application/octet-stream",resumable=True,chunksize=8*1024**2)
        if record:
            call=drive.files().update(fileId=record["id"],media_body=media,fields="id")
        else:
            call=drive.files().create(body=dict(name=name,parents=[FOLDER]),media_body=media,fields="id")
        identifier=call.execute(num_retries=5)["id"]
        proof=drive.files().get(fileId=identifier,fields="id,md5Checksum,size,owners(emailAddress)").execute(num_retries=5)
        if proof["owners"][0]["emailAddress"]!=ACCOUNT or int(proof["size"])!=path.stat().st_size or proof["md5Checksum"]!=sha(path,"md5"):
            raise ValueError("Drive read-back checksum mismatch: "+name)
        return proof

    name=key+"_state.json"
    existing_state=existing(name)
    state=dict(key=key, runtime=socket.gethostname(), dataset=row["dataset"], model=row["model"],
               protocol_sha256=sha(bundle/"protocols"/key/"protocol.json"), started=time.time())
    if existing_state:
        saved=json.loads(drive.files().get_media(fileId=existing_state["id"]).execute(num_retries=5))
        if saved["protocol_sha256"] != state["protocol_sha256"]:
            raise ValueError("Remote protocol differs")
        if saved["status"]=="verified_complete":
            complete=existing(key+"_results.zip")
            assert complete and complete["md5Checksum"]==saved["result_md5"]
            print(json.dumps(saved),flush=True)
            return
        if saved["runtime"] != state["runtime"] or not (out/"budget_ledger.json").exists():
            raise RuntimeError("A prior invocation exists: inspect its recovery archive and charged updates before resuming")
        state=saved
    def publish(**change):
        state.update(change)
        path=bundle/name
        write(path,state)
        upload(name,path)
    state["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,memory.total","--format=csv,noheader"],text=True).strip()
    assert any(g in state["gpu"].split(",")[0] for g in ("L4","A100"))
    publish(status="starting")
    py=bundle/"venv/bin/python"
    src=bundle/"source"
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[bundle/"compatibility",src,src/"scripts",src/"baselines/nnUNet"])),
             PYTHONUNBUFFERED="1",nnUNet_compile="false",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",WANDB_MODE="disabled")
    cmd=[str(py),str(bundle/"worker.py"),key,"--bundle",str(bundle),"--work",str(work)]
    metrics=["timestamp","name","utilization.gpu","memory.used","memory.total"]
    with (out/"training.log").open("a") as log,(out/"gpu.csv").open("a") as gpu_log:
        monitor=subprocess.Popen(["nvidia-smi","--query-gpu="+",".join(metrics),"--format=csv","-l","10"],stdout=gpu_log)
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        previous=0
        try:
            publish(status="worker_started",pid=proc.pid)
            while proc.poll() is None:
                ledger=out/"budget_ledger.json"
                if ledger.exists() and time.monotonic()-previous>180:
                    previous=time.monotonic()
                    latest=out/"checkpoint_latest.pt"
                    if latest.is_file():
                        # Atomic trainer replacements keep open checkpoint inodes immutable.
                        with tempfile.TemporaryDirectory(dir=bundle) as temp:
                            recovery=Path(temp)/"recovery.zip"
                            with zipfile.ZipFile(recovery,"w",zipfile.ZIP_STORED) as z:
                                z.write(latest,latest.name)
                                for n in ["config.json","calibration.json","budget_ledger.json","exposure.jsonl","metrics.csv"]:
                                    if (out/n).exists(): z.write(out/n,n)
                            upload(key+"_recovery.zip",recovery)
                    publish(status="running",last_observed_attempt=read(ledger)["charged_attempts"])
                time.sleep(5)
            if proc.returncode:
                raise RuntimeError(f"Training exited {proc.returncode}; see training.log")
        except BaseException:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                proc.wait()
            upload(key+"_failure.log",out/"training.log")
            publish(status="needs_inspection")
            raise
        finally:
            monitor.terminate()
            monitor.wait()
    done=read(out/"completion.json")
    cfg=read(out/"config.json")
    val=read(out/"validation_latest.json")
    coverage=read(out/"coverage.json")
    assert done["status"]=="complete" and done["updates"]==row["target_updates"]
    assert done["discarded_attempts"]==0 and done["consumed_training_flops"]<=row["budget_flops"]
    assert len(val["cases"])==len(cfg["validation_cases"]) and {c["case_name"] for c in val["cases"]}==set(cfg["validation_cases"])
    train=read(out/"training_final.json")
    assert {c["case_name"] for c in train["cases"]}==set(cfg["train_cases"])
    assert all("confusion" in c and "foreground_fp" in c and "foreground_fn" in c for c in train["cases"]+val["cases"])
    assert all(sha(out/file)==done[field] for file,field in [("checkpoint_best.pt","checkpoint_best_sha256"),("checkpoint_latest.pt","checkpoint_latest_sha256")])
    archives=bundle/"archives"
    archives.mkdir(exist_ok=True)
    result=archives/(key+"_results.zip")
    with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix in {".json",".log",".csv",".jsonl"}: z.write(path,path.name)
        for p in (bundle/"protocols"/key).iterdir(): z.write(p,"protocols/"+p.name)
        z.write(bundle/"worker.py","worker.py")
        z.write(bundle/"checksums.json","input_checksums.json")
    with zipfile.ZipFile(result) as z: assert z.testzip() is None
    proof=upload(result.name,result)
    publish(status="verified_complete",final_dice=val["summary"]["macro_hard_dice"],updates=done["updates"],
            unique_crops_sampled=len(coverage["sampled_crops"]),result_id=proof["id"],result_md5=proof["md5Checksum"])
    print(json.dumps(state),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("bundle",type=Path)
    args=parser.parse_args()
    main(args.key,args.bundle)
