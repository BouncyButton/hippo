"""Supervise one assigned MSD fold and publish compact, recoverable progress."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from googleapiclient.http import MediaInMemoryUpload
from continue_colab import FOLDER, existing, session

HOME = Path("/content")
ROOT = HOME / "standard_msd_medsam"


def active_processes():
    active = []
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = p.read_bytes().split(bytes(1))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if os.fsencode(ROOT/"standard_msd_medsam.py") in argv or os.fsencode(ROOT/"covered_medical.py") in argv:
            active.append(int(p.parent.name))
        elif os.fsencode(ROOT/"stage_msd_public.py") in argv and b"--start" in argv:
            active.append(int(p.parent.name))
        elif b"-c" in argv and any(arg.startswith(b"import fcntl,os,sys,json;from pathlib import Path;") and b"worker.lock" in arg for arg in argv):
            active.append(int(p.parent.name))
    return active


def read(path, default):
    return json.loads(path.read_bytes()) if path.is_file() else default


def run(fold, ds):
    keys=read(ROOT/"assignment.json",[])
    assert set(keys)=={f"MSD_medsam3_f{fold}_s{s}_standard_fold" for s in (17,83,191)}
    name=f"MSD_standard_fold_{fold}_progress.json"
    previous=None
    drive=None
    guard=(ROOT/"supervisor.lock").open("a")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            if drive is None:
                drive=session()
            statuses=[]
            for key in keys:
                work=ROOT/"work"/key
                path=ROOT/(key+"_results.zip")
                pub=existing(drive,path.name) if path.is_file() else None
                if pub:
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==pub["sha256Checksum"]
                status=dict(key=key,updates=len(read(work/"training_history.json",[])),
                            evaluated=len(read(work/"evaluation.json",{}).get("records",[])),verified=bool(pub))
                statuses.append(status)
            previous_run=None
            if ds!="MSD":
                adni="ADNI_medsam3_f1_s"+("0" if ds=="ADNI" else "1")+"_coverage_control"
                dep=HOME/"covered_medical_ADNI/work"/adni
                previous_run=dict(key=adni,evaluated=len(read(dep/"evaluation.json",{}).get("records",[])),
                                  completed=(dep/"completion.json").is_file())
            pids=active_processes()
            snapshot=dict(fold=fold,compute=ds,processes=pids,cases_per_fold=208,validation_per_fold=52,
                          keys=statuses,earlier_run=previous_run)
            if previous!=snapshot:
                body=json.dumps(dict(snapshot,utc=time.time()),sort_keys=True).encode()
                media=MediaInMemoryUpload(body,mimetype="application/json")
                prior=existing(drive,name)
                op=(drive.files().update(fileId=prior["id"],media_body=media,fields="id,sha256Checksum") if prior else
                    drive.files().create(body=dict(name=name,parents=[FOLDER]),media_body=media,fields="id,sha256Checksum"))
                evidence=op.execute(num_retries=4)
                assert evidence["sha256Checksum"]==hashlib.sha256(body).hexdigest()
                previous=snapshot
            if all(s["verified"] for s in statuses):
                return
            if not pids:
                command=([sys.executable,"-u",str(ROOT/"standard_msd_medsam.py"),*keys] if ds=="MSD" else
                         [sys.executable,"-u",str(ROOT/"stage_msd_public.py"),ds,str(fold),"--start"])
                with (HOME/f"msd_standard_fold{fold}.log").open("a") as log:
                    subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(HOME)),
                                     start_new_session=True,cwd=ROOT)
                print("Resumed assigned queue",fold,flush=True)
        except (OSError,ValueError,RuntimeError) as exc:
            print("Status retry",type(exc).__name__,str(exc)[:160],flush=True)
            drive=None
        except Exception as exc:
            from googleapiclient.errors import HttpError
            if not isinstance(exc,HttpError):
                raise
            print("Drive retry",exc.resp.status,flush=True)
            drive=None
        time.sleep(90)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("MSD","HFH","ADNI"))
    parser.add_argument("fold",type=int)
    args=parser.parse_args()
    assert {"MSD":0,"ADNI":1,"HFH":3}[args.dataset]==args.fold
    run(args.fold,args.dataset)
