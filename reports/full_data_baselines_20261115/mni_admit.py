"""Admit fixed MNI blocks within the existing account-wide Slurm limits."""
import os
from pathlib import Path
import fcntl
import json
import subprocess
import time

STAGE=(Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())) / 'full_data_baselines_20261115')


def queue():
    text=subprocess.check_output(["squeue",'--me','-h',"-o","%i %T"],text=True)
    return dict(line.split() for line in text.splitlines() if line.strip())


def release_scratch_receipt(path, jobs):
    """Allow the scratch sweep to resume after the collector acknowledges a late archive."""
    receipt=json.loads(path.read_text())
    job=receipt["job"]
    if job in jobs or not job.isdigit():
        return
    outcome=subprocess.check_output(["sacct","-X","-j",job,"-nPo","State,ExitCode"],text=True)
    if outcome.strip()!="FAILED|75:0":
        return
    keys=json.loads((STAGE/"mni_scratch/cells.json").read_text())
    pending=False
    for key in keys:
        out=STAGE/"runs"/key
        if (out/"completion.json").exists():
            ack=out/"cleanup.json"
            if not ack.is_file() or json.loads(ack.read_text()).get("status")!="complete":
                return
        else:
            pending=True
    if pending:
        path.rename(STAGE/f"scratch_submission_{job}.failed.json")


def main():
    from scripts.cross_dataset_campaign.storage import quota
    from scripts.cross_dataset_campaign.common import write
    blocks=[("scratch",["mni_scratch.sbatch"])]
    left=[f"MNI_medsam3_f{f}_s{s}" for f in (1,2) for s in (0,1,2)]
    for i in range(0,len(left),2):
        blocks.append((f"adapter_{i//2}",["mni_adapter/pair.sbatch",*left[i:i+2]]))
    with (STAGE/"campaign_admission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=queue()
        if len(jobs)>1 or any(s not in {"RUNNING","COMPLETING","CONFIGURING"} for s in jobs.values()):
            return
        if not (STAGE/"runs/MNI_medsam3_f0_s0/cleanup.json").exists():
            return
        for name,args in blocks:
            path=STAGE/(name+"_submission.json")
            if name=="scratch" and path.exists():
                release_scratch_receipt(path,jobs)
            if path.exists():continue
            used=quota()
            if used["free_gib"]<12: raise RuntimeError("Insufficient personal BeeGFS headroom")
            command=["sbatch","--parsable"]
            if jobs:command.append("--dependency=afterany:"+next(iter(jobs)))
            command.extend(["--output="+str(STAGE/"slurm-%j.out"),*args])
            token=subprocess.check_output(command,cwd=STAGE,text=True).strip()
            job=token.split(";")[0]
            if not job.isdigit(): raise ValueError("Unexpected Slurm receipt")
            actual=queue()
            if len(actual)>2 and actual.get(job)=="PENDING":
                subprocess.run(["scancel",job],check=True)
                raise RuntimeError("Withdrew own pending job after a concurrent submission")
            write(path,dict(job=job,block=name,args=args,prior_jobs=jobs,account_queue=actual,quota=used,time=time.time()))
            print(name,job,flush=True)
            break


if __name__=="__main__":
    main()
