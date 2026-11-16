"""Collect finished MNI runs as they close and admit the remaining fixed blocks."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/"mni_adapter"))
from collect import HOST, ROOT, STAGE, PARENT, read, write

SNAPSHOT="""
import json,pathlib,sys,subprocess
s=pathlib.Path(sys.argv[1])
def read(p):return json.loads(p.read_text())
rows={}
for p in sorted((s/'runs').glob('MNI_*')):
    record={'complete':(p/'completion.json').is_file(),'prepared':(p/'archive_manifest.json').is_file()}
    if (p/'gpu_worker.json').is_file():record['job']=read(p/'gpu_worker.json')['slurm_job_id']
    elif (p/'allocation.json').is_file():record['job']=read(p/'allocation.json')['job']
    if (p/'budget_ledger.json').is_file():record['step']=read(p/'budget_ledger.json')['charged_attempts']
    rows[p.name]=record
print(json.dumps({'runs':rows,'submissions':[read(p) for p in s.glob('*_submission.json')],
    'queue':dict(l.split() for l in subprocess.check_output(['squeue','--me','-h','-o','%i %T'],text=True).splitlines())}))
"""


def remote(command):
    return subprocess.check_output(["ssh","-o","ConnectTimeout=12","-o","ServerAliveInterval=15",
                                    "-o","ServerAliveCountMax=2",HOST,shlex.join(list(map(str,command)))],text=True)


def once():
    state=json.loads(remote(["python3","-c",SNAPSHOT,STAGE]))
    rows=read(D/"manifest.json")
    changed=False
    for submission in state["submissions"]:
        keys=submission["args"][1:] if submission["block"]!="scratch" else []
        for key in keys:
            if rows[key]["status"]=="planned":
                rows[key].update(status="queued",job_or_runtime="slurm:"+submission["job"])
                changed=True
    for key,r in state["runs"].items():
        row=rows[key]
        if r.get("job") and row["status"]!="verified_complete" and state["queue"].get(r["job"])=="RUNNING":
            updates=dict(status="running",job_or_runtime="slurm:"+r["job"])
            if "step" in r: updates["last_verified_update"]=r["step"]
            if any(row.get(k)!=v for k,v in updates.items()):row.update(updates);changed=True
    if changed:write(D/"manifest.json",rows)
    for key,r in state["runs"].items():
        if rows[key]["status"]=="verified_complete" or not r["complete"]:continue
        if rows[key]["model"]=="medsam3":
            if key=="MNI_medsam3_f0_s0" and r.get("job") in state["queue"]:continue
            program=D/"mni_adapter/collect.py"
        else:
            if not r["prepared"]:continue
            program=D/"collect_scratch.py"
        subprocess.run([sys.executable,str(program),key],check=True)
        print("Archived",key,flush=True)
    admitted=remote(["env",f"PYTHONPATH={ROOT/PARENT/'source'}","python3",STAGE/"mni_admit.py"])
    if admitted: print(admitted.strip(),flush=True)
    current=read(D/"manifest.json")
    return all(r["status"]=="verified_complete" for r in current.values() if r["dataset"]=="MNI")


if __name__=="__main__":
    with (D/"mni_monitor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if once():break
            except subprocess.CalledProcessError as e:
                print("Collection or connection needs retry",e.returncode,time.time(),flush=True)
            time.sleep(30)
