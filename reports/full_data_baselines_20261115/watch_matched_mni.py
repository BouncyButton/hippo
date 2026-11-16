"""Admit native MNI controls and retire heavy files after verified off-cluster archiving."""
import os
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zipfile

D=Path(__file__).resolve().parent
STAGE=(Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())) / 'full_data_baselines_20261115')
REMOTE=STAGE/"matched_MNI"
DEST=D/"matched_controls"/"archive"
HOST=os.environ.get("HIPPO_CLUSTER_HOST", "hippo-cluster")


def ssh(*args):
    return subprocess.check_output(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12",HOST,
                                   shlex.join([str(s) for s in args])],text=True)


def remote(code,*args):
    return ssh("python3","-c",code,REMOTE,*args)


def collect(name,sha):
    local=DEST/name
    if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest()!=sha:
        subprocess.run(["rsync","-rt",f"{HOST}:{REMOTE/name}",str(local)+".partial"],check=True)
        temp=Path(str(local)+".partial")
        assert hashlib.sha256(temp.read_bytes()).hexdigest()==sha
        temp.replace(local)
    with zipfile.ZipFile(local) as z:
        assert z.testzip() is None
        r=json.loads(z.read("matched_receipt.json"))
        assert name==r["key"]+"_paper_matched_results.zip" and r["observed"]==r["eligible"]
    subprocess.run([sys.executable, str(D / "comparisons.py")], check=True)
    # The ZIP has all compact outputs. Delete only that run's named checkpoints.
    remote("""import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);name,expected,dest=sys.argv[2:5];key=name.removesuffix('_paper_matched_results.zip')
assert key.startswith('MNI_') and '/' not in key and hashlib.sha256((p/name).read_bytes()).hexdigest()==expected
out=p/'work'/key
assert json.loads((out/'matched_receipt.json').read_text())['key']==key
ack=out/'durable_ack.json';tmp=ack.with_suffix('.tmp')
for name in ('checkpoint_latest.pt','checkpoint_best.pt'):(out/name).unlink(missing_ok=True)
tmp.write_text(json.dumps(dict(sha256=expected,archive=dest)));tmp.replace(ack)
""",name,sha,str(local))
    print("Archived",r["key"],r["dice"],flush=True)


STATE="""import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);todo={}
for f in p.glob('*_paper_matched_results.zip'):
 k=f.name.removesuffix('_paper_matched_results.zip')
 if not (p/'work'/k/'durable_ack.json').exists():todo[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
print(json.dumps(dict(todo=todo,complete=sum(1 for f in (p/'work').glob('*/durable_ack.json')))))"""

ADMIT="""import fcntl,json,os,pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[1]);stage=p.parent
sys.path.insert(0,str(stage.parent/'mni_medsam3_n5_f0_s0_400_20261110/source'))
from scripts.cross_dataset_campaign.storage import quota
with (stage/'campaign_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 def queue():return dict(x.split() for x in subprocess.check_output(['squeue','--me','-h','-o','%i %T'],text=True).splitlines())
 jobs=queue();receipt=p/'scheduler.json'
 if len(jobs)>1 or any(v not in ('RUNNING','COMPLETING','CONFIGURING') for v in jobs.values()):sys.exit()
 if receipt.exists():
  prev=json.loads(receipt.read_text());job=prev['job']
  if job in jobs:sys.exit()
  result=subprocess.check_output(['sacct','-X','-j',job,'-nPo','State,ExitCode'],text=True).strip()
  if not (result=='FAILED|75:0' or result.startswith('TIMEOUT|')):print('Previous job:',job,result);sys.exit()
  if any((f.parent/'completion.json').exists() and not (f.parent/'durable_ack.json').exists() for f in (p/'work').glob('*/completion.json')):sys.exit()
 if quota()['free_gib']<24:raise RuntimeError('Insufficient BeeGFS headroom for the two workers')
 cmd=['sbatch','--parsable','--output='+str(p/'slurm-%j.out')]
 if jobs:cmd.append('--dependency=afterany:'+next(iter(jobs)))
 job=subprocess.check_output(cmd+[str(stage/'matched_mni.sbatch')],cwd=stage,text=True).strip().split(';')[0];assert job.isdigit()
 if len(queue())>2:
  if queue().get(job)=='PENDING':subprocess.run(['scancel',job],check=True)
  raise RuntimeError('Concurrent submission')
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(dict(job=job,time=time.time())));tmp.replace(receipt)
 print('Admitted',job,flush=True)
"""


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST.parent/"mni.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                state=json.loads(remote(STATE))
                for name,sha in state["todo"].items():collect(name,sha)
                if state["complete"]==36:return
                msg=remote(ADMIT)
                if msg.strip():print(msg.strip(),flush=True)
            except (subprocess.CalledProcessError,OSError) as e:print("Retry",e,flush=True)
            time.sleep(50)


if __name__=="__main__":main()
