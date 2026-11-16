"""Admit at most one queued job; copy and verify results before retiring run checkpoints."""
import os
import fcntl,hashlib,json,shlex,subprocess,time,zipfile
from pathlib import Path
D=Path(__file__).resolve().parent
S=os.environ['HIPPO_CLUSTER_ROOT'].rstrip('/') + '/full_data_baselines_20261115'
P=S+'/full_lr_v1'
HOST=os.environ.get('HIPPO_CLUSTER_HOST', 'hippo-cluster')
def ssh(*args):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,shlex.join(list(map(str,args)))],text=True)
def remote(code,*args):return ssh('python3','-c',code,P,*args)
inspect="""from pathlib import Path
import sys,json,hashlib
p=Path(sys.argv[1]);todo={}
for path in p.glob('*/*_full_lr_v1_results.zip'):
 k=path.name.split('_full_lr_v1_results.zip')[0]
 if not (path.parent/'work'/(k+'_lr2x')/'cleanup.json').exists():todo[str(path.relative_to(p))]=hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(dict(todo=todo,complete=len(list(p.glob('*/work/*/cleanup.json'))))))"""
admit="""import sys,fcntl,subprocess,json,time
from pathlib import Path
p=Path(sys.argv[1]);s=p.parent
sys.path.insert(0,str(s.parent/'mni_medsam3_n5_f0_s0_400_20261110/source'))
from scripts.cross_dataset_campaign.storage import quota
with (s/'campaign_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 def queue():return dict(l.split() for l in subprocess.check_output(['squeue','--me','-h','-o','%i %T'],text=True).splitlines())
 q=queue()
 if len(q)>1 or any(t!='RUNNING' for t in q.values()):print('Admission waiting:',q);sys.exit(0)
 receipt=p/'scheduler.json'
 if receipt.exists():
  j=json.loads(receipt.read_bytes())['job']
  if j in q:sys.exit(0)
  st=subprocess.check_output(['sacct','-X','-j',j,'-nPo','State,ExitCode'],text=True).strip()
  if not(st.startswith('TIMEOUT|') or st=='FAILED|75:0'):print('Inspect previous job',j,st);sys.exit(0)
 storage=quota();assert storage['free_gib']>=24
 c=['sbatch','--parsable','--output='+str(p/'slurm-%j.out')]
 if q:c.append('--dependency=afterany:'+next(iter(q)))
 j=subprocess.check_output(c+[str(p/'cluster.sbatch')],cwd=p,text=True).strip().split(';')[0];assert j.isdigit()
 receipt.write_text(json.dumps(dict(job=j,quota=storage,time=time.time())))
 print('Submitted',j)
"""
def step():
    state=json.loads(remote(inspect))
    for rel,h in state['todo'].items():
        ds,name=rel.split('/');k=name.removesuffix('_full_lr_v1_results.zip')
        assert ds in ('MNI','HFH') and k in json.loads((D/'plan.json').read_bytes())
        f=D/'archive'/name; f.parent.mkdir(exist_ok=True);tmp=f.with_suffix('.partial')
        subprocess.run(['rsync','-rt',HOST+':'+P+'/'+rel,str(tmp)],check=True);assert hashlib.sha256(tmp.read_bytes()).hexdigest()==h
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None and json.loads(z.read('lr_contrast_receipt.json'))['key']==k
        tmp.replace(f)
        remote("""import json,sys,pathlib,subprocess
p=pathlib.Path(sys.argv[1]);ds,k,h,d=sys.argv[2:]
a=p/ds/'work'/(k+'_lr2x')/'durable_ack.json';a.write_text(json.dumps(dict(sha256=h,archive=d)))
subprocess.run([sys.executable,str(p/'run.py'),k,'--clean'],check=True)""",ds,k,h,str(f))
    if state['complete']==54:return False
    print(remote(admit).strip(),flush=True);return True
if __name__=='__main__':
    g=(D/'watch.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            if not step():break
        except (subprocess.SubprocessError,OSError) as e:print('Connection retry',e,flush=True)
        time.sleep(50)
