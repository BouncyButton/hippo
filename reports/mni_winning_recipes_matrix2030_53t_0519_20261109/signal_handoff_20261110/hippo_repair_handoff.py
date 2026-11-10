import base64
import json
import shlex
import subprocess
import sys
from pathlib import Path
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE

NAME = 'mni_winning_recipes_matrix2030_53t_0519_20261109'
ROOT = Path('/Users/filippofocaccia/Desktop/hippo')
LOCAL = ROOT/'experiments'/NAME
REPORT = ROOT/'reports'/NAME/'signal_handoff_20261110'
STAGE = REMOTE/NAME
rev = read(REPORT/'prepared_revision.json')
old, new = rev['old_manifest_sha256'], rev['new_manifest_sha256']
mode = sys.argv[1]

if mode == 'snapshot':
    code = '''import fcntl,json,hashlib,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);old=sys.argv[2]
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
with (p/'worker.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(p/'manifest.json')==old
 names=['manifest.json','source/source_manifest.json','source/scripts/mni_constraints/winning_matrix_runner.py','source_version.json','preflight.json','failure.json','status.json','control.json']
 files={n:dict(text=(p/n).read_text(),sha256=sha(p/n)) for n in names if (p/n).exists()}
 recovery={}
 for name in ['MNI_unetr_vit_b16_dice_depth_f0s19','MNI_basic_unet16_dice_depth_f0s19']:
  d=p/'runs'/name
  recovery[name]={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in d.iterdir() if f.is_file()}
 complete=[d for d in (p/'runs').iterdir() if (d/'completion.json').exists()]
 assert len(complete)==148
 for d in complete:
  assert json.loads((d/'archive_acknowledgment.json').read_text())['status']=='verified'
  assert json.loads((d/'cleanup.json').read_text())['status']=='complete'
 queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True)
 print(json.dumps(dict(files=files,recovery=recovery,completed=len(complete),queue=queue,captured_at=time.time())))
'''
    value=json.loads(remote(['python3','-c',code,STAGE,old],timeout=600))
    for n,info in value['files'].items():
        target=ROOT/'reports'/NAME/'source_versions'/old/'remote_metadata'/n
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(info['text'])
        assert digest(target)==info['sha256']
    write(REPORT/'predeployment_snapshot.json',value)
    print(json.dumps(dict(completed=value['completed'],queue=value['queue'],recovery=value['recovery']),indent=2))

elif mode == 'deploy':
    snap=read(REPORT/'predeployment_snapshot.json')
    assert digest(LOCAL/'manifest.json')==new
    files={n:base64.b64encode((LOCAL/n).read_bytes()).decode() for n in [*rev['changed_files'],'manifest.json']}
    payload=dict(old=old,new=new,files=files,snapshot=snap)
    code='''import base64,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);v=json.load(sys.stdin);old=v['old']
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def put(f,b):
 f.parent.mkdir(parents=True,exist_ok=True);t=f.with_name(f.name+'.repair.tmp');t.write_bytes(b);t.replace(f)
with (p/'worker.lock').open('a') as worker,(p/'submission.lock').open('a') as submit:
 fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(submit,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(p/'manifest.json')==old
 queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True)
 rows=[r.split('|') for r in queue.splitlines() if r.strip()]
 assert len(rows)<=1 and all(r[1] in ('RUNNING','COMPLETING','CONFIGURING') and 'medsam' in r[2].lower() for r in rows),rows
 for name,members in v['snapshot']['recovery'].items():
  for n,info in members.items():assert sha(p/'runs'/name/n)==info['sha256'],(name,n)
 archive=p/'source_versions'/old
 for n,info in v['snapshot']['files'].items():
  assert sha(p/n)==info['sha256'],n
  put(archive/n,(p/n).read_bytes());assert sha(archive/n)==info['sha256']
 for n,b in v['files'].items():put(p/n,base64.b64decode(b))
 assert sha(p/'manifest.json')==v['new']
 for n,h in json.loads((p/'manifest.json').read_text()).items():assert sha(p/n)==h,n
 for n in ['failure.json','preflight.json']:
  assert sha(archive/n)==sha(p/n);(p/n).unlink()
 state=json.loads((p/'status.json').read_text());state.update(status='repair_preflight',active=[],updated_at=time.time(),runtime_manifest_sha256=v['new'])
 put(p/'status.json',json.dumps(state,indent=2).encode())
 print(json.dumps(dict(status='deployed_awaiting_preflight',manifest_sha256=v['new'],queue=queue,recovery_unchanged=True,time=time.time())))
'''
    process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','bocconi-cluster',shlex.join(['python3','-c',code,str(STAGE)])],input=json.dumps(payload),text=True,capture_output=True,timeout=900)
    if process.returncode:
        print(process.stderr);sys.exit(process.returncode)
    result=json.loads(process.stdout)
    write(REPORT/'deployment.json',result);print(json.dumps(result,indent=2))

elif mode == 'preflight':
    env=':'.join(str(STAGE/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    runtime=read(LOCAL/'plan.json')['conventional_python']
    result=remote(['env','PYTHONPATH='+env,'MPLCONFIGDIR=/tmp/hippo-signal-preflight','MPLBACKEND=Agg','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',runtime,'-m','scripts.mni_constraints.winning_matrix2030','preflight','--stage',STAGE],timeout=1800)
    actual=json.loads(remote(['cat',STAGE/'preflight.json']))
    assert actual['status']=='passed' and actual['manifest_sha256']==new
    write(REPORT/'production_preflight.json',actual)
    write(ROOT/'reports'/NAME/'production_preflight.json',actual)
    print(json.dumps(actual,indent=2))

elif mode == 'submit':
    preflight=read(REPORT/'production_preflight.json')
    assert preflight['status']=='passed' and preflight['manifest_sha256']==new
    env=':'.join(str(STAGE/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    runtime=read(LOCAL/'plan.json')['conventional_python']
    result=remote(['env','PYTHONPATH='+env,runtime,'-m','scripts.mni_constraints.winning_matrix2030','submit','--stage',STAGE],timeout=900)
    receipt=json.loads(remote(['cat',STAGE/'submission.json']))
    assert receipt['manifest_sha256']==new
    write(REPORT/'submission.json',receipt)
    print(json.dumps(receipt,indent=2))
