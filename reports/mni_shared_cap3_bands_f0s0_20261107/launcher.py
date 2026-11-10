"""Queue the four bands pilots after the active baseline study, preserving two jobs maximum."""
import json
import subprocess
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE,read,write
from scripts.mni_constraints.headroom_bands import ROOT,NAME,PARENT,MODULE,verify


def main():
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;parent=REMOTE/PARENT
    report=ROOT/'reports'/NAME;report.mkdir(parents=True,exist_ok=True)
    plan=verify(stage);assert not (report/'submission.json').exists(),'Already submitted'
    parent_env=':'.join(str(parent/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,sys
from pathlib import Path
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);own={f.stem for f in (p/'submissions').glob('*.json')};q=controller.queue()
assert len(q)<=2 and all(r[0] in own for r in q),q
assert not json.loads((p/'control.json').read_text())['pause']
state=json.loads((p/'status.json').read_text());assert state['status'] in ('running','complete'),state['status']
usage=quota();assert usage['free_gib']>=12
print(json.dumps({'queue':q,'baseline_status':state['status'],'baseline_completed':state['completed'],'quota':usage}))'''
    write(report/'access_status.json',json.loads(remote(['env','PYTHONPATH='+parent_env,'python3','-c',code,parent])))
    remote(['python3','-c','import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not p.exists();p.mkdir()',dest])
    subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python']]
    output=remote(command+['-m',MODULE,'preflight','--stage',dest],timeout=600)
    write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]))
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.cross_dataset_campaign.collect','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
    code='''import importlib,json,subprocess,sys,time
from pathlib import Path
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read,write
stage,parent=map(Path,sys.argv[1:3]);m=importlib.import_module(sys.argv[3]);controller.verify=m.verify
m.verify(stage);q=controller.queue();own={f.stem for f in (parent/'submissions').glob('*.json')}
assert len(q)<=2 and all(r[0] in own for r in q),q
state=read(parent/'status.json');assert state['status'] in ('running','complete'),state['status']
pending=[r for r in q if r[1]=='PENDING'];running=[r for r in q if r[1] in ('RUNNING','COMPLETING','CONFIGURING')]
assert len(pending)<=1 and len(running)<=1
assert running or state['status']=='complete'
cancelled=[]
for row in pending:
 receipt=read(parent/'submissions'/(row[0]+'.json'))
 assert running and receipt['dependency']==running[0][0]
 assert read(parent/'successor.json')['job_id']==row[0]
 subprocess.run(['scancel',row[0]],check=True);cancelled.append(receipt)
for _ in range(15):
 if all(r[0] not in {x['job_id'] for x in cancelled} for r in controller.queue()):break
 time.sleep(1)
try:
 receipt=controller.submit(stage)
except Exception:
 # Preserve the old continuation if the replacement submission fails.
 now=controller.queue()
 if cancelled and len(now)==1 and running and now[0][0]==running[0][0]:
  old=importlib.import_module('scripts.mni_constraints.headroom_confirm');controller.verify=old.verify
  restored=controller.submit(parent,continuation=running[0][0]);write(stage/'restored_baseline_continuation.json',restored)
 raise
replacement={'cancelled_baseline_continuations':cancelled,'bands_submission':receipt,'baseline_recipes_changed':False}
write(stage/'queue_replacement.json',replacement)
print(json.dumps(replacement))'''
    result=json.loads(remote(command+['-c',code,dest,parent,MODULE]))
    write(report/'queue_replacement.json',result);write(report/'submission.json',result['bands_submission'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
