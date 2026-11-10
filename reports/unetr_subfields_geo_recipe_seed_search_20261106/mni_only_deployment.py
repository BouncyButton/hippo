import json,subprocess,time,sys
from pathlib import Path
sys.path.insert(0,'/Users/filippofocaccia/Desktop/hippo')
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.unetr_subfields.geo_search import NAME,ROOT,verify
stage=ROOT/'experiments'/NAME; report=ROOT/'reports'/NAME; dest=REMOTE/NAME
amendment=read(report/'mni_only_amendment.json'); plan=verify(stage)
assert not (report/'mni_only_submission.json').exists()
assert not remote(['squeue','-h','-u','3160552','-o','%i']).strip()
code=r'''import json,hashlib,shutil,sys,time
from pathlib import Path
p=Path(sys.argv[1]); assert json.loads((p/'control.json').read_text())['pause']
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==sys.argv[2]
r=p/'revisions/before_mni_only';r.mkdir(parents=True,exist_ok=False)
for name in ('manifest.json','source/source_manifest.json','source/scripts/unetr_subfields/seed_search.py','status.json','failure.json','preflight.json','control.json'):
 f=p/name
 if f.exists():
  t=r/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,t)
for name in ('status.json','failure.json'):
 f=p/name
 if f.exists():f.rename(r/(name+'.retired'))
c=json.loads((p/'control.json').read_text());c['active_datasets']=['MNI'];c['scope_reason']='User deferred COBRA; pursue three qualifying MNI seeds.'
t=p/'control.json.tmp';t.write_text(json.dumps(c,indent=2)+'\n');t.replace(p/'control.json')
print(json.dumps({'previous_status':json.loads((r/'status.json').read_text()),'previous_failure':json.loads((r/'failure.json').read_text()) if (r/'failure.json').exists() else None}))'''
old=json.loads(remote(['python3','-c',code,dest,amendment['previous_manifest_sha256']]))
write(report/'revisions/before_mni_only/remote_stop.json',old)
subprocess.run(['rsync','-azR','./source/scripts/unetr_subfields/seed_search.py','./source/source_manifest.json','./manifest.json',f'bocconi-cluster:{dest}/'],cwd=stage,check=True)
subprocess.run(['rsync','-az',str(report/'mni_only_amendment.json'),f'bocconi-cluster:{dest}/'],check=True)
env=':'.join(str(dest/n) for n in ('source','source/scripts','source/baselines/nnUNet'))
cmd=['env','PYTHONPATH='+env,plan['conventional_python'],'-m','scripts.unetr_subfields.geo_search']
print(remote(cmd+['preflight','--stage',dest],timeout=600),flush=True)
code=r'''import sys,json
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write
from scripts.unetr_subfields import geo_search,seed_search
p=Path(sys.argv[1]); plan=geo_search.verify(p);c=read(p/'control.json'); assert c['active_datasets']==['MNI']; c['pause']=False;write(p/'control.json',c)
seed_search.summarize(p,plan,'starting')
s=read(p/'status.json'); c=read(p/'control.json'); assert all(t['id'] in c['disabled_tasks'] for t in plan['tasks'] if t['dataset']=='COBRA')
print(json.dumps(s))'''
state=json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',code,dest]))
write(report/'live.json',{'status.json':state,'ready':[],'cleanup':[]})
python=ROOT/'.venv/bin/python'
for label,args in [('collector',['-m','scripts.cross_dataset_campaign.collect','--stage',str(dest),'--local-stage',str(stage),'--destination',str(report)]),('summary',['-m','scripts.summarize_unetr_seed_search','--report',str(report),'--watch'])]:
 log=(report/('collector.log' if label=='collector' else 'summary_watcher.log')).open('a')
 p=subprocess.Popen(['caffeinate','-i',str(python),'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
 write(report/(label+'_process.json'),dict(pid=p.pid,started_at=time.time(),scope='MNI only'))
receipt=json.loads(remote(cmd+['submit','--stage',dest]));write(report/'submission.json',receipt);write(report/'mni_only_submission.json',receipt)
print(json.dumps(receipt,indent=2),flush=True)
