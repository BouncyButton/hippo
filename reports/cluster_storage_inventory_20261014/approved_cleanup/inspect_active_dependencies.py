"""Record independent active jobs and verify they do not use retirement targets."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
names=list(json.loads((BASE/'inventory.json').read_text())['targets'])+['storage_retirement_audit_20261014']
code=r"""
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path('/mnt/beegfsstudents/home/3160552')
campaign=root/'cross_dataset_exploration_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j|%Z'],text=True)
jobrows={}
for line in queue.splitlines():
 job,state,name,workdir=line.split('|',3)
 assert job in ['700232','700235'] and Path(workdir)==campaign,line
 desc=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
 assert 'Command='+str(campaign/'run.sbatch')+' ' in desc
 jobrows[job]={'workdir':workdir,'command':str(campaign/'run.sbatch'),'state':state,'description':desc}
manifest=json.loads((campaign/'manifest.json').read_text())
for rel,digest in manifest.items():assert sha(campaign/rel)==digest,rel
files=[campaign/'run.sbatch',campaign/'plan.json']+list((campaign/'protocols').rglob('*.json'))+list((campaign/'source/scripts/cross_dataset_campaign').glob('*.py'))
for rel in ['scripts/train_compute_matched_baselines.py','scripts/campaign90.py','scripts/optimization90.py','scripts/train_medsam3_few_shot.py','scripts/train_medsam3_constraint_few_shot.py','scripts/medsam3_supervised_constraints.py','evaluation/medsam3_zero_shot.py']:
 p=campaign/'source'/rel
 if p.exists():files.append(p)
for rel in ['scripts/capped_depth_delay','scripts/adni_baselines']:
 files+=list((campaign/'source'/rel).rglob('*.py'))
for p in files:
 content=p.read_text()
 assert not any(n in content for n in NAMES),str(p)
for current,dirs,names in os.walk(campaign):
 for name in dirs+names:
  p=Path(current)/name
  if p.is_symlink():assert not any(p.resolve()==root/n or root/n in p.resolve().parents for n in NAMES),str(p)
print(json.dumps({'status':'verified_independent','jobs':jobrows,'campaign':str(campaign),'manifest_sha256':sha(campaign/'manifest.json'),'checked_files':{str(p):sha(p) for p in files},'excluded_targets':NAMES}))
"""
result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster','python3 -'],input=('NAMES='+repr(names)+'\n'+code).encode())
d=json.loads(result);(BASE/'active_job_dependency_check.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(status=d['status'],jobs=list(d['jobs']),checked_files=len(d['checked_files']))))
