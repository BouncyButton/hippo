
import fcntl,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
stage=Path('/mnt/beegfsstudents/home/3160552/adni_constraints_noaug_recovery_20261013')
root=Path('/mnt/beegfsstudents/home/3160552/adni_constraints_noaug_5pct_fold0_seed0_20261013_results')
cert=json.load(sys.stdin)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,value):
 q=p.with_suffix('.json.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
assert os.getuid()==1395
with (stage/'final_condition.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 receipt=stage/'final_condition_submission.json'
 if receipt.exists():print(receipt.read_text());sys.exit(0)
 for name,digest in json.loads((stage/'manifest.json').read_text()).items():assert sha(stage/name)==digest
 queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).strip()
 assert not queue,queue
 summary=json.loads((root/'summary.json').read_text())
 assert summary['completed_runs']==55 and len(summary['runs'])==55 and summary['augmentation']=='none'
 for row in summary['runs']:
  p=root/row['model']/row['objective']/'fold0/seed0/run/completion.json'
  assert sha(p)==row['completion_sha256']
  assert json.loads(p.read_text())['status']=='complete'
 missing=root/'nnunet/dice_bands_edge_sdf_depth/fold0/seed0/run'
 assert not missing.exists()
 old=root/'_interrupted_job698935/unetr_vit_b16/dice_edge_depth/fold0/seed0/run'
 retired=old/'checkpoint_retirement.json'
 assert {x['name'] for x in cert['files']}=={'checkpoint_latest.pt','checkpoint_best.pt'}
 if not retired.exists():
  for item in cert['files']:
   p=old/item['name']
   assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
  save(retired,dict(**cert,verified_at_unix=time.time(),state='verified_for_retirement'))
 else:assert json.loads(retired.read_text())['files']==cert['files']
 for item in cert['files']:
  p=old/item['name']
  if p.exists():
   assert sha(p)==item['sha256'];p.unlink()
 save(retired,dict(**cert,retired_at_unix=time.time(),state='retired',original_metadata_preserved=True))
 time.sleep(32)
 quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
 match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
 assert match and float(match[2])-float(match[1])>=1.5,quota
 job=subprocess.check_output(['sbatch','--parsable','--job-name=hippo-adni-noaug-final',str(stage/'run.sbatch')],cwd=stage,text=True).strip().split(';')[0]
 assert job.isdigit()
 record=dict(job_id=int(job),preceding_job_id=699170,completed_runs_reused=55,new_runs=1,objective='dice_bands_edge_sdf_depth',model='nnunet',augmentation='none',fold=0,seed=0,new_replay_overhead_flops=0,released_duplicate_file_bytes=sum(x['bytes'] for x in cert['files']),quota=quota,submitted_at_unix=time.time())
 save(receipt,record);print(json.dumps(record,indent=2),flush=True)
