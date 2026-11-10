
import fcntl,hashlib,json,os,sys,time
from pathlib import Path
root=Path('/mnt/beegfsstudents/home/3160552/adni_constraints_noaug_5pct_fold0_seed0_20261013_results')
inv=json.load(sys.stdin)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
assert os.getuid()==1395
with (root/'user_checkpoint_retirement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 receipt=root/'user_checkpoint_retirement.json'
 if receipt.exists() and json.loads(receipt.read_text())['status']=='complete':
  print(receipt.read_text());sys.exit(0)
 assert digest(root/'summary.json')=='5e036959a5e679cc9dc0c2d4609710ad4be77e54475f01352ed552a15af16e1c'
 summary=json.loads((root/'summary.json').read_text())
 assert summary['status']=='complete' and summary['completed_runs']==56
 expected={x['path'] for x in inv['files']}
 assert {str(p.relative_to(root)) for p in root.rglob('*.pt')}==expected
 for item in inv['files']:
  p=root/item['path']
  assert p.resolve().is_relative_to(root) and not p.is_symlink()
  assert p.name=='checkpoint_best.pt' and p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
  c=json.loads((p.parent/'completion.json').read_text())
  assert c['status']=='complete' and c['checkpoint_best_sha256']==item['sha256']
 record=dict(status='verified_for_deletion',reason='User explicitly requested deleting artifacts from the completed ADNI sweep',files=inv['files'],released_bytes=inv['total_bytes'],metrics_configs_calibration_and_source_preserved=True)
 receipt.write_text(json.dumps(record,indent=2)+'\n')
 for item in inv['files']:(root/item['path']).unlink()
 record.update(status='complete',finished_at_unix=time.time())
 receipt.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
