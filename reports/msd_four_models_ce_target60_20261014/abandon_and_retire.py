"""One-off archive/retirement of the explicitly abandoned MSD campaign."""
import io,json,subprocess,shlex,tarfile,time,hashlib
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.collect import remote,upload_json
from scripts.cross_dataset_campaign.storage import verify_archive
stage=Path('/mnt/beegfsstudents/home/3160552/msd_four_models_ce_target60_20261014')
dest=Path('/Users/filippofocaccia/Desktop/hippo/reports/msd_four_models_ce_target60_20261014')
code="""import json,sys,hashlib,subprocess
from pathlib import Path
def sha_file(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
p=Path(sys.argv[1]); c=json.loads((p/'control.json').read_text())
assert c['pause'] and c['abandoned']
own={f.stem for f in (p/'submissions').glob('*.json')}
queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i'],text=True).split()
assert not own.intersection(queue),'Campaign is still active'
metadata={};bulky=[]
for f in sorted((p/'runs').rglob('*')):
 if not f.is_file():continue
 assert not f.is_symlink()
 name=str(f.relative_to(p));info=dict(bytes=f.stat().st_size,sha256=sha_file(f))
 if f.suffix in ('.pt','.tmp') or f.name.endswith('.nii.gz'):bulky.append(dict(name=name,**info))
 elif f.suffix in ('.json','.csv','.jsonl','.log','.lock','.png'):metadata[name]=info
 else:raise ValueError(str(f))
for f in list(p.glob('slurm-*'))+[p/'status.json',p/'control.json',p/'submission.json',p/'successor.json']:
 if f.is_file():metadata[str(f.relative_to(p))]=dict(bytes=f.stat().st_size,sha256=sha_file(f))
print(json.dumps(dict(metadata=metadata,deletion_candidates=bulky,authorization='Explicit user abandonment of MSD sweep on 2026-10-14',partial_results='Incomplete UNETR SDF run is abandoned, not a completed-budget comparison')))
"""
manifest=json.loads(remote(['python3','-c',code,stage]))
write(dest/'abandonment_manifest.json',manifest)
names=list(manifest['metadata'])
blob=remote(['tar','-czf','-','-C',stage,*names])
target=dest/'abandoned_campaign_snapshot';target.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
 assert set(archive.getnames())==set(names) and all(m.isfile() for m in archive.getmembers())
 archive.extractall(target,filter='data')
# Existing verifier accepts additional suffixes as UTF-8 and validates every JSON/CSV.
verify_archive(target,manifest)
for folder in (target/'runs').iterdir():
 if (folder/'completion.json').exists():
  verify_archive(folder,read(folder/'archive_manifest.json'))
  original=dest/'runs'/folder.name
  verify_archive(original,read(original/'archive_manifest.json'))
write(target/'abandonment_manifest.json',manifest)
ack=dict(status='verified',source_verified=True,durable_archive=str(target),manifest_sha256=digest(target/'abandonment_manifest.json'),canonical_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),verified_at=time.time())
write(target/'abandonment_acknowledgment.json',ack)
upload_json(manifest,stage/'abandonment_manifest.json');upload_json(ack,stage/'abandonment_acknowledgment.json')
code="""import json,sys,time,hashlib,subprocess
from pathlib import Path
def sha_file(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
p=Path(sys.argv[1]);m=json.loads((p/'abandonment_manifest.json').read_text());ack=json.loads((p/'abandonment_acknowledgment.json').read_text())
assert json.loads((p/'control.json').read_text())['abandoned']
assert ack['source_verified'] and ack['status']=='verified' and ack['durable_archive'].startswith('/Users/filippofocaccia/Desktop/hippo/reports/')
assert hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()==ack['canonical_manifest_sha256']
own={f.stem for f in (p/'submissions').glob('*.json')};assert not own.intersection(subprocess.check_output(['squeue','-h','-u','3160552','-o','%i'],text=True).split())
for name,info in m['metadata'].items():
 f=p/name;assert f.stat().st_size==info['bytes'] and sha_file(f)==info['sha256']
for info in m['deletion_candidates']:
 f=p/info['name'];assert f.resolve().is_relative_to((p/'runs').resolve()) and not f.is_symlink() and f.suffix in ('.pt','.tmp')
 assert f.stat().st_size==info['bytes'] and sha_file(f)==info['sha256']
receipt=dict(status='in_progress',deleted=m['deletion_candidates'],released_file_bytes=sum(v['bytes'] for v in m['deletion_candidates']),durable_archive=ack['durable_archive'],time=time.time())
(p/'abandonment_cleanup.json').write_text(json.dumps(receipt,indent=2))
for info in m['deletion_candidates']:(p/info['name']).unlink()
receipt['status']='complete';(p/'abandonment_cleanup.json').write_text(json.dumps(receipt,indent=2))
s=json.loads((p/'status.json').read_text());s.update(status='paused',disposition='abandoned_by_user',reason='Replaced by four transformer bands10/20 runs',updated_at=time.time());(p/'status.json').write_text(json.dumps(s,indent=2))
print(json.dumps(receipt))
"""
receipt=json.loads(remote(['python3','-c',code,stage]));write(dest/'abandonment_cleanup.json',receipt)
print(json.dumps(receipt,indent=2))
