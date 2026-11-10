"""Remove only verified, now-unused duplicate outputs of job 735608."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
assert read(root / 'final_verification.json')['status'] == 'passed'
bundle_sha = hashlib.sha256((root / 'payload/msd_bundle.tar.gz').read_bytes()).hexdigest()
figures = read(root / 'recovered/figures/archive_manifest.json')['files']
receipts = {p.name: read(p)['sha256'] for p in (root/'receipts').glob('*.tar.gz.json')}
payload = dict(bundle_sha=bundle_sha, figures=figures, receipts=receipts)
remote = '''from pathlib import Path
import hashlib,json,subprocess,time
s=Path('/home/3160552/thesis_qualitative_predictions_20261103').resolve()
p=json.loads(PAYLOAD)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert len(p['receipts'])==7
for name,digest in p['receipts'].items():
 r=json.loads((s/'receipts'/name).read_text());assert r['status']=='verified_off_cluster' and r['sha256']==digest
for f in s.glob('worker*_complete.json'):
 r=json.loads(f.read_text());assert r=={'status':'complete','failures':[]}
assert not list((s/'runs').rglob('checkpoint_latest.pt'))
assert not list((s/'runs').rglob('*.npz'))
candidates=[(s/'msd_bundle.tar.gz',p['bundle_sha'])]+[(s/'figures'/name,digest) for name,digest in p['figures'].items()]
for path,digest in candidates:
 if path.exists():
  assert not path.is_symlink() and path.resolve().is_relative_to(s)
  assert sha(path)==digest
removed=[]
for path,digest in candidates:
 if path.exists():
  removed.append(dict(path=str(path.relative_to(s)),bytes=path.stat().st_size,sha256=digest));path.unlink()
record=dict(time=time.time(),removed=removed,scope='Only verified duplicate bundle and generated figures; original datasets, weights, source repositories and durable archives preserved.')
(s/'cleanup/final_generated_copies.json').write_text(json.dumps(record,indent=2)+'\\n')
q=subprocess.run(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True,capture_output=True)
print(json.dumps(dict(cleanup=record,quota=q.stdout,quota_error=q.stderr,quota_exit=q.returncode)))
'''.replace('PAYLOAD', repr(json.dumps(payload)))
r = subprocess.run(['ssh','bocconi-cluster','python3 -'],input=remote,text=True,capture_output=True,check=True)
record = json.loads(r.stdout)
(root / 'final_cluster_cleanup.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(removed_files=len(record['cleanup']['removed']),
                     removed_mib=sum(x['bytes'] for x in record['cleanup']['removed'])/2**20,
                     quota=record['quota'],quota_error=record['quota_error']),indent=2))
