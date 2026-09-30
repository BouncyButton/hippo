"""Retire exactly two remote weights only after verified local archival."""
from pathlib import Path
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parent
items=json.loads((root/'ALLOWLIST.json').read_text())
expected={'pcgrad_edge_20260929_01/fold0/seed0/pcgrad/checkpoint_best.pt', 'separated_edge_20260929_01/fold0/seed0/separated/checkpoint_best.pt'}
assert {r['relative_path'] for r in items}==expected and len(items)==2
for r in items:
 p=root/'checkpoints'/r['relative_path']
 assert p.is_file() and p.stat().st_size>70_000_000
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 r['archived_bytes']=p.stat().st_size
(root/'LOCAL_VERIFICATION.json').write_text(json.dumps(items,indent=2)+'\n')
code='''from pathlib import Path
import hashlib,json,datetime
base=Path('/mnt/beegfsstudents/home/3160552')
items=ITEMS
protected=[base/'pcgrad_50cases_seed0_20260929_01/fold0/seed0/pcgrad/checkpoint_best.pt',base/'pcgrad_50cases_seed0_20260929_01/fold0/seed0/pcgrad/checkpoint_latest.pt']
protected += [base/f'separated_edge_20260929_01/fold0/seed{s}/pooled/checkpoint_best.pt' for s in range(3)]
protected += [base/f'pcgrad_edge_20260929_01/fold0/seed{s}/sum/checkpoint_best.pt' for s in range(3)]
assert all(p.is_file() for p in protected)
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
for r in items:
 p=base/r['relative_path']
 assert p.is_file() and not p.is_symlink() and p.resolve()==p
 assert p.stat().st_size==r['archived_bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
for r in items:
 (base/r['relative_path']).unlink()
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
assert before==after
print(json.dumps(dict(status='archived_locally_then_remote_copy_removed',files=items,protected_checkpoint_hashes=after,medsam3_touched=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''.replace('ITEMS',repr(items))
result=subprocess.run(['ssh','bocconi-cluster','python3','-'],input=code,text=True,capture_output=True,check=True)
receipt=json.loads(result.stdout)
(root/'REMOTE_REMOVAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],file_count=len(items),logical_bytes_archived=sum(r['archived_bytes'] for r in items),medsam3_touched=False)))
