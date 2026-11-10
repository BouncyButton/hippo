"""Preserve verified evidence before removing superseded operational run folders."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from scripts.geofwi_cluster.campaign import read, write, sha
from scripts.geofwi_constraints.prepare import ROOT
from scripts.geofwi_constraints.transfer import remote, SSH, PYTHON

NAME = 'geofwi3d_reweighted_full_suite_20261021'
REPORT = ROOT / 'reports' / NAME
REMOTE_HOME = Path('/mnt/beegfsstudents/home/3160552')
STAGES = ['geofwi3d_constraints_3fold_3seed_20261020', 'geofwi3d_constraint_strength_20261020',
          'geofwi3d_strength_confirmation_exploration_20261021']

COLLECT = r'''
import hashlib,json,sys,tarfile,time,subprocess
from pathlib import Path
home=Path(sys.argv[1]);names=json.loads(sys.argv[2]);label=sys.argv[3]
sys.path.insert(0,str(home/names[0]));import campaign as c
assert not subprocess.check_output(['squeue','-h','-u',str(__import__('os').getuid())],text=True).strip(), 'Account must be idle'
target=home/(label+'_prior_evidence.tar.gz')
files={};completed=[];deletions=[];recovery=[]
def add(path,name):
 assert path.is_file() and not path.is_symlink(), str(path)
 files[name]=dict(path=path,sha256=c.sha(path),bytes=path.stat().st_size)
for name in names:
 stage=home/name;plan=c.read(stage/'plan.json');archive=c.ClusterArchive(stage)
 add(stage/'manifest.json',name+'/source_manifest.json')
 for t in plan['tasks']:
  out=stage/'runs'/t['id'];done=(out/'result.json').exists()
  keep=t['model'] in ('swinunetr','basic_unet16') or t.get('strength')=={'unetr_vit_b16':'x4','nnunet':'x6'}.get(t['model'])
  if done:
   row=c.validate_run(stage,t);m=c.read(out/'archive_manifest.json')
   archive.verify_archive(out,m);archive.verify_archive(stage/'archives'/t['id'],m)
   assert c.read(out/'cleanup.json')['status']=='complete'
   for file in m['metadata']: add(out/file,name+'/runs/'+t['id']+'/'+file)
   for file in ('archive_manifest.json','result.json','cleanup.json','recovery_cleanup.json','archive_acknowledgment.json'):
    if (out/file).exists():add(out/file,name+'/runs/'+t['id']+'/'+file)
   completed.append(dict(row=row,source_stage=name,keep=keep,evidence=name+'/runs/'+t['id']))
  elif out.exists():
   for p in out.iterdir():
    if p.is_file() and p.suffix!='.pt' and p.name!='run.lock':add(p,name+'/runs/'+t['id']+'/'+p.name)
  if not keep and out.exists():
   deletions.append(dict(stage=name,task=t['id'],kind='completed' if done else 'interrupted' if list(out.iterdir()) else 'unstarted',path=str(out)))
   rec=stage/'recovery'/t['id']
   if rec.exists():
    for p in rec.glob('*.json'):add(p,name+'/recovery/'+t['id']+'/'+p.name)
    marker=rec/'latest.json'
    if marker.exists():
     info=c.read(marker);assert c.sha(rec/info['file'])==info['sha256']
     recovery.append(dict(task=t['id'],stage=name,marker=info,explicitly_abandoned=True))
    deletions.append(dict(stage=name,task=t['id'],kind='superseded_recovery',path=str(rec)))
# Deletion snapshots cover only generated operational directories, never durable archives.
for deletion in deletions:
 root=Path(deletion['path']);deletion['files']={}
 for p in root.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():deletion['files'][str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=c.sha(p))
document=dict(status='verified',completed=completed,deletions=deletions,abandoned_recovery=recovery,
              files={n:{k:v for k,v in info.items() if k!='path'} for n,info in files.items()},time=time.time())
blob=json.dumps(document,indent=2).encode()
import io
with tarfile.open(target,'w:gz',compresslevel=1) as tar:
 for name,info in sorted(files.items()):tar.add(info['path'],arcname=name,recursive=False)
 entry=tarfile.TarInfo('evidence_inventory.json');entry.size=len(blob);tar.addfile(entry,io.BytesIO(blob))
print(json.dumps(dict(path=str(target),sha256=c.sha(target),bytes=target.stat().st_size,
                     completed=len(completed),retained=sum(x['keep'] for x in completed),deletion_directories=len(deletions))))
'''


def collect():
    REPORT.mkdir(parents=True, exist_ok=True)
    existing = json.loads(remote(r'''import sys,json,tarfile,hashlib
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():print(json.dumps(None))
else:
 with tarfile.open(p) as tar:d=json.load(tar.extractfile('evidence_inventory.json'))
 assert d['status']=='verified'
 print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                      completed=len(d['completed']),retained=sum(x['keep'] for x in d['completed']),
                      deletion_directories=len(d['deletions']),reconciled_existing_export=True)))
''', REMOTE_HOME / (NAME + '_prior_evidence.tar.gz')))
    receipt = existing or json.loads(remote(COLLECT, REMOTE_HOME, json.dumps(STAGES), NAME))
    payload = REPORT / 'prior_evidence.tar.gz'
    with payload.open('wb') as stream:
        subprocess.run(SSH + [shlex.join(['cat', receipt['path']])], stdout=stream, check=True, timeout=300)
    assert sha(payload) == receipt['sha256']
    destination = REPORT / 'prior_evidence'
    destination.mkdir(exist_ok=True)
    with tarfile.open(payload) as tar:
        for member in tar.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        tar.extractall(destination, filter='data')
    inventory = read(destination / 'evidence_inventory.json')
    for name, info in inventory['files'].items():
        p = destination / name
        assert p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], name
    receipt.update(status='locally_verified',local_archive=str(payload),verified_files=len(inventory['files']))
    write(REPORT / 'evidence_receipt.json', receipt)
    print(json.dumps(receipt))


DELETE = r'''
import sys,json,tarfile,subprocess,shutil,time,hashlib
from pathlib import Path
home=Path(sys.argv[1]);bundle=Path(sys.argv[2]);expected=sys.argv[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(bundle)==expected
assert not subprocess.check_output(['squeue','-h','-u',str(__import__('os').getuid())],text=True).strip(), 'Do not delete while any job is active'
with tarfile.open(bundle) as tar:doc=json.load(tar.extractfile('evidence_inventory.json'))
receipt=home/(sys.argv[4]+'_superseded_cleanup.json')
if receipt.exists():
 prior=json.loads(receipt.read_text());assert prior['status']=='complete' and prior['evidence_sha256']==expected
 print(json.dumps(prior));sys.exit(0)
allowed=set(json.loads(sys.argv[5]));released=0
# Validate every exact deletion path and file digest before any mutation.
for d in doc['deletions']:
 root=Path(d['path']);assert d['stage'] in allowed
 stage=home/d['stage'];assert root.parent in (stage/'runs',stage/'recovery') and root.name==d['task']
 assert d['task'].startswith(('GeoFWI3D_unetr_vit_b16_','GeoFWI3D_nnunet_'))
 assert not root.is_symlink() and root.is_dir()
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 assert actual==set(d['files'])
 for name,info in d['files'].items():
  p=root/name;assert not p.is_symlink() and p.stat().st_size==info['bytes'] and sha(p)==info['sha256']
  released+=info['bytes']
result=dict(status='in_progress',evidence_sha256=expected,local_archive_verified=True,
            authorization='User explicitly requested deletion/replacement of old UNETR and nnU-Net constrained runs',
            directories=doc['deletions'],released_bytes=released,time=time.time())
receipt.write_text(json.dumps(result,indent=2))
for d in doc['deletions']:shutil.rmtree(d['path'])
for name in allowed:
 stage=home/name;(stage/'STOP').touch()
 marker=dict(status='superseded',replacement=sys.argv[4],retained_durable_archives=True,
             deleted_run_ids=[d['task'] for d in doc['deletions'] if d['stage']==name and Path(d['path']).parent.name=='runs'])
 (stage/'SUPERSEDED.json').write_text(json.dumps(marker,indent=2))
result['status']='complete';receipt.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


def cleanup():
    receipt = read(REPORT / 'evidence_receipt.json')
    assert receipt['status'] == 'locally_verified' and sha(Path(receipt['local_archive'])) == receipt['sha256']
    result = json.loads(remote(DELETE, REMOTE_HOME, receipt['path'], receipt['sha256'], NAME, json.dumps(STAGES)))
    write(REPORT / 'superseded_cleanup.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='directories'}))
    print('Removed directories:',len(result['directories']))


if __name__ == '__main__':
    import sys
    cleanup() if '--cleanup' in sys.argv else collect()
