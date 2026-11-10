"""Verify a completed performance benchmark locally, then retire scratch models."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from scripts.cross_dataset_campaign.common import REMOTE,read,write,digest
from scripts.cross_dataset_campaign.msd_mixed_losses_job import ROOT,NAME


def remote(code,*args,input=None):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',
        shlex.join(['python3','-c',code,*map(str,args)])],input=input)


def collect(name,superseded=False):
    stage=REMOTE/name;destination=ROOT/'reports'/name
    parent=destination/'parent_frozen'
    if not parent.exists():shutil.copytree(ROOT/'experiments'/NAME,parent)
    if digest(parent/'manifest.json')!=read(destination/'manifest.json')['parent_manifest_sha256']:
        raise ValueError('Parent source manifest differs')
    for relative,sha in read(parent/'manifest.json').items():
        if digest(parent/relative)!=sha:raise ValueError('Parent source hash mismatch')
    if superseded:
        successor=read(ROOT/'reports/msd_runtime_benchmark_v2_20261014/runs/summary.json')
        if successor['status']!='passed':raise ValueError('Keep first-attempt comparison checkpoints until the corrected benchmark passes')
    else:
        summary=json.loads(remote('import sys;from pathlib import Path;print((Path(sys.argv[1])/"runs/summary.json").read_text())',stage))
        if summary['status']!='passed':raise ValueError('Benchmark is not complete')
    code='''import sys,json,hashlib,subprocess
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
p=Path(sys.argv[1]);job=json.loads((p/'submission.json').read_text())['job_id']
if str(job) in subprocess.check_output(['squeue','-h','-u','3160552','-o','%i'],text=True).split():raise ValueError('Benchmark still active')
metadata={};bulky={}
for f in sorted(p.rglob('*')):
 if not f.is_file() or f.name in ('archive_manifest.json','archive_acknowledgment.json','cleanup.json'):continue
 if f.is_symlink():raise ValueError('Unexpected symlink')
 record=dict(bytes=f.stat().st_size,sha256=sha(f))
 if f.suffix=='.pt':bulky[str(f.relative_to(p))]=record
 elif f.suffix in ('.json','.out','.err','.sbatch','.csv','.jsonl','.log','.lock','.png'):metadata[str(f.relative_to(p))]=record
 else:raise ValueError('Unexpected artifact: '+str(f))
print(json.dumps(dict(metadata=metadata,bulky=bulky,job_id=job)))
'''
    manifest=json.loads(remote(code,stage));write(destination/'archive_manifest.json',manifest)
    names=list(manifest['metadata'])
    blob=subprocess.check_output(['ssh','bocconi-cluster',shlex.join(['tar','-czf','-','-C',str(stage),*names])])
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
        if set(tar.getnames())!=set(names) or any(not m.isfile() for m in tar.getmembers()):raise ValueError('Archive inventory mismatch')
        tar.extractall(destination,filter='data')
    for relative,item in manifest['metadata'].items():
        path=destination/relative
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('Archive hash mismatch')
        if path.suffix=='.json':read(path)
        elif path.suffix=='.jsonl':
            for line in path.read_text().splitlines():json.loads(line)
        elif path.suffix=='.csv':
            import csv
            with path.open() as stream:list(csv.DictReader(stream))
        elif path.suffix!='.png':path.read_text()
    for relative,sha in read(destination/'manifest.json')['additional_sources'].items():
        if digest(destination/'source'/relative)!=sha:raise ValueError('Frozen benchmark source mismatch')
    ack=dict(status='verified',durable_archive=str(destination),verified_at=time.time(),
        archive_manifest_sha256=digest(destination/'archive_manifest.json'),
        superseded_input_fixture_attempt=superseded,comparison_dependency_complete=True)
    write(destination/'archive_acknowledgment.json',ack)
    code='''import sys,json,hashlib,subprocess,time
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
p=Path(sys.argv[1]);request=json.load(sys.stdin);m=request['manifest'];ack=request['ack']
if str(m['job_id']) in subprocess.check_output(['squeue','-h','-u','3160552','-o','%i'],text=True).split():raise ValueError('Benchmark restarted')
for group in ('metadata','bulky'):
 for name,item in m[group].items():
  f=p/name
  if f.is_symlink() or not f.resolve().is_relative_to(p.resolve()):raise ValueError('Unsafe path')
  if f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']:raise ValueError('Artifact changed')
(p/'archive_manifest.json').write_text(json.dumps(m,indent=2)+'\\n')
(p/'archive_acknowledgment.json').write_text(json.dumps(ack,indent=2)+'\\n')
receipt=dict(status='in_progress',deleted=m['bulky'],released_bytes=sum(x['bytes'] for x in m['bulky'].values()),time=time.time())
(p/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\\n')
for name in m['bulky']:
 if Path(name).suffix!='.pt' or not name.startswith('runs/'):raise ValueError('Deletion outside generated checkpoints')
 (p/name).unlink()
receipt['status']='complete';(p/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
    receipt=json.loads(remote(code,stage,input=json.dumps(dict(manifest=manifest,ack=ack)).encode()))
    write(destination/'cleanup.json',receipt)
    print(dict(name=name,status='verified_and_cleaned',released_gib=receipt['released_bytes']/2**30))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--superseded',action='store_true')
    args=parser.parse_args()
    if not args.name.startswith('msd_runtime_benchmark_') or '/' in args.name:raise ValueError('Invalid benchmark name')
    collect(args.name,args.superseded)
