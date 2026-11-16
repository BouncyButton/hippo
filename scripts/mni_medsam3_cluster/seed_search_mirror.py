"""Optional off-BeeGFS archive for dynamically frozen candidate stages."""
import os
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import traceback

from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_medsam3_cluster.constraint_mirror import snapshot

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
     '-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/hippo-mni-search-%C',os.environ.get('HIPPO_CLUSTER_HOST', 'hippo-cluster')]


def remote(arguments):
    return subprocess.check_output([*SSH,shlex.join([str(v) for v in arguments])],timeout=180)


def upload_json(value,path):
    code='import sys,os;from pathlib import Path;p=Path(sys.argv[1]);t=p.with_name(p.name+".tmp");t.write_text(sys.stdin.read());os.replace(t,p)'
    subprocess.run([*SSH,shlex.join(['python3','-c',code,str(path)])],input=json.dumps(value).encode(),check=True,timeout=180)


def frozen_stage(stage,local,expected):
    if local.exists():
        if digest(local/'manifest.json')!=expected:raise ValueError('Local frozen source identity differs')
        return
    manifest_bytes=remote(['cat',stage/'manifest.json']);manifest=json.loads(manifest_bytes)
    import hashlib
    if hashlib.sha256(manifest_bytes).hexdigest()!=expected:raise ValueError('Registered source differs')
    names=list(manifest)+['manifest.json']
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe manifest')
    blob=remote(['tar','--dereference','-czf','-','-C',stage,*names])
    local.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
        if set(archive.getnames())!=set(names) or any(not m.isfile() for m in archive.getmembers()):
            raise ValueError('Frozen archive inventory differs')
        archive.extractall(local,filter='data')
    if (local/'manifest.json').read_bytes()!=manifest_bytes:raise ValueError('Frozen manifest changed')
    for name,sha in manifest.items():
        if digest(local/name)!=sha:raise ValueError('Frozen source archive differs: '+name)


def serve(stage,destination,once=False):
    destination.mkdir(parents=True,exist_ok=True)
    collect.remote=remote;collect.upload_json=upload_json;collect.snapshot=snapshot
    collect.resume_after_archive=lambda *args:None
    code='''import json,sys
from pathlib import Path
p=Path(sys.argv[1]);print(json.dumps({n:json.loads((p/n).read_text()) for n in ('status.json','state.json','registry.json','service_error.json','final_analysis.json','service_start.json') if (p/n).exists()}))'''
    while True:
        try:
            state=json.loads(remote(['python3','-c',code,stage]))
            write(destination/'latest.json',state)
            for entry in state.get('registry.json',{}).get('stages',[]):
                child=Path(entry['path']);relative=child.relative_to(stage)
                local=destination/'frozen_candidates'/relative
                frozen_stage(child,local,entry['manifest_sha256'])
                collect.serve(child,local,destination/'candidates'/relative,once=True)
                extra=json.loads(remote(['python3','-c',code,child]))
                write(destination/'candidates'/relative/'reports.json',extra)
            if 'final_analysis.json' in state:
                write(destination/'final_analysis.json',state['final_analysis.json'])
                (destination/'final_cells.csv').write_bytes(remote(['cat',stage/'final_cells.csv']))
            if once:return
        except Exception:
            write(destination/'mirror_error.json',dict(time=time.time(),traceback=traceback.format_exc()))
            if once:raise
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();serve(args.stage,args.destination,args.once)


if __name__=='__main__':main()
