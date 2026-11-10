"""Copy completed exports off BeeGFS, verify members, then acknowledge cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parent
STAGE='/mnt/beegfsstudents/home/3160552/thesis_qualitative_predictions_20261103'
PYTHON='/mnt/beegfsstudents/home/3160552/medsam3_runtime_20260928/venv/bin/python'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def ssh(command,**kwargs):
    return subprocess.run(['rtk','proxy','ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',command],check=True,**kwargs)

def collect():
    script="from pathlib import Path;import json;p=Path("+repr(STAGE)+");print(json.dumps({'ready':[json.loads(f.read_text()) for f in (p/'exports').glob('*.ready.json')],'workers':[json.loads(f.read_text()) for f in p.glob('worker*_complete.json')]}))"
    state=json.loads(ssh('python3 -c '+shlex.quote(script),capture_output=True,text=True).stdout)
    for item in sorted(state['ready'],key=lambda x:x['file']=='figures.tar.gz'):
        name=item['file'];assert Path(name).name==name
        receipt=ROOT/'receipts'/(name+'.json')
        if receipt.exists():continue
        archive=ROOT/'archives'/name;archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            tmp=archive.with_suffix('.partial')
            with tmp.open('wb') as f:ssh('cat '+shlex.quote(STAGE+'/exports/'+name),stdout=f)
            assert sha(tmp)==item['sha256'];tmp.replace(archive)
        assert sha(archive)==item['sha256']
        with tarfile.open(archive) as tar:
            members=tar.getmembers()
            prefix='figures' if name=='figures.tar.gz' else item['run']
            for member in members:
                assert member.isfile() and member.name.startswith(prefix+'/') and '..' not in Path(member.name).parts
            manifest=json.load(tar.extractfile(prefix+'/archive_manifest.json'))
            assert set(m.name for m in members)=={prefix+'/'+r for r in manifest['files']}|{prefix+'/archive_manifest.json'}
            for rel,expected in manifest['files'].items():
                h=hashlib.sha256()
                with tar.extractfile(prefix+'/'+rel) as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
                assert h.hexdigest()==expected,rel
            dest=ROOT/'recovered';dest.mkdir(exist_ok=True)
            # Members have been restricted to regular files under one prefix.
            tar.extractall(dest)
            assert all(sha(dest/prefix/rel)==expected for rel,expected in manifest['files'].items())
        value=dict(status='verified_off_cluster',file=name,sha256=item['sha256'],local_archive=str(archive),verified_members=len(manifest['files']),time=time.time())
        command=' '.join(shlex.quote(x) for x in [PYTHON,STAGE+'/driver.py','ack','--stage',STAGE,'--receipt-json',json.dumps(value)])
        ssh(command,capture_output=True,text=True)
        write(receipt,value);print('ARCHIVED AND ACKNOWLEDGED',name,flush=True)
    write(ROOT/'collector_status.json',dict(time=time.time(),**state))
    done=(ROOT/'receipts/figures.tar.gz.json').exists()
    failed=len(state['workers'])==2 and any(w['status']=='failed' for w in state['workers'])
    return done or failed

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    deadline=time.time()+30*3600
    while time.time()<deadline:
        try:
            if collect() or args.once:break
        except Exception as exc:
            write(ROOT/'collector_error.json',dict(time=time.time(),error=repr(exc)))
            print('Collector retry:',repr(exc),flush=True)
            if args.once:raise
        time.sleep(60)
