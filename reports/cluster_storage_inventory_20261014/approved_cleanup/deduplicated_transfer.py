"""Resume complete result preservation using content-addressed transfer."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parent
ACTION = sys.argv[1]
PART = int(sys.argv[2]) if len(sys.argv)>2 else None
PARTS = int(sys.argv[3]) if len(sys.argv)>3 else None
sys.argv = [str(BASE/'archive_cleanup.py'),'archive_results']
spec=importlib.util.spec_from_file_location('archive_worker',BASE/'archive_cleanup.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
CACHE=BASE/'transfer_blobs';CACHE.mkdir(exist_ok=True)

def sha(data):return hashlib.sha256(data).hexdigest()
def remote(code,out):
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bocconi-cluster','python3 -'],input=code.encode(),stdout=out,check=True)

def plan():
    config=dict(root=a.ROOT,targets=a.TARGETS,mode='inspect')
    code='CONFIG='+repr(config)+'\n'+a.REMOTE+r'''
result={'root':str(ROOT),'timestamp_utc':now(),'targets':{},'files':{}}
for name in TARGETS:
    before=snap(name)
    for rel,d in sorted(before.items()):
        if not stat.S_ISREG(d['mode']) or Path(rel).suffix in {'.pt','.pth','.ckpt','.safetensors'}:continue
        p=ROOT/name/rel;h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);n+=len(b)
        after=meta(p)
        assert n==d['size'] and all(after[k]==d[k] for k in ('mode','size','inode','device','uid'))
        assert abs(after['mtime_ns']-d['mtime_ns'])<=1000000000
        result['files'][name+'/'+rel]={'bytes':n,'sha256':h.hexdigest()}
    result['targets'][name]=snap(name)
print(json.dumps(result))
'''
    with (BASE/'content_manifest.partial.json').open('wb') as f:remote(code,f)
    d=json.loads((BASE/'content_manifest.partial.json').read_text())
    inv=json.loads((BASE/'inventory.json').read_text());diffs={}
    for name,entries in d['targets'].items():
        assert set(entries)==set(inv['targets'][name])
        for rel,new in entries.items():
            old=inv['targets'][name][rel]
            assert all(new[k]==old[k] for k in ('mode','size','inode','device','uid')),(name,rel)
            assert abs(new['mtime_ns']-old['mtime_ns'])<=1000000000,(name,rel)
            if new!=old:diffs[name+'/'+rel]={'before':old,'after':new}
    (BASE/'content_manifest.partial.json').rename(BASE/'content_manifest.json')
    (BASE/'content_metadata_refresh.json').write_text(json.dumps(diffs,indent=2))
    inv['targets']=d['targets'];(BASE/'inventory.json').write_text(json.dumps(inv)+'\n')
    unique={v['sha256']:v['bytes'] for v in d['files'].values()}
    print(json.dumps(dict(files=len(d['files']),unique_files=len(unique),total_gib=sum(v['bytes'] for v in d['files'].values())/2**30,unique_gib=sum(unique.values())/2**30)),flush=True)

def recover(archives=None):
    manifest=json.loads((BASE/'content_manifest.json').read_text());wanted={v['sha256']:v['bytes'] for v in manifest['files'].values()}
    if archives is None:
        archives=list((BASE/'results_archives').glob('*.tar.gz*'))+list((BASE/'parallel_results_archives').glob('*.tar.gz*'))+list(BASE.glob('missing_blobs*.tar.gz.partial'))
    for archive in archives:
        recovered=0
        try:
            with tarfile.open(archive,mode='r|gz') as tar:
                for item in tar:
                    if not item.isfile() or item.name=='ARCHIVE_MANIFEST.json':continue
                    data=tar.extractfile(item).read()
                    if len(data)!=item.size:break
                    digest=sha(data)
                    if digest in wanted and len(data)==wanted[digest] and not (CACHE/digest).exists():
                        tmp=CACHE/(digest+'.'+str(os.getpid())+'.partial');tmp.write_bytes(data);tmp.replace(CACHE/digest);recovered+=1
        except (EOFError,tarfile.ReadError,OSError):pass
        print(json.dumps(dict(recovered_from=archive.name,new_blobs=recovered)),flush=True)
    missing={v['sha256']:v['bytes'] for v in manifest['files'].values() if not (CACHE/v['sha256']).exists()}
    print(json.dumps(dict(missing_unique_files=len(missing),missing_gib=sum(missing.values())/2**30)),flush=True)

def download():
    manifest=json.loads((BASE/'content_manifest.json').read_text());missing={}
    for path,record in manifest['files'].items():
        selected=PART is None or int(record['sha256'],16)%PARTS==PART
        if selected and not (CACHE/record['sha256']).exists():missing.setdefault(record['sha256'],dict(path=path,**record))
    if not missing:return
    config=dict(root=a.ROOT,targets=a.TARGETS,mode='inspect',missing=missing)
    code='CONFIG='+repr(config)+'\n'+a.REMOTE+r'''
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
    for digest,row in CONFIG['missing'].items():
        p=ROOT/row['path'];assert p.relative_to(ROOT).parts[0] in TARGETS
        data=p.read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==digest,str(p)
        info=tarfile.TarInfo(digest);info.size=len(data);info.mode=0o600
        tar.addfile(info,io.BytesIO(data))
'''
    print(json.dumps(dict(event='download_missing',unique_files=len(missing),gib=sum(x['bytes'] for x in missing.values())/2**30)),flush=True)
    path=BASE/('missing_blobs.tar.gz.partial' if PART is None else f'missing_blobs_{PART}of{PARTS}.tar.gz.partial')
    with path.open('wb') as f:remote(code,f)
    recover([path])
    assert all((CACHE/r['sha256']).exists() for r in missing.values())

def assemble():
    manifest=json.loads((BASE/'content_manifest.json').read_text())
    for name,entries in manifest['targets'].items():
        files={p:r for p,r in manifest['files'].items() if p.startswith(name+'/')}
        missing=[r for r in files.values() if not (CACHE/r['sha256']).exists()]
        if missing and ACTION=='assemble_ready':
            print(json.dumps(dict(waiting=name,missing_files=len(missing))),flush=True);continue
        excludes=[rel for rel,d in entries.items() if stat.S_ISREG(d['mode']) and name+'/'+rel not in files]
        assert all(Path(x).suffix in {'.pt','.pth','.ckpt','.safetensors'} for x in excludes)
        final=BASE/'results_archives'/(name+'.tar.gz');tmp=final.with_suffix('.assembled')
        with tarfile.open(tmp,mode='w:gz',compresslevel=1) as tar:
            for rel,d in sorted(entries.items()):
                if rel in excludes:continue
                path=name if rel=='.' else name+'/'+rel
                info=tarfile.TarInfo(path);info.mode=stat.S_IMODE(d['mode']);info.mtime=d['mtime_ns']/1e9
                if stat.S_ISDIR(d['mode']):info.type=tarfile.DIRTYPE;tar.addfile(info)
                elif stat.S_ISLNK(d['mode']):info.type=tarfile.SYMTYPE;info.linkname=d['link_target'];tar.addfile(info)
                else:
                    record=files[path];blob=CACHE/record['sha256']
                    assert blob.stat().st_size==record['bytes'] and a.digest_file(blob)==record['sha256']
                    info.size=record['bytes']
                    with blob.open('rb') as f:tar.addfile(info,f)
            data=json.dumps(dict(target=name,root=a.ROOT,timestamp_utc=manifest['timestamp_utc'],entries=entries,files=files,excluded_checkpoint_paths=excludes,transfer='Content-addressed transfer; every original non-checkpoint file stored independently with its original contents.')).encode()
            info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
        tmp.replace(final)
        print(json.dumps(dict(assembled=name,archive_mib=final.stat().st_size/2**20)),flush=True)

{'plan':plan,'recover':recover,'recover_downloads':lambda:recover(list(BASE.glob('missing_blobs_*of3.tar.gz.partial'))),'download':download,'assemble':assemble,'assemble_ready':assemble}[ACTION]()
