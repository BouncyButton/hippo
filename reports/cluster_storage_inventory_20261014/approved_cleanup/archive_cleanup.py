"""Archive explicitly selected experiment directories before verified removal.

Only the delete subcommand modifies remote storage. Result archives preserve every
non-checkpoint regular file, directory, and symbolic link; excluded weights are inventoried.
"""
import datetime
import base64
import zlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = '/mnt/beegfsstudents/home/3160552'
GROUPS = {
    '1': ['geofwi3d_swinunetr_5pct_four_losses_500updates_seed0_20261005_01',
          'geofwi3d_swinunetr_20261004_01',
          'geofwi3d_swinunetr_5pct_500updates_seed0_20261004_01',
          'geofwi3d_swinunetr_5pct_bands_500updates_seed0_20261004_01'],
    '2': ['outer_combo_fold0_seed0_5pct_20261012_01'],
    '3': ['budget_bands_94603_684431', 'budget_baselines_94603_684354'],
    '4': ['depth_swinunetr_fold0_seed0_20261006_01', 'depth_weight_variants_fold0_seed0_20261006_01'],
    '5': ['band_focus_fold0_seed0_5pct_20261011_02',
          'band_evolution_fold0_seed0_5pct_20261010_01',
          'native_branch_constraints_fold0_seed0_5pct_20261008_01',
          'band_chamfer_fold0_seed0_5pct_20261008_01',
          'band_focus_fold0_seed0_5pct_20261011_01',
          'band_evolution_fold0_seed0_5pct_20261009_01'],
    '6': ['capped_depth_delay_31976_687658'],
}
TARGETS = [name for names in GROUPS.values() for name in names]
RESULTS_ONLY = sys.argv[1].endswith('_results')
ARCHIVE_DIR = 'results_archives' if RESULTS_ONLY else 'archives'
VERIFY_NAME = 'results_verification.json' if RESULTS_ONLY else 'verification.json'

REMOTE = r'''
import datetime, hashlib, io, json, os, stat, subprocess, sys, tarfile
from pathlib import Path
ROOT = Path(CONFIG['root'])
TARGETS = CONFIG['targets']
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def jobs():
    raw = subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j|%Z'], text=True)
    if raw.strip():
        guard=CONFIG.get('independent_job_guard')
        if not guard or guard['status']!='verified_independent':
            raise RuntimeError('Jobs need dependency inspection: '+raw)
        campaign=Path(guard['campaign'])
        assert hashlib.sha256((campaign/'manifest.json').read_bytes()).hexdigest()==guard['manifest_sha256']
        for path,digest in guard['checked_files'].items():
            p=Path(path)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, 'Active dependency changed: '+path
            assert not any(p.resolve()==ROOT/n or ROOT/n in p.resolve().parents for n in TARGETS)
        for current,dirs,files in os.walk(campaign):
            for filename in dirs+files:
                p=Path(current)/filename
                if p.is_symlink():
                    assert not any(p.resolve()==ROOT/n or ROOT/n in p.resolve().parents for n in TARGETS)
        for line in raw.splitlines():
            job,state,name,workdir=line.split('|',3)
            assert job in guard['jobs'], 'New job needs dependency inspection: '+line
            expected=guard['jobs'][job]
            assert workdir==expected['workdir'], 'Job working directory changed'
            description=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
            assert 'Command='+expected['command']+' ' in description, 'Job entrypoint changed'
    return raw
def meta(p):
    s=p.lstat()
    d=dict(mode=s.st_mode,size=s.st_size,blocks=s.st_blocks,mtime_ns=s.st_mtime_ns,
           inode=s.st_ino,device=s.st_dev,uid=s.st_uid)
    if stat.S_ISLNK(s.st_mode): d['link_target']=os.readlink(p)
    return d
def snap(name):
    if name not in TARGETS or '/' in name or name.startswith('.'):
        raise RuntimeError('Unapproved target '+name)
    p=ROOT/name
    if not os.path.lexists(p): return None
    if p.is_symlink() or not p.is_dir(): raise RuntimeError('Not a direct directory '+str(p))
    entries={'.':meta(p)}
    for current, dirs, files in os.walk(p, followlinks=False):
        if '.git' in dirs or '.git' in files:
            raise RuntimeError('Source repository must be preserved: '+current)
        for n in sorted(dirs+files):
            q=Path(current)/n; d=meta(q)
            if d['uid'] != os.getuid(): raise RuntimeError('Foreign-owned file '+str(q))
            if d['device'] != entries['.']['device']: raise RuntimeError('Nested mount '+str(q))
            if not any(f(d['mode']) for f in [stat.S_ISREG,stat.S_ISDIR,stat.S_ISLNK]):
                raise RuntimeError('Unsupported special file '+str(q))
            entries[str(q.relative_to(p))]=d
    return entries
def verify_snapshot(name):
    current=snap(name);expected=CONFIG['expected'][name]
    if current==expected:return
    if current is None or expected is None or set(current)!=set(expected):
        raise RuntimeError('Target entries changed before deletion: '+name)
    for rel,original in expected.items():
        actual=current[rel]
        if actual==original:continue
        correction=CONFIG.get('metadata_timestamp_corrections',{}).get(name+'/'+rel)
        if not correction:raise RuntimeError('Target changed before deletion: '+name+'/'+rel)
        assert Path(rel).suffix in {'.pt','.pth','.ckpt','.safetensors'}
        assert {k:v for k,v in actual.items() if k!='mtime_ns'}=={k:v for k,v in original.items() if k!='mtime_ns'}
        assert actual['mtime_ns'] in correction['allowed_mtime_ns'] and original['mtime_ns'] in correction['allowed_mtime_ns']
        h=hashlib.sha256()
        with (ROOT/name/rel).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        assert h.hexdigest()==correction['sha256'], 'Checkpoint content differs from archived evaluation: '+rel
class HashedReader:
    def __init__(self, f): self.f=f; self.h=hashlib.sha256(); self.n=0
    def read(self,n=-1):
        b=self.f.read(n); self.h.update(b); self.n+=len(b); return b
mode=CONFIG['mode']
if mode=='inventory':
    queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j|%Z'],text=True)
    data=dict(root=str(ROOT),timestamp_utc=now(),active_jobs=queue.splitlines(),targets={})
    for name in TARGETS: data['targets'][name]=snap(name)
    print(json.dumps(data))
elif mode=='archive':
    # Read-only archival uses before/after snapshots; an independent evaluation
    # may read these files while writing diagnostics outside the selected roots.
    name=CONFIG['target']; before=snap(name)
    if before is None: raise RuntimeError('Target disappeared before archival')
    if before != CONFIG['expected']: raise RuntimeError('Target changed after inventory: '+name)
    excluded = {rel for rel,d in before.items() if stat.S_ISREG(d['mode']) and Path(rel).suffix in {'.pt','.pth','.ckpt','.safetensors'}} if CONFIG.get('results_only') else set()
    manifest=dict(target=name,root=str(ROOT),timestamp_utc=now(),entries=before,files={},excluded_checkpoint_paths=sorted(excluded))
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
        for rel,d in sorted(before.items()):
            if rel in excluded: continue
            p=ROOT/name if rel=='.' else ROOT/name/rel
            arcname=name if rel=='.' else name+'/'+rel
            info=tar.gettarinfo(str(p),arcname=arcname)
            if stat.S_ISREG(d['mode']):
                # Store hardlinked paths independently for complete per-path verification.
                info.type=tarfile.REGTYPE; info.linkname=''; info.size=d['size']
                with p.open('rb') as f:
                    reader=HashedReader(f); tar.addfile(info,reader)
                if reader.n != d['size'] or meta(p)!=d: raise RuntimeError('File changed '+str(p))
                manifest['files'][arcname]=dict(bytes=reader.n,sha256=reader.h.hexdigest())
            else: tar.addfile(info)
        if snap(name)!=before: raise RuntimeError('Target changed during archival '+name)
        raw=json.dumps(manifest).encode()
        info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(raw);info.mode=0o600
        tar.addfile(info,io.BytesIO(raw))
elif mode=='delete':
    import shutil
    jobs()
    receipt=dict(started_utc=now(),root=str(ROOT),archive_verification=CONFIG['verification'],independent_active_jobs=list((CONFIG.get('independent_job_guard') or {}).get('jobs',{})),metadata_timestamp_corrections=CONFIG.get('metadata_timestamp_corrections',{}),deleted=[])
    # Verify the entire scope before any deletion, then recheck each target.
    for name in TARGETS:
        verify_snapshot(name)
    for relative, expected_hash in CONFIG.get('content_hashes',{}).items():
        path=ROOT/relative
        if path.relative_to(ROOT).parts[0] not in TARGETS:raise RuntimeError('Hash path outside approved scope')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        if h.hexdigest()!=expected_hash:raise RuntimeError('Original differs from verified archive: '+relative)
    for name in TARGETS:
        jobs()
        before=CONFIG['expected'][name]
        if before is None: continue
        verify_snapshot(name)
        shutil.rmtree(ROOT/name)
        if os.path.lexists(ROOT/name): raise RuntimeError('Deletion incomplete: '+name)
        row=dict(target=name,entries=len(before),allocated_bytes=sum(d['blocks']*512 for d in before.values()),deleted_utc=now())
        receipt['deleted'].append(row)
        print(json.dumps(dict(event='deleted',**row)),flush=True)
    receipt['finished_utc']=now()
    receipt['all_targets_absent']=all(not os.path.lexists(ROOT/n) for n in TARGETS)
    print(json.dumps(dict(event='receipt',receipt=receipt)),flush=True)
'''

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def remote(config, output):
    config = dict(root=ROOT, targets=TARGETS, **config)
    code = 'CONFIG = ' + repr(config) + '\n' + REMOTE
    encoded=base64.b64encode(zlib.compress(code.encode(),9)).decode()
    command='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))\n'
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
                    'bocconi-cluster','python3 -'], input=command.encode(), stdout=output, check=True)

def inventory():
    scope = dict(authorization='User explicitly approved deletion of all six annotated experiment groups, after recording all results.',
                 annotation_groups=GROUPS, root=ROOT, checkpoint_policy=('User choice: preserve every non-checkpoint file including predictions; omit and delete checkpoint weights.' if RESULTS_ONLY else 'Full local archive includes every regular file.'),
                 archive_only_provenance='band_focus_cleanup_20261011 is an existing cleanup receipt and is not a deletion target.')
    (BASE/'scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    with (BASE/'inventory.partial.json').open('wb') as out:
        remote(dict(mode='inventory'), out)
    data=json.loads((BASE/'inventory.partial.json').read_text())
    (BASE/'inventory.partial.json').rename(BASE/'inventory.json')
    total=sum(d['size'] for entries in data['targets'].values() if entries for d in entries.values() if stat.S_ISREG(d['mode']))
    print(json.dumps(dict(targets=len(TARGETS),regular_file_gib=total/2**30,local_free_gib=shutil.disk_usage(BASE).free/2**30)),flush=True)

def archive():
    inv=json.loads((BASE/'inventory.json').read_text())
    archives=BASE/ARCHIVE_DIR;archives.mkdir(exist_ok=True)
    for name in TARGETS:
        if inv['targets'][name] is None: continue
        target=archives/(name+'.tar.gz')
        if target.exists():
            print(json.dumps(dict(event='archive_exists',target=name)),flush=True);continue
        size=sum(d['size'] for rel,d in inv['targets'][name].items() if stat.S_ISREG(d['mode']) and not (RESULTS_ONLY and Path(rel).suffix in {'.pt','.pth','.ckpt','.safetensors'}))
        if shutil.disk_usage(BASE).free < size+5*2**30:
            raise RuntimeError('Need archive size plus 5 GiB local headroom for '+name)
        print(json.dumps(dict(event='archive_started',target=name,source_gib=size/2**30,timestamp_utc=now())),flush=True)
        partial=archives/(name+'.tar.gz.partial')
        with partial.open('wb') as out:
            remote(dict(mode='archive',target=name,expected=inv['targets'][name],results_only=RESULTS_ONLY),out)
            out.flush();os.fsync(out.fileno())
        partial.rename(target)
        print(json.dumps(dict(event='archive_saved',target=name,archive_gib=target.stat().st_size/2**30,timestamp_utc=now())),flush=True)

def verify():
    inv=json.loads((BASE/'inventory.json').read_text())
    results=[]
    for name,entries in inv['targets'].items():
        if entries is None: continue
        p=BASE/ARCHIVE_DIR/(name+'.tar.gz');seen={};manifest=None;members={};readability={'json_files':0,'npz_files':0,'preexisting_json_issues':[]}
        with tarfile.open(p,mode='r|gz') as tar:
            for item in tar:
                if item.name in members: raise RuntimeError('Duplicate archive member')
                members[item.name]=dict(type=item.type.decode(),linkname=item.linkname)
                if item.name=='ARCHIVE_MANIFEST.json':
                    manifest=json.load(tar.extractfile(item));continue
                if item.isfile():
                    h=hashlib.sha256();n=0
                    f=tar.extractfile(item)
                    check_data=item.name.endswith(('.json','.npz'))
                    chunks=[]
                    for b in iter(lambda:f.read(8*1024**2),b''):
                        h.update(b);n+=len(b)
                        if check_data:chunks.append(b)
                    seen[item.name]=dict(bytes=n,sha256=h.hexdigest())
                    if check_data:
                        data=b''.join(chunks)
                        if item.name.endswith('.npz'):
                            with zipfile.ZipFile(io.BytesIO(data)) as z:
                                assert z.testzip() is None, 'Unreadable prediction/array archive: '+item.name
                            readability['npz_files']+=1
                        else:
                            try:json.loads(data);readability['json_files']+=1
                            except (ValueError,UnicodeDecodeError) as e:readability['preexisting_json_issues'].append({'path':item.name,'error':str(e)})
        if manifest is None or manifest['entries']!=entries or manifest['files']!=seen:
            raise RuntimeError('Archive differs from source manifest: '+name)
        excluded={rel for rel,d in entries.items() if stat.S_ISREG(d['mode']) and Path(rel).suffix in {'.pt','.pth','.ckpt','.safetensors'}} if RESULTS_ONLY else set()
        if set(manifest.get('excluded_checkpoint_paths',[]))!=excluded: raise RuntimeError('Unexpected exclusion')
        expected={name if rel=='.' else name+'/'+rel for rel in entries if rel not in excluded}
        if set(members)-{'ARCHIVE_MANIFEST.json'}!=expected:
            raise RuntimeError('Missing or extra entries: '+name)
        for rel,entry in entries.items():
            key=name if rel=='.' else name+'/'+rel
            if stat.S_ISLNK(entry['mode']) and members[key]['linkname']!=entry['link_target']:
                raise RuntimeError('Symlink differs: '+key)
        manifest_dir=BASE/('results_manifests' if RESULTS_ONLY else 'manifests');manifest_dir.mkdir(exist_ok=True)
        (manifest_dir/(name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        row=dict(target=name,archive=str(p),archive_bytes=p.stat().st_size,archive_sha256=digest_file(p),
                 verified_regular_files=len(seen),verified_entries=len(expected),excluded_checkpoint_files=len(excluded),source_bytes=sum(v['bytes'] for v in seen.values()),readability=readability,status='verified')
        results.append(row)
        print(json.dumps(row),flush=True)
    result=dict(status='verified',verified_at_utc=now(),archives=results,inventory_sha256=digest_file(BASE/'inventory.json'))
    (BASE/VERIFY_NAME).write_text(json.dumps(result,indent=2)+'\n')

def delete():
    if len(TARGETS)==16 and RESULTS_ONLY:
        gate=json.loads((BASE/'evaluation_gate.json').read_text())
        assert gate['status']=='verified' and gate['hippocampus_endpoints']==100 and gate['geofwi3d_runs']==9
        assert digest_file(Path(gate['evaluation_archive']))==gate['evaluation_archive_sha256']
    verification=json.loads((BASE/VERIFY_NAME).read_text())
    assert verification['status']=='verified'
    assert verification['inventory_sha256']==digest_file(BASE/'inventory.json')
    inv=json.loads((BASE/'inventory.json').read_text())
    assert {a['target'] for a in verification['archives']}=={n for n,e in inv['targets'].items() if e is not None}
    for row in verification['archives']:
        p=Path(row['archive'])
        assert p.is_relative_to(BASE/ARCHIVE_DIR) and p.stat().st_size==row['archive_bytes']
        assert digest_file(p)==row['archive_sha256'], 'Archive changed after verification'
    receipt_path=BASE/'deletion_events.jsonl'
    prior_deleted=[];prior_event_files=[]
    for event_file in sorted(BASE.glob('partial_deletion_events_*.jsonl')):
        prior_event_files.append(str(event_file))
        for line in event_file.read_text().splitlines():
            event=json.loads(line)
            if event['event']=='deleted':
                assert event['target'] in TARGETS
                assert event['target'] not in {r['target'] for r in prior_deleted}
                prior_deleted.append({k:v for k,v in event.items() if k!='event'})
    expected=dict(inv['targets'])
    for row in prior_deleted:expected[row['target']]=None
    content_hashes={}
    manifest_dir=BASE/('results_manifests' if RESULTS_ONLY else 'manifests')
    for name in TARGETS:
        if expected[name] is None:continue
        path=manifest_dir/(name+'.json')
        if path.exists():
            content_hashes.update({p:r['sha256'] for p,r in json.loads(path.read_text())['files'].items()})
    guard_path=BASE/'active_job_dependency_check.json'
    if not guard_path.exists():guard_path=BASE.parent/'active_job_dependency_check.json'
    guard=json.loads(guard_path.read_text()) if guard_path.exists() else None
    correction_path=BASE/'metadata_timestamp_corrections.json'
    corrections=json.loads(correction_path.read_text()) if correction_path.exists() else {}
    with receipt_path.open('wb') as out:
        remote(dict(mode='delete',expected=expected,content_hashes=content_hashes,independent_job_guard=guard,metadata_timestamp_corrections=corrections,verification=dict(path=str(BASE/VERIFY_NAME),sha256=digest_file(BASE/VERIFY_NAME))),out)
        out.flush();os.fsync(out.fileno())
    events=[json.loads(line) for line in receipt_path.read_text().splitlines()]
    receipt=next(e['receipt'] for e in events if e['event']=='receipt')
    assert receipt['all_targets_absent']
    receipt['deleted']=prior_deleted+receipt['deleted']
    receipt['prior_event_files']=prior_event_files
    receipt['resumed_after_verified_partial_deletions']=len(prior_deleted)
    assert len(receipt['deleted'])==len(TARGETS)
    (BASE/'deletion_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(deleted_targets=len(receipt['deleted']),deleted_allocated_gib=sum(r['allocated_bytes'] for r in receipt['deleted'])/2**30)),flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    {'inventory':inventory,'archive':archive,'verify':verify,'delete':delete,
     'archive_results':archive,'verify_results':verify,'delete_results':delete}[mode]()
