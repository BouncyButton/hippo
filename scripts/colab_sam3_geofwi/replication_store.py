"""Single-worker SAM3 Drive transport: local work, acknowledged charges and archives."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import store
import request_gate

ACCOUNT = 'focacciafilippo@gmail.com'
RESULTS = '1EDy575RclOhGBxhuylaHszLA67wT80qU'


class Cloud(store.Cloud):
    def execute(self, request):
        request_gate.pace(spacing=.5)
        return super().execute(request)

    def check_account(self):
        about = self.about()
        if about['user']['emailAddress'].lower() != ACCOUNT:
            raise RuntimeError('Authenticate '+ACCOUNT+'; refusing another Drive account.')
        quota = about['storageQuota']
        if 'limit' in quota and int(quota['limit'])-int(quota['usage']) < 8*1024**3:
            raise RuntimeError('Need 8 GiB free Drive storage, including trash, before proceeding.')
        return about


def pack(root, paths, target):
    """Stable ZIP with an independent hash for every member; stream bulky tensors."""
    root, target = Path(root), Path(target)
    files = {str(p.relative_to(root)): store.digest(p) for p in sorted(set(paths)) if p.is_file()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', allowZip64=True) as z:
        for name in files:
            store.member(name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name.endswith('.pt') else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (root/name).open('rb') as source, z.open(info, 'w', force_zip64=True) as dest:
                shutil.copyfileobj(source, dest, 8*1024**2)
        z.writestr('_snapshot_manifest.json', json.dumps(files, sort_keys=True))
    return files


def unpack(archive, root):
    """Validate every byte and path before replacing any live local state."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate snapshot members')
        manifest = json.loads(z.read('_snapshot_manifest.json'))
        if set(names) != set(manifest) | {'_snapshot_manifest.json'}:
            raise ValueError('Unexpected snapshot members')
        for name, expected in manifest.items():
            store.member(name)
            h = hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
            if h.hexdigest() != expected:
                raise ValueError('Snapshot member checksum differs: '+name)
        for name in manifest:
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name+'.restore')
            with z.open(name) as source, temp.open('wb') as dest:
                shutil.copyfileobj(source, dest, 8*1024**2)
            temp.replace(path)


class Transport:
    """One VM only. A charged update cannot be silently replayed after VM loss."""
    def __init__(self, root):
        self.root = Path(root)
        self.local = self.root/'transport'; self.local.mkdir(exist_ok=True)
        self.cloud = Cloud(RESULTS)
        store.pace_write = lambda:request_gate.pace(spacing=.5)
        self.about = self.cloud.check_account()
        found = [m for m in self.cloud.children() if m['name'] == 'campaign_state.json']
        if len(found) > 1: raise ValueError('Duplicate campaign state')
        self.state_file = self.local/'campaign_state.json'
        self.state_id = found[0]['id'] if found else None
        if found:
            self.cloud.download(self.state_id, self.state_file)
            self.state = json.loads(self.state_file.read_text())
        else:
            self.state = dict(version=1, objects={}, charges={}, completed={}, cleanup=[])
        self.source = store.digest(self.root/'bundle_manifest.json')
        if self.state.get('bundle_sha256', self.source) != self.source:
            raise ValueError('Campaign source differs; do not mix recipes')
        self.state['bundle_sha256'] = self.source
        self.save()
        self.drain_retired()

    def save(self):
        store.atomic(self.state_file, self.state)
        receipt = self.cloud.upload(self.state_file, 'campaign_state.json', file_id=self.state_id)
        self.state_id = receipt['id']

    def probe(self):
        # Independent server GET, not a mounted-filesystem cache readback.
        target = self.local/'server_readback.json'
        target.unlink(missing_ok=True)
        self.cloud.download(self.state_id, target, store.digest(self.state_file))
        if json.loads(target.read_text()) != self.state: raise ValueError('Server write/read failed')
        print('Drive account, storage and server write/read checks passed.', flush=True)

    def delete_owned(self, receipt):
        meta = self.cloud.metadata(receipt['id'])
        if meta.get('parents') != [RESULTS] or meta.get('md5Checksum') != receipt['md5Checksum']:
            raise ValueError('Refusing to delete changed/outside-campaign object')
        self.cloud.execute(self.cloud.service().files().delete(fileId=receipt['id']))

    def drain_retired(self):
        pending = list(self.state.get('retired', []))
        if not pending: return
        active = {v['id'] for v in self.state['objects'].values()}
        for receipt in pending:
            if receipt['id'] in active: raise ValueError('Cannot retire an active snapshot')
            try: self.delete_owned(receipt)
            except Exception as exc:
                if getattr(getattr(exc, 'resp', None), 'status', None) != 404: raise
            self.state['cleanup'].append(dict(drive_id=receipt['id'],bytes=int(receipt['size']),
                reason='replaced by server-verified snapshot',time=time.time()))
        self.state['retired'] = []
        self.save()

    def publish(self, key, paths):
        archive = self.local/(key.replace('/', '_')+'.zip')
        pack(self.root, paths, archive)
        sha = store.digest(archive)
        old = self.state['objects'].get(key)
        if old and old['local_sha256'] == sha:
            meta = self.cloud.metadata(old['id'])
            if not self.cloud.matches(archive, meta): raise ValueError('Remote snapshot changed')
            archive.unlink(); return old
        new = self.cloud.put_immutable(archive, 'sam3', task=key.replace('/', '_'))
        self.state['objects'][key] = new
        if old and old['id'] != new['id']:
            self.state.setdefault('retired', []).append(old)
        # Pointer is acknowledged before the previous recovery can be retired.
        self.save()
        self.drain_retired()
        archive.unlink()
        return new

    def restore(self, key):
        record = self.state['objects'].get(key)
        if not record: return False
        archive = self.local/(key.replace('/', '_')+'.zip')
        self.cloud.download(record['id'], archive, record['local_sha256'])
        unpack(archive, self.root); archive.unlink()
        return True

    def charge(self, task, value):
        self.state['charges'][task] = value
        self.save()  # MUST acknowledge before optimizer.step.

    def retire(self, task):
        if task not in self.state['completed']: raise ValueError('No acknowledged completed archive')
        checkpoint = self.root/'work'/task/'checkpoint_latest.pt'
        removed = []
        if checkpoint.exists():
            removed.append(dict(path=str(checkpoint.relative_to(self.root)), bytes=checkpoint.stat().st_size))
            checkpoint.unlink()
        for kind in ('recovery', 'progress'):
            key = kind+'/'+task
            if key in self.state['objects']:
                old = self.state['objects'][key]
                try: self.delete_owned(old)
                except Exception as exc:
                    if getattr(getattr(exc, 'resp', None), 'status', None) != 404: raise
                del self.state['objects'][key]
                removed.append(dict(drive_id=old['id'], bytes=int(old['size'])))
        self.state['charges'].pop(task, None)
        if removed: self.state['cleanup'].append(dict(task=task, removed=removed, time=time.time()))
        self.save()


class Recovery:
    def __init__(self, transport, task, output):
        self.transport, self.task, self.output = transport, task, output
        self.last = -float('inf'); self.checkpoint_sha = None
        self.progress_last = time.monotonic()

    def paths(self, checkpoint=True):
        paths = [p for p in self.output.rglob('*') if p.is_file()
                 and not p.name.endswith(('.tmp', '.restore'))
                 and (checkpoint or not p.name.endswith('.pt'))]
        probes = self.transport.root/'durable/probes'/self.task
        return paths + [p for p in probes.rglob('*') if p.is_file()]

    def capture(self, force=False):
        if not force and time.monotonic()-self.last < 180: return
        path = self.output/'checkpoint_latest.pt'
        if not path.exists(): return
        sha = store.digest(path)
        if sha != self.checkpoint_sha:
            self.transport.publish('recovery/'+self.task, self.paths())
            self.checkpoint_sha = sha
        self.last = time.monotonic()

    def progress(self, force=False):
        if force or time.monotonic()-self.progress_last >= 180:
            self.transport.publish('progress/'+self.task, self.paths(checkpoint=False))
            self.progress_last = time.monotonic()

    def close(self, force=False):
        if force: self.capture(force=True)
