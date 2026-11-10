"""Storage-only hooks around the frozen scientific entry point and scheduler."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import zipfile

from store import Cloud, atomic, digest, upload_slot


def configured():
    value = json.loads(Path(os.environ['HIPPO_CLOUD_CONFIG']).read_text())
    return value, Cloud(value['folder_id'])


def bundle_results(output, manifest):
    archive = output.parent/(output.name+'_results.zip')
    names = [*manifest['metadata'], 'archive_manifest.json']
    for name in ('archive_acknowledgment.json', 'cleanup.json'):
        if (output/name).exists():
            names.append(name)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(names)):
            # A local copy's mtime is not scientific content. Stable headers let
            # worker and controller reuse the SAME already-committed archive.
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, (output/name).read_bytes())
    return archive


def closed_manifest(output, task):
    """Repair only the known append-only log race, never changed science."""
    from scripts.cross_dataset_campaign import storage
    path = output/'archive_manifest.json'
    old = None
    if path.exists():
        old = json.loads(path.read_text())
        if old['task'] != task:
            raise ValueError('Existing archive task differs')
        for name, item in old['metadata'].items():
            current = output/name
            if name == 'train.log':
                # The already-hashed prefix must be exactly preserved. Truncate
                # or edit is corruption, not the known append-only shutdown log.
                with current.open('rb') as stream:
                    prefix = stream.read(item['bytes'])
                if len(prefix) != item['bytes'] or hashlib.sha256(prefix).hexdigest() != item['sha256']:
                    raise ValueError('Original train.log prefix differs; cannot repair')
            elif current.stat().st_size != item['bytes'] or digest(current) != item['sha256']:
                raise ValueError('Scientific archive differs: '+name)
        # Preserve the previous receipt as diagnostic evidence outside the run.
        evidence = output.parent/(output.name+'_prior_manifest_'+digest(path)+'.json')
        if not evidence.exists():
            shutil.copyfile(path, evidence)
    storage.verify_results(output, task)
    metadata, bulky = {}, []
    for current in sorted(output.rglob('*')):
        if current.is_dir():
            continue
        if current.is_symlink():
            raise ValueError('No symlinks in result manifest')
        name = str(current.relative_to(output))
        if name in ('archive_manifest.json','archive_acknowledgment.json','cleanup.json'):
            continue
        item = dict(bytes=current.stat().st_size, sha256=digest(current))
        if current.suffix in ('.pt','.tmp') or current.name.endswith('.nii.gz'):
            bulky.append(dict(name=name, **item))
        elif current.suffix in ('.json','.csv','.jsonl','.log','.lock','.png'):
            metadata[name] = item
        else:
            raise ValueError('Unclassified output: '+name)
    if old and old['metadata']==metadata and old['deletion_candidates']==bulky:
        return old
    manifest = dict(status='ready', task=task, metadata=metadata,
                    deletion_candidates=bulky, created_at=time.time(),
                    completion_sha256=digest(output/'completion.json'))
    atomic(path, manifest)
    return manifest


def commit_result(root, output, task, phase, cloud, live=False):
    from scripts.cross_dataset_campaign import storage
    if live:
        # Copy live logs/metadata; only immutable checkpoint inodes are linked.
        # The worker can keep logging after this returns without invalidating
        # this committed snapshot or poisoning the controller's final manifest.
        with tempfile.TemporaryDirectory(prefix='result-snapshot-', dir=output.parent) as temp:
            staged = Path(temp)
            for current in sorted(output.rglob('*')):
                if not current.is_file():
                    continue
                name = current.relative_to(output)
                if str(name) in ('archive_manifest.json','archive_acknowledgment.json','cleanup.json'):
                    continue
                if current.is_symlink():
                    raise ValueError('Symlink in result snapshot')
                target = staged/name
                target.parent.mkdir(parents=True, exist_ok=True)
                if current.suffix == '.pt':
                    os.link(current, target)
                else:
                    shutil.copyfile(current, target)
            try:
                return commit_result(root, staged, task, phase, cloud)
            finally:
                # These siblings belong ONLY to our temporary snapshot.
                staged.with_name(staged.name+'_results.zip').unlink(missing_ok=True)
                staged.with_name(staged.name+'_server_commit.json').unlink(missing_ok=True)
    manifest = closed_manifest(output, task)
    storage.verify_archive(output, manifest)
    archive = bundle_results(output, manifest)
    saved = output.parent/(output.name+'_server_commit.json')
    if saved.exists():
        previous = json.loads(saved.read_text())
        if previous.get('local_sha256') == digest(archive):
            metadata = cloud.metadata(previous['id'])
            if metadata.get('parents') != [cloud.folder] or not cloud.matches(archive, metadata):
                raise RuntimeError('Previously committed result changed; preserve local files')
            return previous, manifest
    receipt = cloud.put_immutable(archive, 'result', task['id'], phase)
    atomic(saved, receipt)
    return receipt, manifest


def install_entry(entry, output, bundle, stats):
    config, cloud = configured()
    phase = os.environ['HIPPO_CLOUD_PHASE']
    task = json.loads(os.environ['HIPPO_CLOUD_TASK'])
    ledger_id_path = output.parent/(output.name+'_ledger_id.json')
    original_write, original_read = entry.write, entry.read
    original_writer = entry.RecoveryWriter
    fault = Path(config['local_root'])/'SAVE_BLOCKED.json'
    budget_stop = Path(config['local_root'])/'CHECKPOINT_BUDGET_STOP.json'

    def fail(exc):
        atomic(fault, dict(task=task['id'], error=str(exc), time=time.time(),
                          policy='No new updates or launches; retain local artifacts'))

    def write(path, value):
        if Path(path).name != 'charged_ledger.json':
            return original_write(path, value)
        if fault.exists():
            raise RuntimeError('Saving is blocked; no further optimization update is authorized')
        # The server charge is acknowledged BEFORE the frozen trainer's update.
        temporary = output.parent/(output.name+'_next_charge.json')
        atomic(temporary, value)
        started=time.monotonic()
        try:
            file_id = json.loads(ledger_id_path.read_text())['id'] if ledger_id_path.exists() else None
            receipt = cloud.upload(temporary, 'ledger__'+phase+'__'+task['id']+'.json',
                {'kind':'ledger','task':task['id'],'phase':phase}, file_id=file_id)
            atomic(ledger_id_path, receipt)
            original_write(path, value)
        except Exception as exc:
            fail(exc); raise
        finally:
            stats['server_ledger_seconds']=stats.get('server_ledger_seconds',0.)+time.monotonic()-started

    entry.write = write
    # entry.read now reads a local copy of the exact bytes acknowledged by Drive.
    entry.read = original_read

    class VerifiedRecovery(original_writer):
        def __init__(self, output, durable, *args, **kwargs):
            super().__init__(Path(output).resolve(), durable, *args, **kwargs)
            self.interval = config.get('checkpoint_interval_seconds', self.interval)
            self.forced_snapshot = False
            self.last_signature = None
            self.result_committed = False

        def capture(self, force=False):
            # Do not change the upload priority underneath an in-flight future.
            if self.future:
                if force or self.future.done():
                    self.future.result()
                else:
                    return
            self.forced_snapshot = force
            return super().capture(force=force)

        def _persist(self, snapshot):
            archive = snapshot.with_suffix('.tar.gz')
            try:
                if self.result_committed:
                    shutil.rmtree(snapshot)
                    return
                files = {p.name:digest(p) for p in sorted(snapshot.iterdir())}
                signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
                if signature == self.last_signature:
                    shutil.rmtree(snapshot)
                    return
                pointer=self.durable/'cloud_latest.json'
                if pointer.exists():
                    previous=json.loads(pointer.read_text())
                    if previous.get('signature')==signature:
                        meta=cloud.metadata(previous['id'])
                        if (meta.get('parents')!=[cloud.folder] or
                                not previous.get('md5Checksum') or
                                any(meta.get(k)!=previous.get(k) for k in ('size','md5Checksum'))):
                            raise RuntimeError('Previously saved checkpoint changed; local state retained')
                        self.last_signature=signature
                        shutil.rmtree(snapshot)
                        return
                atomic(snapshot/'snapshot_manifest.json', files)
                with archive.open('wb') as raw:
                    with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                        with tarfile.open(fileobj=compressed, mode='w|') as tar:
                            for p in sorted(snapshot.iterdir()):
                                tar.add(p, arcname=p.name, recursive=False)
                # Large uploads are serialized; small charge requests are not.
                with upload_slot(Path(config['local_root'])):
                    if self.result_committed:
                        archive.unlink(missing_ok=True)
                        shutil.rmtree(snapshot)
                        return
                    kwargs=({'priority':'recovery'} if self.forced_snapshot else {})
                    receipt = cloud.put_immutable(archive, 'checkpoint', task['id'], phase, **kwargs)
                self.durable.mkdir(parents=True, exist_ok=True)
                # The local pointer refers to the server object, never a FUSE copy.
                atomic(self.durable/'cloud_latest.json', dict(**receipt, signature=signature,
                    charged_attempts=json.loads((snapshot/'budget_ledger.json').read_text())['charged_attempts']))
                print('SERVER VERIFIED CHECKPOINT',task['id'],
                      '| compressed MiB',round(archive.stat().st_size/1024**2,1),flush=True)
                self.last_signature = signature
            except Exception as exc:
                if getattr(exc,'checkpoint_budget_pause',False):
                    # This is an orderly stop request, not a failed Drive write.
                    # Keep this snapshot; close(force=True) uploads the newest
                    # checkpoint from the reserved allowance before exit.
                    atomic(budget_stop,dict(task=task['id'],reason=str(exc),time=time.time()))
                    print('CHECKPOINT BUDGET: requesting clean stop; recovery reserve remains available.',flush=True)
                    return
                fail(exc)
                # Keep the failed archive and snapshot for a same-VM retry.
                raise
            else:
                archive.unlink(missing_ok=True)
                shutil.rmtree(snapshot)

        def close(self, force=True):
            complete = (self.output/'completion.json').exists()
            try:
                # A successful run commits its complete small scientific archive
                # instead of redundantly uploading another multi-GB checkpoint.
                if complete:
                    receipt, _ = commit_result(bundle, self.output, task, phase, cloud, live=True)
                    self.result_committed = True
                    print('SERVER VERIFIED RESULT',task['id'],flush=True)
                    super().close(force=False)
                    stats['server_result_id'] = receipt['id']
                else:
                    super().close(force=force)
            except Exception as exc:
                fail(exc); raise

    entry.RecoveryWriter = VerifiedRecovery


def install_sweep(sweep, phase):
    config, cloud = configured()
    fault = Path(config['local_root'])/'SAVE_BLOCKED.json'
    budget_stop = Path(config['local_root'])/'CHECKPOINT_BUDGET_STOP.json'
    original_launch = sweep.launch
    def launch(task):
        if budget_stop.exists():
            sweep.request_stop('Routine upload allowance reached; saving workers using recovery reserve')
            return
        if fault.exists():
            sweep.request_stop('Drive saving failed; preserving local work')
            return
        # Scheduler launches workers serially; each child inherits its own copy.
        os.environ['HIPPO_CLOUD_PHASE'] = phase
        os.environ['HIPPO_CLOUD_TASK'] = json.dumps(task)
        return original_launch(task)
    sweep.launch = launch
    original_check = sweep.check_stop
    def check():
        if budget_stop.exists():
            sweep.request_stop('Routine upload allowance reached; saving workers using recovery reserve')
        if fault.exists():
            sweep.request_stop('Drive saving failed; no more launches')
        return original_check()
    sweep.check_stop = check
    last_quota = [0., None]
    def quota(durable):
        if time.monotonic()-last_quota[0] > 60:
            q = cloud.about()['storageQuota']
            last_quota[:] = [time.monotonic(), q]
        q = last_quota[1]
        sweep.r.write(durable/'storage_quota.json', dict(checked_at=time.time(), **q))
        if 'limit' in q and int(q['limit'])-int(q.get('usage',0)) < 8*1024**3:
            raise RuntimeError('Drive has less than 8 GiB available; stop and preserve recovery')
    sweep.r.check_drive_headroom = quota

    def archive(root, output, durable, recovery, task):
        from scripts.cross_dataset_campaign import storage
        # Server commitment must succeed before storage.retire can run.
        committed, manifest = commit_result(root, output, task, phase, cloud)
        target = durable/'runs'/task['id']
        target.mkdir(parents=True, exist_ok=True)
        for name in (*manifest['metadata'], 'archive_manifest.json'):
            sweep.r.copy_verified(output/name, target/name)
        ack = dict(status='verified', manifest_sha256=digest(output/'archive_manifest.json'),
            source_verified=True, source_manifest_sha256=digest(root/'bundle_manifest.json'),
            durable_archive='https://drive.google.com/file/d/'+committed['id']+'/view',
            server_commit=committed, verified_at=time.time())
        atomic(output/'archive_acknowledgment.json', ack)
        atomic(target/'archive_acknowledgment.json', ack)
        receipt = storage.retire(output)
        atomic(target/'cleanup.json', receipt)
        packed = bundle_results(target, manifest)
        cloud.put_immutable(packed, 'archive', task['id'], phase)
        # Retire only new-store recovery objects after complete results commit.
        for f in cloud.children():
            p = f.get('appProperties', {})
            if p.get('task') == task['id'] and p.get('phase') == phase and p.get('kind') in ('checkpoint','ledger'):
                cloud.execute(cloud.service().files().delete(fileId=f['id']))
        if recovery.exists():
            shutil.rmtree(recovery)
        packed.unlink(missing_ok=True)
    sweep.r.archive_completed = archive
