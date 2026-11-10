"""External, lossless Drive transport; scientific entry and charge ordering stay frozen.

All coordination/counters live on the VM SSD. This cannot increase Google's
account quotas or coordinate other VMs. A throttle requests a clean campaign
stop; bounded retries are only for the SAME idempotent transfer/read/write.
"""
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import threading
import time


def local_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False) + '\n')
    tmp.replace(path)


def throttled(exc):
    text = str(exc).lower()
    status = getattr(getattr(exc, 'resp', None), 'status', None)
    # Do not retry permission errors, corruption, missing files or arbitrary EIO.
    return status == 429 or any(s in text for s in (
        'user rate limit exceeded', 'userratelimitexceeded', 'ratelimitexceeded',
        'rate limit exceeded', 'too many requests', 'download quota exceeded',
        'upload quota exceeded', 'daily limit exceeded', 'dailylimitexceeded'))


def coordination_root(drive_root, local_root=None):
    root = Path(local_root or os.environ.get('HIPPO_DRIVE_IO_LOCAL', '/content/hippo_adni_drive_io'))
    key = hashlib.sha256(str(Path(drive_root).resolve()).encode()).hexdigest()[:20]
    return root/key


def quota_since(drive_root, started, local_root=None):
    """The two-phase coordinator must also see throttling during final uploads."""
    path = coordination_root(drive_root, local_root)/'quota_event.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value['at'] >= started:
            return value
    return None


class DriveIO:
    def __init__(self, runtime, drive_root, local_root=None, spacing=.5, attempts=5):
        self.r = runtime
        self.drive_root = Path(drive_root).resolve()
        self.drive_prefixes = (Path(os.path.abspath(drive_root)), self.drive_root)
        self.local = coordination_root(self.drive_root, local_root)
        self.local.mkdir(parents=True, exist_ok=True)
        if self.local.resolve().is_relative_to(self.drive_root):
            raise ValueError('Drive coordination must stay on local SSD')
        self.spacing, self.attempts = spacing, attempts
        self.started = time.time()
        self.raw_write, self.raw_read = runtime.write, runtime.read
        self.raw_copy, self.raw_quota = runtime.copy_verified, runtime.check_drive_headroom
        self.stats = dict(uploaded_bytes=0, duplicate_bytes_avoided=0, copies=0,
                          duplicate_copies=0, json_writes=0, retries=0,
                          throttle_events=0, pacing_seconds=0., recovery_duplicates=0,
                          recovery_input_bytes=0, recovery_compressed_bytes=0,
                          compression_seconds=0., phase='idle')
        self._stats_lock = threading.RLock()
        self._directories = {}
        self._written = {}
        self._quota_checked = -float('inf')
        self.server_ledger = None
        if os.environ.get('HIPPO_ADNI_SERVER_ROOT'):
            from server_ledger import ServerLedger
            self.server_ledger = ServerLedger(self, os.environ['HIPPO_ADNI_SERVER_ROOT'])

    def durable(self, path):
        # Paths are supplied by the verified runner. Avoid resolve()/lstat of
        # every Drive path component on every update; root aliases bind once.
        absolute = Path(os.path.abspath(path))
        return any(absolute.is_relative_to(root) for root in self.drive_prefixes)

    def record(self, **changes):
        with self._stats_lock:
            self.stats.update(changes)
            local_json(self.local/f'process_{os.getpid()}.json',
                       dict(self.stats, pid=os.getpid(), updated_at=time.time()))

    def add(self, **changes):
        with self._stats_lock:
            self.record(**{k: self.stats.get(k, 0)+v for k, v in changes.items()})

    @contextmanager
    def lock(self, name):
        with (self.local/(name+'.lock')).open('a') as handle:
            last_message = time.monotonic()
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()-last_message >= 30:
                        print('DRIVE WAIT: waiting for another worker\'s '+name+'; no transfer progress yet', flush=True)
                        last_message = time.monotonic()
                    time.sleep(.1)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def wait(self, seconds):
        deadline = time.monotonic()+max(0., seconds)
        while time.monotonic() < deadline:
            left = deadline-time.monotonic()
            if left >= 30:
                print(f'DRIVE WAIT: backing off for {left:.0f}s; no transfer progress yet', flush=True)
            time.sleep(min(30., max(0., left)))

    def pace(self, mutation):
        start = time.monotonic()
        # Very short metadata lane; big transfers use a SEPARATE lock so ledger
        # writes are never queued behind a multi-GB upload holding this lock.
        with self.lock('request_pacing'):
            path = self.local/'pacing.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            until = value.get('cooldown_until', 0.)
            if mutation:
                until = max(until, value.get('next_write', 0.))
            self.wait(until-time.time())
            if mutation:
                value['next_write'] = time.time()+self.spacing
                local_json(path, value)
        self.add(pacing_seconds=time.monotonic()-start)

    def quota_event(self):
        path = self.local/'quota_event.json'
        if path.exists():
            value = json.loads(path.read_text())
            if value['at'] >= self.started:
                return value
        return None

    def call(self, label, operation, mutation=False):
        for attempt in range(self.attempts):
            self.pace(mutation)
            try:
                return operation()
            except Exception as exc:
                if not throttled(exc):
                    raise
                self.add(throttle_events=1)
                delay = min(64., 4.*2**attempt) + random.SystemRandom().uniform(0., 1.)
                event = dict(at=time.time(), operation=label, attempt=attempt+1,
                             retry_in_seconds=delay, error=str(exc)[:500])
                with self.lock('request_pacing'):
                    path = self.local/'pacing.json'
                    value = json.loads(path.read_text()) if path.exists() else {}
                    value['cooldown_until'] = max(value.get('cooldown_until', 0.), time.time()+delay)
                    local_json(path, value)
                    local_json(self.local/'quota_event.json', event)
                # The frozen trainer tests this only at its existing safe
                # boundaries. Never skip the update whose ledger just committed.
                os.environ['HIPPO_DEADLINE'] = '0'
                print(f'DRIVE THROTTLED: {label}; clean stop requested; '
                      f'idempotent retry {attempt+1}/{self.attempts}', flush=True)
                if attempt+1 == self.attempts:
                    raise RuntimeError('Drive throttling persisted; original accounting/recovery retained; '
                                       'no further update may execute') from exc
                self.add(retries=1)

    def write(self, path, value):
        if not self.durable(path):
            return self.raw_write(path, value)
        path = Path(path)
        if path.name == 'storage_quota.json':
            # Live quota checks remain; their diagnostic receipt need not be
            # uploaded on every worker launch.
            return self.raw_write(self.local/'storage_quota.json', value)
        if path.name == 'charged_ledger.json' and self.server_ledger is not None:
            return self.server_ledger.write(path, value)
        # NEVER coalesce or defer a charge. Entry still reads it back before
        # every original update, and calibration charges use the same path.
        charge = path.name == 'charged_ledger.json'
        fingerprint = json.dumps(value, sort_keys=True, allow_nan=False)
        if not charge and self._written.get(str(path)) == fingerprint:
            if self.read(path) == value:
                return
        def persist():
            self.raw_write(path, value)
            # Read back the recovery pointer before old-archive retirement.
            # Entry's separate ledger readback remains unchanged.
            if path.name == 'latest.json' and self.raw_read(path) != value:
                raise ValueError('Drive JSON readback differs: '+str(path))
        self.call('write '+path.name, persist, mutation=True)
        self._written[str(path)] = fingerprint
        self.add(json_writes=1)

    def read(self, path):
        if not self.durable(path):
            return self.raw_read(path)
        if Path(path).name == 'charged_ledger.json' and self.server_ledger is not None:
            return self.server_ledger.read(path)
        return self.call('read '+Path(path).name, lambda: self.raw_read(path))

    def copy(self, source, target):
        if not self.durable(target):
            return self.raw_copy(source, target)
        source, target = Path(source), Path(target)
        # The source is an immutable local snapshot or archived metadata file.
        expected, size = self.r.digest(source), source.stat().st_size
        self.record(phase='waiting for bulk upload lane: '+target.name)
        with self.lock('bulk_upload'):
            self.record(phase='uploading/verifying '+target.name)
            try:
                def transfer():
                    parent = str(target.parent)
                    if parent not in self._directories:
                        self._directories[parent] = {p.name for p in target.parent.iterdir()} if target.parent.exists() else set()
                    names = self._directories[parent]
                    if target.name in names and target.exists() and self.r.digest(target) == expected:
                        if self.server_ledger is not None:
                            self.server_ledger.confirm_file(source, target)
                        self.add(duplicate_copies=1, duplicate_bytes_avoided=size)
                        return
                    self.raw_copy(source, target)  # Original full hash verification.
                    if self.server_ledger is not None:
                        self.server_ledger.confirm_file(source, target)
                    # Publication can fail after rename: retry checks final name
                    # before re-uploading, without trusting a .uploading file.
                    names.add(target.name)
                    self.add(copies=1, uploaded_bytes=size)
                def attempt():
                    try:
                        transfer()
                    except Exception:
                        self._directories.pop(str(target.parent), None)
                        raise
                return self.call('upload '+target.name, attempt, mutation=True)
            finally:
                self.record(phase='idle')

    def quota(self, durable):
        # A controller's simultaneous launches can share a check for 60 seconds.
        # Failure never advances the cache; trash remains included by raw_quota.
        if time.monotonic()-self._quota_checked >= 60:
            result = self.call('check Drive storage quota', lambda: self.raw_quota(durable))
            self._quota_checked = time.monotonic()
            return result

    def recovery_class(self, base):
        io = self
        class CompressedRecovery(base):
            # Inherit capture(), interval=180, immutable hardlink snapshots,
            # single in-flight future and forced close without modification.
            def _persist(self, snapshot):
                archive = snapshot.with_suffix('.tar.gz')
                succeeded = False
                try:
                    manifest = {p.name: io.r.digest(p) for p in snapshot.iterdir()}
                    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                    previous = io.read(self.durable/'latest.json') if (self.durable/'latest.json').exists() else None
                    if getattr(self, '_verified_signature', None) == signature:
                        if previous != self._verified_pointer:
                            raise ValueError('Recovery pointer changed; refusing duplicate suppression')
                        existing = self.durable/io.r.safe_member(previous['file'])
                        if io.call('verify identical recovery', lambda: io.r.digest(existing)) != previous['sha256']:
                            raise ValueError('Existing recovery archive hash differs')
                        io.add(recovery_duplicates=1, duplicate_bytes_avoided=existing.stat().st_size)
                        succeeded = True
                        return
                    io.raw_write(snapshot/'snapshot_manifest.json', manifest)
                    started = time.monotonic()
                    io.record(phase='compressing recovery on local SSD')
                    # Level 1 prioritizes CPU throughput. mtime/header metadata
                    # normalized for content-addressed idempotent retry; payload
                    # files and their SHA256s are NEVER changed or reserialized.
                    with archive.open('wb') as raw:
                        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as zipped:
                            with tarfile.open(fileobj=zipped, mode='w|') as tar:
                                for path in sorted(snapshot.iterdir()):
                                    info = tar.gettarinfo(str(path), arcname=path.name)
                                    info.uid = info.gid = info.mtime = 0
                                    info.uname = info.gname = ''
                                    with path.open('rb') as stream:
                                        tar.addfile(info, stream)
                        raw.flush(); os.fsync(raw.fileno())
                    # Confirm decompression/member readability locally. Equality
                    # of the complete uploaded hash then proves the same on Drive.
                    expected_members = dict(manifest)
                    expected_members['snapshot_manifest.json'] = io.r.digest(snapshot/'snapshot_manifest.json')
                    with tarfile.open(archive) as tar:
                        members = tar.getmembers()
                        if len(members) != len(expected_members) or {m.name for m in members} != set(expected_members):
                            raise ValueError('Incomplete or duplicate recovery members')
                        for member in members:
                            if not member.isfile():
                                raise ValueError('Unexpected recovery member')
                            h = hashlib.sha256()
                            with tar.extractfile(member) as stream:
                                for block in iter(lambda: stream.read(8*1024**2), b''):h.update(block)
                            if h.hexdigest() != expected_members[member.name]:
                                raise ValueError('Compressed recovery member hash differs')
                    io.add(compression_seconds=time.monotonic()-started,
                           recovery_input_bytes=sum((snapshot/n).stat().st_size for n in manifest),
                           recovery_compressed_bytes=archive.stat().st_size)
                    sha = io.r.digest(archive)
                    name = 'recovery-'+sha+'.tar.gz'
                    io.copy(archive, self.durable/name)
                    pointer = dict(file=name, sha256=sha,
                                   charged_attempts=io.raw_read(snapshot/'budget_ledger.json')['charged_attempts'])
                    io.write(self.durable/'latest.json', pointer)
                    # Latest is published and READ BACK before retiring the old
                    # archive. The frozen restore handles old tar and gzip tar.
                    if previous and previous['file'] != name:
                        old = self.durable/io.r.safe_member(previous['file'])
                        if old.exists() and io.call('verify superseded recovery', lambda: io.r.digest(old)) == previous['sha256']:
                            io.call('retire verified superseded recovery', lambda: old.unlink(missing_ok=True), mutation=True)
                    self._verified_signature, self._verified_pointer = signature, pointer
                    succeeded = True
                finally:
                    if succeeded:
                        archive.unlink(missing_ok=True)
                        shutil.rmtree(snapshot)
                    else:
                        io.record(retained_local_recovery=str(snapshot), retained_local_archive=str(archive))
                        print('RECOVERY RETAINED on local SSD after transfer failure:', snapshot, flush=True)
        return CompressedRecovery


def install(runtime, drive_root, local_root=None):
    if getattr(runtime, '_drive_io', None) is not None:
        return runtime._drive_io
    io = DriveIO(runtime, drive_root, local_root)
    runtime.write, runtime.read, runtime.copy_verified = io.write, io.read, io.copy
    runtime.check_drive_headroom = io.quota
    runtime.RecoveryWriter = io.recovery_class(runtime.RecoveryWriter)
    runtime._drive_io = io
    return io


class ReportPublisher:
    """Derived comparisons can coalesce; per-run archives and charges cannot."""
    def __init__(self, runtime, local, durable, interval=900):
        self.r, self.local, self.durable = runtime, local, durable
        self.interval, self.last, self.pending = interval, -float('inf'), None

    def publish(self, value):
        self.r.write(self.local, value)
        self.pending = value
        if time.monotonic()-self.last >= self.interval:
            self.flush()

    def flush(self):
        if self.pending is not None:
            self.r.write(self.durable, self.pending)
            self.last, self.pending = time.monotonic(), None
