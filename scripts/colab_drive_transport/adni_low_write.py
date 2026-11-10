"""Storage-only reductions; never defer charges or weaken archive verification."""
import hashlib
import json
from pathlib import Path
import tarfile
import time


def publish_control(runtime, control, durable, session):
    """Keep one fully verified copy per control version, plus a session reference."""
    manifest_path = control/'control_manifest.json'
    sha = runtime.digest(manifest_path)
    manifest = runtime.read(manifest_path)
    destination = durable/'parallel'/('control_'+sha)
    names = {**manifest['files'], 'control_manifest.json': sha}
    marker = destination/'control_manifest.json'
    if marker.exists():
        # A published version is immutable. Corruption fails, never overwrites.
        for name, expected in names.items():
            if Path(name).name != name or runtime.digest(destination/name) != expected:
                raise ValueError('Saved controller provenance differs: '+name)
        print('REUSED VERIFIED CONTROLLER:', sha[:12], '| no source uploads', flush=True)
    else:
        for name, expected in names.items():
            if Path(name).name != name or runtime.digest(control/name) != expected:
                raise ValueError('Controller provenance source differs: '+name)
            runtime.copy_verified(control/name, destination/name)
        # Manifest is the last copied file: incomplete publication is retried.
    runtime.write(session/'control_reference.json', dict(
        control_sha256=sha, directory=str(destination), files_verified=len(names)))
    return destination


def prior_snapshot_matches(runtime, archive, pointer, manifest):
    """Verify the entire old archive before reusing its exact payload hashes."""
    if runtime.digest(archive) != pointer['sha256']:
        raise ValueError('Existing recovery archive hash differs')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Unsafe existing recovery archive')
        if set(names) != set(manifest) | {'snapshot_manifest.json'}:
            return False
        with tar.extractfile('snapshot_manifest.json') as stream:
            saved = json.load(stream)
        if saved != manifest:
            return False
        for name, expected in manifest.items():
            h = hashlib.sha256()
            with tar.extractfile(name) as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Existing recovery payload hash differs: '+name)
    return True


def install(io):
    """Keep volatile quota snapshots on SSD; the live quota request still runs."""
    original_write, original_read = io.write, io.read
    def quota_path(path):
        return io.local/('quota_'+hashlib.sha256(str(path).encode()).hexdigest()+'.json')
    def write(path, value):
        if Path(path).name == 'storage_quota.json' and io.durable(path):
            io.add(quota_snapshot_drive_writes_avoided=1)
            return io.raw_write(quota_path(path), value)
        return original_write(path, value)
    def read(path):
        if Path(path).name == 'storage_quota.json' and io.durable(path):
            local = quota_path(path)
            if local.exists(): return io.raw_read(local)
        return original_read(path)
    io.write, io.read = write, read


def server_save_probe(folder_id, expected_account):
    """A real API write/read roundtrip before any scientific work; no FUSE cache."""
    import io as streams
    import uuid
    import google.auth
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from request_gate import request_call
    credentials = google.auth.default()[0]
    service = build('drive', 'v3', http=google_auth_httplib2.AuthorizedHttp(
        credentials, http=httplib2.Http(timeout=120)), cache_discovery=False)
    def execute(request): return request_call(lambda: request.execute(num_retries=0))
    user = execute(service.about().get(fields='user(emailAddress)'))['user']['emailAddress']
    if user.lower() != expected_account.lower(): raise RuntimeError('Wrong Drive account')
    identifier = execute(service.files().generateIds(count=1, space='drive'))['ids'][0]
    payload = json.dumps(dict(probe=uuid.uuid4().hex, at=time.time())).encode()
    print('PHASE: check server-acknowledged Drive saving before launching workers', flush=True)
    try:
        execute(service.files().create(body=dict(id=identifier, name='adni_save_probe_'+identifier,
            parents=[folder_id]), media_body=MediaIoBaseUpload(streams.BytesIO(payload),
            mimetype='application/json', resumable=False), fields='id'))
        received = execute(service.files().get_media(fileId=identifier))
        if received != payload: raise ValueError('Drive server save probe readback differs')
    except Exception:
        # The generated ID is retained locally for inspection after an unknown
        # create outcome. Never infer that a timeout means the write failed.
        Path('/content/adni_failed_save_probe.json').write_text(json.dumps(dict(id=identifier)))
        raise
    execute(service.files().delete(fileId=identifier))
    print('PASSED: Drive server write and independent readback; quota may still change later.', flush=True)
