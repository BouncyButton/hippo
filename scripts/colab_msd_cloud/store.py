"""Drive API durability, immutable result objects and bounded checkpoint traffic.

Never treats a mounted Drive readback as a server acknowledgment. The old
campaign folders are read-only inputs. All writes are confined to one new folder.
"""
from contextlib import contextmanager
import fcntl
import errno
import hashlib
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import socket
import ssl
import threading
import time
import uuid
import zipfile

ACCOUNT = 'focacciafilippo@gmail.com'
CAMPAIGN = '1ipls98K9dmRQ5rZwfbepw1JyWVL3Yq0P'
STORE_NAME = 'results_seeds0220_server_verified_v2'
FIELDS = 'id,name,size,md5Checksum,appProperties,modifiedTime,parents,version'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def member(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe archive member: ' + name)
    return p


def extract_zip(archive, destination):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        for info in z.infolist():
            member(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlink in ZIP')
        if z.testzip() is not None:
            raise ValueError('Corrupt ZIP')
        z.extractall(destination)


class DriveBlocked(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def error_detail(exc):
    """Keep useful failure evidence without logging signed upload URLs."""
    message = re.sub(r'https?://\S+', '[URL omitted]', str(exc))[:700]
    return f'{type(exc).__name__}: {message}'


def transient_error(exc):
    """Retry transport interruptions and explicit throttling, never corruption."""
    if isinstance(exc, (TimeoutError, ConnectionError, http.client.IncompleteRead,
                        http.client.RemoteDisconnected, ssl.SSLEOFError)):
        return True
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    if isinstance(exc, OSError) and exc.errno in (
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
            errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE):
        return True
    status = getattr(getattr(exc, 'resp', None), 'status', None)
    try:
        reasons = {e.get('reason') for e in json.loads(exc.content).get('error', {}).get('errors', [])}
    except (AttributeError, TypeError, ValueError):
        reasons = set()
    return status in (429, 500, 502, 503, 504) or bool(
        reasons & {'rateLimitExceeded', 'userRateLimitExceeded'})


def close_connections(request):
    # The request keeps its credentials, body, ID and resumable-session state.
    # Only the failed thread-local socket pool is discarded.
    transport = getattr(request, 'http', None)
    transport = getattr(transport, 'http', transport)
    close = getattr(transport, 'close', None)
    if callable(close):
        close()


def retry_wait(attempt, label, exc):
    delay = min(2**(attempt+1)+random.SystemRandom().random(), 32)
    print('DRIVE RETRY:', label, '|', error_detail(exc),
          f'| attempt {attempt+2}/6 in {delay:.0f}s; training waits for acknowledgment', flush=True)
    time.sleep(delay)


def pace_write():
    """Pace worker mutations on local SSD; never write progress to Drive."""
    config = os.environ.get('HIPPO_CLOUD_CONFIG')
    if not config:
        return
    root = Path(json.loads(Path(config).read_text())['local_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'api_write_pacing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root/'api_write_pacing.json'
        previous = json.loads(target.read_text()) if target.exists() else {}
        delay = min(.5, max(0., previous.get('next_write', 0.)-time.time()))
        if delay:
            time.sleep(delay)
        atomic(target, {'next_write':time.time()+.5})


class Cloud:
    def __init__(self, folder=None, credentials=None):
        import google.auth
        self.credentials = credentials or google.auth.default()[0]
        self.folder = folder
        self.local = threading.local()
        self.owned = set()

    def service(self):
        if not hasattr(self.local, 'service'):
            import google_auth_httplib2
            from googleapiclient.discovery import build
            from googleapiclient.http import build_http
            # Timeout belongs to the transport, not individual requests.
            # build_http also removes 308 from httplib2.redirect_codes. Drive's
            # 308 means Resume Incomplete, NOT an HTTP redirect. A raw Http()
            # breaks multi-chunk checkpoints while small uploads still succeed.
            transport = build_http()
            transport.timeout = 120
            http = google_auth_httplib2.AuthorizedHttp(self.credentials, http=transport)
            self.local.service = build('drive', 'v3', http=http, cache_discovery=False)
        return self.local.service

    def execute(self, request):
        for attempt in range(6):
            try:
                return request.execute(num_retries=0)
            except Exception as exc:
                # Metadata GETs are safe to repeat. Uploads have their own
                # ID/checksum reconciliation loop below; unknown other mutations
                # remain the caller's responsibility (for example leases).
                is_http_error = getattr(getattr(exc, 'resp', None), 'status', None) is not None
                if not transient_error(exc) or (not is_http_error and getattr(request, 'method', None) != 'GET'):
                    raise
                if attempt == 5:
                    raise DriveBlocked('Drive request failed after bounded retries: '+error_detail(exc)) from exc
                close_connections(request)
                retry_wait(attempt, 'metadata request', exc)
        raise AssertionError('Unreachable')

    def about(self):
        return self.execute(self.service().about().get(fields='user(emailAddress),storageQuota'))

    def check_account(self):
        about = self.about()
        if about['user']['emailAddress'].lower() != ACCOUNT:
            raise RuntimeError('Authenticate '+ACCOUNT+' for this MSD campaign.')
        return about

    def children(self, parent=None):
        result, token = [], None
        while True:
            page = self.execute(self.service().files().list(q="'"+(parent or self.folder)+"' in parents and trashed=false",
                pageSize=1000, pageToken=token, fields='nextPageToken,files('+FIELDS+',mimeType)'))
            result.extend(page.get('files', [])); token = page.get('nextPageToken')
            if not token:
                return result

    def ensure_folder(self):
        self.check_account()
        matches = [x for x in self.children(CAMPAIGN) if x['name'] == STORE_NAME]
        if len(matches) > 1:
            raise RuntimeError('Duplicate destination folders; refusing to select silently')
        if matches:
            self.folder = matches[0]['id']
        else:
            value = self.execute(self.service().files().create(body={'name':STORE_NAME,
                'mimeType':'application/vnd.google-apps.folder','parents':[CAMPAIGN]}, fields='id'))
            self.folder = value['id']
        return self.folder

    def metadata(self, file_id):
        return self.execute(self.service().files().get(fileId=file_id, fields=FIELDS))

    def download(self, file_id, path, expected_sha=None):
        from googleapiclient.http import MediaIoBaseDownload
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.metadata(file_id)
        if path.exists() and self.matches(path, meta) and (not expected_sha or digest(path) == expected_sha):
            return path
        tmp = path.with_name(path.name+'.download')
        with tmp.open('wb') as f:
            stream = MediaIoBaseDownload(f, self.service().files().get_media(fileId=file_id), chunksize=8*1024**2)
            done = False
            while not done:
                _, done = stream.next_chunk(num_retries=5)
        if not self.matches(tmp, meta) or (expected_sha and digest(tmp) != expected_sha):
            raise IntegrityError('Downloaded bytes differ from server checksum or recorded SHA: '+meta['name'])
        tmp.replace(path)
        return path

    @staticmethod
    def matches(path, meta):
        return (int(meta.get('size', -1)) == Path(path).stat().st_size
                and bool(meta.get('md5Checksum')) and digest(path, 'md5') == meta['md5Checksum'])

    def upload(self, path, name=None, properties=None, file_id=None):
        """Immutable create by preallocated ID; mutable writes only with an explicit ID.

        A timed-out final response is resolved by querying that same ID. An
        unverified response never permits checkpoint deletion or another update.
        """
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError
        path = Path(path)
        if not self.folder:
            raise ValueError('Missing destination folder')
        sha = digest(path)
        # One exclusive task owns each mutable ledger. Reusing its exact server
        # acknowledgment is safe; a changed charge always takes a new write.
        acknowledged = getattr(self, '_acknowledged', {})
        if file_id is not None and acknowledged.get(file_id, {}).get('local_sha256') == sha:
            return acknowledged[file_id]
        creating = file_id is None
        if creating:
            file_id = self.execute(self.service().files().generateIds(count=1, space='drive'))['ids'][0]
        media = MediaFileUpload(str(path), mimetype='application/octet-stream',
                                chunksize=16*1024**2, resumable=path.stat().st_size > 4*1024**2)
        if creating:
            request = self.service().files().create(body={'id':file_id,'name':name or path.name,
                'parents':[self.folder], 'appProperties':properties or {}}, media_body=media, fields=FIELDS)
        else:
            if file_id not in self.owned:
                current = self.execute(self.service().files().get(fileId=file_id, fields='parents'))
                if current.get('parents') != [self.folder]:
                    raise RuntimeError('Refusing to update a file outside the new result store')
                self.owned.add(file_id)
            request = self.service().files().update(fileId=file_id, media_body=media, fields=FIELDS)
        result = None
        failures = 0
        last_progress = time.monotonic()
        while result is None:
            try:
                pace_write()
                if media.resumable():
                    progress, result = request.next_chunk(num_retries=0)
                    if progress and time.monotonic()-last_progress >= 30:
                        print('UPLOADING', name or path.name, round(100*progress.progress()),
                              '% | server verification pending', flush=True)
                        last_progress = time.monotonic()
                else:
                    result = request.execute(num_retries=0)
                failures = 0
            except Exception as exc:
                # A lost response may mean it DID commit. Check the SAME ID
                # before retrying the SAME bytes; never allocate a replacement.
                close_connections(request)
                reconciliation = None
                check_request = self.service().files().get(fileId=file_id, fields=FIELDS)
                try:
                    # One reconciliation read per bounded upload attempt, not
                    # a nested six-by-six retry loop during a persistent outage.
                    meta = check_request.execute(num_retries=0)
                except Exception as check_exc:
                    close_connections(check_request)
                    reconciliation = check_exc
                    meta = None
                if meta is not None and self.matches(path, meta):
                    result = meta
                    break
                if (meta is not None and meta.get('parents') != [self.folder]):
                    raise IntegrityError('Upload destination ownership differs; local file retained')
                if (creating and meta is not None and not media.resumable()):
                    raise IntegrityError('Preallocated upload ID contains different bytes; local file retained')
                check_status = getattr(getattr(reconciliation, 'resp', None), 'status', None)
                check_retryable = (reconciliation is None or transient_error(reconciliation)
                                   or isinstance(reconciliation, DriveBlocked)
                                   or (creating and check_status == 404))
                if not transient_error(exc) or not check_retryable or failures >= 5:
                    detail = ' | reconciliation: '+error_detail(reconciliation) if reconciliation else ''
                    raise DriveBlocked('Upload failed after reconciliation; local file retained: '+path.name+
                                       ' | '+error_detail(exc)+detail) from None
                retry_wait(failures, name or path.name, exc)
                failures += 1
        # This metadata is Google's upload response, not a FUSE cache read.
        # Use an extra GET only to reconcile an incomplete/unknown response.
        meta=result if self.matches(path,result) else self.metadata(file_id)
        if not self.matches(path, meta):
            raise DriveBlocked('Server checksum verification failed: '+path.name)
        self.owned.add(file_id)
        receipt = dict(meta, local_sha256=sha, verified_at=time.time())
        acknowledged[file_id] = receipt
        self._acknowledged = acknowledged
        return receipt

    def put_immutable(self, path, kind, task='', phase='', *, priority=None):
        sha = digest(path)
        name = kind+'__'+phase+'__'+task+'__'+sha+Path(path).suffix
        escaped=name.replace('\\','\\\\').replace("'","\\'")
        found=self.execute(self.service().files().list(q="'"+self.folder+"' in parents and trashed=false and name='"+escaped+"'",
            pageSize=100,fields='files('+FIELDS+')')).get('files',[])
        if len(found) > 1:
            raise RuntimeError('Duplicate immutable objects: '+name)
        if found:
            if not self.matches(path, found[0]):
                raise RuntimeError('Immutable remote object changed')
            return dict(found[0], local_sha256=sha)
        properties={'kind':kind,'task':task,'phase':phase,'sha256':sha}
        if priority is not None:
            properties['upload_priority']=priority
        return self.upload(path, name, properties)


@contextmanager
def upload_slot(directory):
    """One bulky snapshot upload per VM; ledgers use separate small API requests."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'checkpoint_upload.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
