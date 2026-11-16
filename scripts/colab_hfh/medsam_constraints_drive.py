"""Verified MedSAM storage using Colab ADC, independent of rclone's OAuth app."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import threading
import time

ROOT_ID = os.environ.get('HIPPO_HFH_DRIVE_ROOT', '')
EMAIL = os.environ.get('HIPPO_STORAGE_EMAIL', '')
FOLDER = 'application/vnd.google-apps.folder'
FIELDS = 'id,name,mimeType,size,md5Checksum'
_LOCAL = threading.local()


def credentials():
    """Refresh through the explicitly selected Google SDK account, never print tokens."""
    if os.environ.get('HFH_DRIVE_CREDENTIALS') != 'gcloud':
        import google.auth
        return google.auth.default()[0]
    from datetime import datetime, timedelta, timezone
    from google.auth.credentials import Credentials

    class AccountCredentials(Credentials):
        def refresh(self, request):
            result = subprocess.run(['gcloud', 'auth', 'print-access-token', '--account='+EMAIL,
                                     '--quiet'], capture_output=True, text=True, timeout=60)
            if result.returncode or not result.stdout.strip():
                raise RuntimeError('Second-account Google SDK authorization needs renewal')
            self.token = result.stdout.strip()
            self.expiry = (datetime.now(timezone.utc)+timedelta(minutes=5)).replace(tzinfo=None)

    return AccountCredentials()


def checksum(path):
    value = hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def parts(remote):
    if not remote.startswith('hfh_storage:'):
        raise ValueError('Only the HFH storage remote is permitted')
    name = remote.split(':', 1)[1].lstrip('/')
    result = PurePosixPath(name).parts
    if '..' in result or '\\' in name:
        raise ValueError('Unsafe Drive path')
    return result


class Drive:
    def __init__(self, service=None):
        if service is None:
            from googleapiclient.discovery import build
            service = build('drive', 'v3', credentials=credentials(), cache_discovery=False)
        self.service = service
        self.cache = {(): dict(id=ROOT_ID, mimeType=FOLDER, name='HFH')}
        about = service.about().get(fields='user(emailAddress),storageQuota').execute(num_retries=5)
        if about['user']['emailAddress'].lower() != EMAIL:
            raise ValueError('Drive ADC must authenticate '+EMAIL+'; no files were written')
        root = service.files().get(fileId=ROOT_ID, fields='id,mimeType,owners(emailAddress)').execute(num_retries=5)
        if root['mimeType'] != FOLDER or EMAIL not in [x['emailAddress'].lower() for x in root['owners']]:
            raise ValueError('HFH folder owner differs; no files were written')
        self.quota = about['storageQuota']

    def children(self, parent):
        result, token = [], None
        while True:
            page = self.service.files().list(q=f"'{parent}' in parents and trashed=false",
                fields='nextPageToken,files('+FIELDS+')', pageSize=1000, pageToken=token).execute(num_retries=5)
            result.extend(page.get('files', []))
            token = page.get('nextPageToken')
            if not token:
                return result

    def resolve(self, remote, create=False):
        path = parts(remote)
        for depth in range(1, len(path)+1):
            prefix = path[:depth]
            if prefix in self.cache:
                continue
            parent = self.cache[prefix[:-1]]
            if parent['mimeType'] != FOLDER:
                raise ValueError('Drive path parent is not a folder')
            matches = [x for x in self.children(parent['id']) if x['name'] == prefix[-1]]
            if len(matches) > 1:
                raise ValueError('Ambiguous duplicate Drive path: '+str(prefix))
            if not matches:
                if not create:
                    raise FileNotFoundError(remote)
                # Preallocated IDs make a retried create idempotent.
                file_id = self.service.files().generateIds(count=1, space='drive').execute(num_retries=5)['ids'][0]
                body = dict(id=file_id, name=prefix[-1], mimeType=FOLDER, parents=[parent['id']])
                try:
                    row = self.service.files().create(body=body, fields=FIELDS).execute(num_retries=0)
                except Exception:
                    # Never blindly replay a create with an uncertain outcome.
                    row = self.service.files().get(fileId=file_id, fields=FIELDS).execute(num_retries=5)
                matches = [row]
            self.cache[prefix] = matches[0]
        return self.cache[path]

    def metadata(self, file_id):
        return self.service.files().get(fileId=file_id, fields=FIELDS).execute(num_retries=5)

    def download(self, remote, destination):
        from googleapiclient.http import MediaIoBaseDownload
        row = self.metadata(self.resolve(remote)['id'])
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == int(row['size']) and checksum(target) == row['md5Checksum']:
            return
        temporary = target.with_name(target.name+'.downloading')
        with temporary.open('wb') as stream:
            transfer = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=row['id']), chunksize=16*1024**2)
            done = False
            while not done:
                _, done = transfer.next_chunk(num_retries=5)
        if temporary.stat().st_size != int(row['size']) or checksum(temporary) != row['md5Checksum']:
            raise IOError('Downloaded Drive content failed independent checksum verification')
        temporary.replace(target)

    def upload(self, source, remote):
        from googleapiclient.http import MediaFileUpload
        source = Path(source)
        path = parts(remote)
        if not path:
            raise ValueError('Cannot upload over HFH root')
        parent = self.resolve('hfh_storage:'+'/'.join(path[:-1]), create=True)
        expected = checksum(source)
        try:
            row = self.resolve(remote)
        except FileNotFoundError:
            row = None
        file_id = row['id'] if row else self.service.files().generateIds(count=1, space='drive').execute(num_retries=5)['ids'][0]
        expected_size=source.stat().st_size
        last_error=None
        verified={}
        for attempt in range(4):
            if source.stat().st_size!=expected_size or checksum(source)!=expected:
                raise IOError('Local upload source changed; retained for inspection')
            media=MediaFileUpload(str(source),mimetype='application/octet-stream',chunksize=16*1024**2,resumable=True)
            if row:
                request=self.service.files().update(fileId=file_id,media_body=media,fields=FIELDS)
            else:
                request=self.service.files().create(body=dict(id=file_id,name=path[-1],parents=[parent['id']]),media_body=media,fields=FIELDS)
            response=None
            try:
                while response is None:
                    _,response=request.next_chunk(num_retries=5)
            except Exception as error:
                # Preserve the actual HTTP/transport cause, rather than hiding
                # a failed PATCH behind a later checksum mismatch.
                last_error=error
            for check in range(3):
                try:
                    verified=self.metadata(file_id)
                    row=verified  # An uncertain create must retry as an update.
                    if int(verified.get('size',-1))==expected_size and verified.get('md5Checksum')==expected:
                        self.cache[path]=verified
                        return
                except Exception as error:
                    last_error=error
                if check<2:time.sleep(2*(check+1))
            if attempt<3:
                print('DRIVE_WRITE_RETRY',path[-1],attempt+1,
                      type(last_error).__name__ if last_error else 'metadata checksum not yet matched',flush=True)
                time.sleep(2**(attempt+1))
        detail=dict(file_id=file_id,expected_bytes=expected_size,expected_md5=expected,
                    observed_bytes=verified.get('size'),observed_md5=verified.get('md5Checksum'),
                    cause=type(last_error).__name__ if last_error else None,
                    http_status=getattr(getattr(last_error,'resp',None),'status',None))
        raise IOError('Uploaded Drive content failed independent checksum verification: '+json.dumps(detail)) from last_error

    @staticmethod
    def listing(row):
        return dict(ID=row['id'], Name=row['name'], Size=int(row.get('size', 0)),
                    IsDir=row['mimeType'] == FOLDER, Hashes={'md5': row.get('md5Checksum', '')})

    def command(self, *args):
        operation, remote, *options = args
        if operation == 'about':
            used = int(self.quota.get('usage', 0))
            total = int(self.quota.get('limit', 2**63-1))
            return json.dumps(dict(used=used, total=total, free=total-used))
        if operation == 'mkdir':
            self.resolve(remote, create=True)
            return ''
        if operation == 'copyto':
            destination = options[0]
            if remote.startswith('hfh_storage:'):
                self.download(remote, destination)
            else:
                self.upload(remote, destination)
            return ''
        if operation == 'lsjson':
            row = self.resolve(remote)
            if '--stat' in options:
                return json.dumps(self.listing(self.metadata(row['id'])))
            rows = self.children(row['id'])
            if '--files-only' in options:
                rows = [x for x in rows if x['mimeType'] != FOLDER]
            if '--dirs-only' in options:
                rows = [x for x in rows if x['mimeType'] == FOLDER]
            return json.dumps([self.listing(x) for x in rows])
        if operation == 'delete':
            # Checkpoint retirement remains a separate explicit archive operation.
            raise OSError('Remote recovery retained; ADC transport does not delete recovery')
        raise ValueError('Unsupported Drive transport operation: '+operation)


def client():
    if not hasattr(_LOCAL, 'drive'):
        _LOCAL.drive = Drive()
    return _LOCAL.drive
