"""Resolve experiment folders directly, retaining verified rclone transfers."""
import os
import configparser
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import time
import uuid

from scripts.colab_hfh.bands_campaign import Drive as VerifiedDrive


STORAGE_ROOT = os.environ.get('HIPPO_HFH_DRIVE_ROOT', '')
STORAGE_EMAIL = os.environ.get('HIPPO_STORAGE_EMAIL', '')
FOLDER_MIME = 'application/vnd.google-apps.folder'
FILE_FIELDS = 'id,name,mimeType,parents,owners(emailAddress),size,md5Checksum,trashed'
MAX_REQUEST_ATTEMPTS = 12
QUOTA_REASONS = {'rateLimitExceeded', 'userRateLimitExceeded'}
CACHE_DIRECTORY = Path('/content/.hfh_storage_auth/constraint_drive_cache')
def load_metadata_cache(folder):
    if not re.fullmatch(r'results_[a-z0-9_]+', folder):
        raise ValueError('Unexpected experiment folder name')
    path = CACHE_DIRECTORY/(folder+'.json')
    if not path.exists():
        return None, {}
    value = json.loads(path.read_text())
    if value.get('root') != STORAGE_ROOT or value.get('owner') != STORAGE_EMAIL or value.get('name') != folder:
        raise ValueError('Drive metadata cache scope differs')
    folder_id, files = value['folder_id'], value['files']
    if (not isinstance(files, dict) or not isinstance(folder_id, str)
            or not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id)
            or any(not isinstance(k,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',k)
                   or not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',v)
                   for k,v in files.items())):
        raise ValueError('Invalid Drive metadata cache')
    return folder_id, files


def save_metadata_cache(folder, folder_id, files):
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIRECTORY/(folder+'.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(root=STORAGE_ROOT, owner=STORAGE_EMAIL,
        name=folder, folder_id=folder_id, files=files), sort_keys=True)+'\n')
    temporary.replace(path)


def upload_verified(request, folder_id, path, name=None, *, file_ids=None, fresh_folder=False):
    """Upload a small result file and verify its identity and server checksum."""
    name = name or path.name
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError('Unexpected result file name')
    payload = path.read_bytes()
    mime = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    expected_md5 = hashlib.md5(payload).hexdigest()

    def identity(item):
        if (not item.get('id') or item.get('name') != name
                or folder_id not in item.get('parents', [])
                or {x.get('emailAddress') for x in item.get('owners', [])} != {STORAGE_EMAIL}
                or item.get('trashed', False)
                or item.get('mimeType', '').startswith('application/vnd.google-apps.')):
            raise ValueError('Unexpected Drive result identity or owner')

    if file_ids is not None and name in file_ids:
        # Revalidate a previously resolved exact ID before every write. Avoid
        # repeated name-search queries, which have a separate project quota.
        found = {'files': [request('GET', 'files/'+file_ids[name],params={'fields': FILE_FIELDS})]}
    elif fresh_folder:
        # This process just created the folder. No previous trial files can
        # exist; avoid an expensive name query before the first create.
        # Lost POST responses still abort, and a restarted process queries.
        found = {'files': []}
    else:
        found = request('GET', 'files', params={
            'q': f"'{folder_id}' in parents and name='{name}' and trashed=false",
            'fields': f'nextPageToken,files({FILE_FIELDS})', 'pageSize': 100})
    matches = found.get('files', [])
    if found.get('nextPageToken') or len(matches) > 1:
        raise ValueError('Ambiguous Drive result file')
    if matches:
        remote = matches[0]
        identity(remote)
        if int(remote.get('size', -1)) == len(payload) and remote.get('md5Checksum') == expected_md5:
            if file_ids is not None:file_ids[name]=remote['id']
            return
        uploaded = request('PATCH', 'files/'+remote['id'], upload=True,
            params={'uploadType': 'media', 'fields': FILE_FIELDS},
            headers={'Content-Type': mime}, data=payload)
    else:
        boundary = 'hfh_'+uuid.uuid4().hex
        metadata = json.dumps({'name': name, 'parents': [folder_id], 'mimeType': mime}).encode()
        body = (f'--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'.encode()
                + metadata + f'\r\n--{boundary}\r\nContent-Type: {mime}\r\n\r\n'.encode()
                + payload + f'\r\n--{boundary}--\r\n'.encode())
        # Do not blindly retry a lost POST response: the next invocation will
        # first discover the exact name and verify any successfully created file.
        uploaded = request('POST', 'files', upload=True,
            params={'uploadType': 'multipart', 'fields': FILE_FIELDS},
            headers={'Content-Type': 'multipart/related; boundary='+boundary}, data=body)
    identity(uploaded)
    # Independent metadata read confirms persisted bytes before cleanup/receipt.
    saved = request('GET', 'files/'+uploaded['id'], params={'fields': FILE_FIELDS})
    identity(saved)
    if (int(saved.get('size', -1)) != len(payload)
            or saved.get('md5Checksum') != expected_md5):
        raise ValueError('Remote checksum mismatch')
    if file_ids is not None:file_ids[name]=saved['id']


def resolve_folder(request, name, folder_id=None):
    """Exact server-side lookup avoids scanning hundreds of experiment folders."""
    if not re.fullmatch(r'results_[a-z0-9_]+', name):
        raise ValueError('Unexpected experiment folder name')
    fields = 'id,name,mimeType,parents,owners(emailAddress),trashed'
    if folder_id is not None:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',folder_id):
            raise ValueError('Invalid cached Drive folder ID')
        result = {'files': [request('GET','files/'+folder_id,params={'fields':fields})]}
    else:
        result = request('GET', 'files', params={
            'q': f"'{STORAGE_ROOT}' in parents and name='{name}' and trashed=false",
            'fields': f'nextPageToken,files({fields})', 'pageSize': 100})
    matches = result.get('files', [])
    if result.get('nextPageToken') or len(matches) > 1:
        raise ValueError('Ambiguous Drive campaign folder')
    if matches:
        folder = matches[0]
    else:
        # A failed create is not retried blindly: the next launch first queries
        # the exact name again, avoiding duplicate folders after lost responses.
        folder = request('POST', 'files', params={'fields': fields}, json={
            'name': name, 'mimeType': FOLDER_MIME, 'parents': [STORAGE_ROOT]})
    if (folder.get('name') != name or folder.get('mimeType') != FOLDER_MIME or folder.get('trashed',False)
            or STORAGE_ROOT not in folder.get('parents', [])
            or {x.get('emailAddress') for x in folder.get('owners', [])} != {STORAGE_EMAIL}):
        raise ValueError('Unexpected Drive folder identity or owner')
    return folder['id']


class Drive(VerifiedDrive):
    def __init__(self, folder):
        import requests
        private = Path('/content/.hfh_storage_auth')
        config = private/'rclone.conf'
        self.command = [str(private/'rclone'), '--config', str(config),
                        '--tpslimit', '2', '--tpslimit-burst', '2',
                        '--retries', '3', '--low-level-retries', '10']
        session = requests.Session()
        last_request = 0.
        self.request_attempts = MAX_REQUEST_ATTEMPTS

        def request(method, endpoint, *, upload=False, **kwargs):
            nonlocal last_request
            extra_headers = kwargs.pop('headers', {})
            for refresh in range(2):
                try:
                    parser = configparser.ConfigParser(interpolation=None)
                    parser.read(config)
                    section = parser['hfh_storage']
                    root_id = section.get('root_folder_id')
                    token = json.loads(section['token'])['access_token']
                except (configparser.Error, KeyError, ValueError):
                    raise RuntimeError('Storage credential configuration unreadable; details omitted') from None
                if root_id != STORAGE_ROOT:
                    raise ValueError('Unexpected storage root')
                # Observed project-quota bursts outlasted the previous 123s
                # backoff window. Allow 363s of bounded backoff; never replay an
                # ambiguous POST response or treat permission errors as quota.
                attempts = self.request_attempts
                for attempt in range(attempts):
                    try:
                        # Retain the existing two-requests/second ceiling while
                        # avoiding repeated CLI startup and path discovery.
                        time.sleep(max(0., .5-(time.monotonic()-last_request)))
                        last_request = time.monotonic()
                        base = 'https://www.googleapis.com/'+('upload/' if upload else '')+'drive/v3/'
                        response = session.request(method, base+endpoint,
                            headers={**extra_headers, 'Authorization': 'Bearer '+token}, timeout=60, **kwargs)
                    except requests.RequestException:
                        if method == 'GET' and attempt < attempts-1:
                            time.sleep(min(60,2**attempt))
                            continue
                        raise RuntimeError('Drive folder request failed; safe to resume after connectivity recovers') from None
                    try:
                        reasons = [x.get('reason','unknown') for x in response.json().get('error',{}).get('errors',[])]
                        reasons = [x for x in reasons if isinstance(x,str) and re.fullmatch(r'[A-Za-z_]{1,60}',x)]
                    except (ValueError,TypeError,AttributeError):
                        reasons = []
                    transient = response.status_code == 429 or (
                        response.status_code == 403 and bool(QUOTA_REASONS.intersection(reasons))) or (
                        method == 'GET' and response.status_code in (500, 502, 503, 504))
                    if transient and attempt < attempts-1:
                        delay = min(60,2**attempt)
                        print('DRIVE RETRY', method, 'HTTP', response.status_code,
                              ','.join(reasons) or 'unspecified',
                              f'attempt {attempt+1}/{attempts}; waiting {delay}s', flush=True)
                        time.sleep(delay)
                        continue
                    break
                if response.status_code == 401 and refresh == 0:
                    result = subprocess.run(self.command+['about', 'hfh_storage:', '--json'],
                                            capture_output=True, timeout=120)
                    if result.returncode:
                        raise RuntimeError('Storage credential refresh failed; token details omitted')
                    continue
                if response.status_code not in (200, 201):
                    raise RuntimeError('Drive folder request failed with HTTP '+str(response.status_code)+' '+','.join(reasons))
                return response.json()
            raise RuntimeError('Storage authorization unavailable')

        started = time.monotonic()
        cached_folder, cached_files = load_metadata_cache(folder)
        created = False
        def resolve_request(method, endpoint, **kwargs):
            nonlocal created
            result = request(method, endpoint, **kwargs)
            if method == 'POST':created = True
            return result
        folder_id = resolve_folder(resolve_request, folder, cached_folder)
        self.request, self.session, self.folder_id = request, session, folder_id
        self.cache_name, self.file_ids = folder, cached_files
        self.fresh_folder = created
        save_metadata_cache(folder, folder_id, self.file_ids)
        self.command += ['--drive-root-folder-id', folder_id]
        self.folder = 'hfh_storage:'
        print('DRIVE FOLDER READY', folder, f'{time.monotonic()-started:.2f}s', flush=True)

    def call(self, *args):
        if (self.fresh_folder and not self.file_ids and len(args)>1
                and args[0]=='copy' and args[1]==self.folder):
            # No remote recovery files exist in a folder just created by this
            # process. Existing folders always retain the recovery download.
            return ''
        return super().call(*args)

    def upload(self, path, name=None):
        started = time.monotonic()
        if path.stat().st_size > 8*1024*1024:
            super().upload(path, name)
        else:
            upload_verified(self.request, self.folder_id, path, name,file_ids=self.file_ids,
                            fresh_folder=self.fresh_folder)
            save_metadata_cache(self.cache_name, self.folder_id, self.file_ids)
        print('DRIVE FILE VERIFIED', name or path.name,
              f'{time.monotonic()-started:.2f}s', flush=True)

    def upload_report(self, path):
        """Defer only quota-throttled derived reports; durable results stay strict."""
        if path.name not in ('full_constraint_matrix.csv', 'matrix_summary.json'):
            raise ValueError('Only rebuildable matrix reports may use deferred synchronization')
        original = self.request_attempts
        self.request_attempts = 2
        try:
            self.upload(path)
            return True
        except RuntimeError as error:
            message = str(error)
            quota = ('HTTP 429' in message or ('HTTP 403' in message
                     and any(reason in message for reason in QUOTA_REASONS)))
            if not quota:
                raise
            print('REPORT SYNC DEFERRED', path.name, 'Drive quota; verified experiment archives are retained', flush=True)
            return False
        finally:
            self.request_attempts = original
