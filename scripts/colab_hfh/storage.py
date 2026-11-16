"""Notebook bootstrap: third-account compute, second-account owned Drive storage.

This file is inlined into notebooks; it is not part of the frozen training bundle.
"""
import configparser
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

COMPUTE_EMAIL = os.environ.get('HIPPO_COMPUTE_EMAIL', '')
STORAGE_EMAIL = os.environ.get('HIPPO_STORAGE_EMAIL', '')
STORAGE_FOLDER_ID = os.environ.get('HIPPO_HFH_DRIVE_ROOT', '')
FOLDER_NAME = 'hippo_hfh_exploration_20261111'
RCLONE_VERSION = '1.74.3'
RCLONE_DRIVE_FILE_ID = os.environ.get('HIPPO_RCLONE_FILE_ID', '')
RCLONE_BINARY_SHA = '9700aa1273ac73d6d0833c43ba63fe830516422cb131960b8c1a24ced789cba0'
RCLONE_ZIP_SHA = 'dbee7ccd7a5d617e4ed4cd4555c16669b511abfe8d31164f61be35ac9e999bd2'
TRANSPORT_SHA = '7c7080209ad6484ba01b8f4aecc91424c61060968e05e0076b706899f76ac7e7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require_storage_account(about):
    email = about.get('user', {}).get('emailAddress', '').lower()
    if email != STORAGE_EMAIL:
        raise RuntimeError('Storage login must be '+STORAGE_EMAIL+'; received '+email)
    quota = about['storageQuota']
    if 'limit' in quota and int(quota['limit'])-int(quota['usage']) < 8*1024**3:
        raise RuntimeError('Keep at least 8 GiB free on the SECOND account.')


def require_owner(metadata):
    owners = {o.get('emailAddress', '').lower() for o in metadata.get('owners', [])}
    if owners != {STORAGE_EMAIL}:
        raise RuntimeError('Storage object is not owned by the second account.')


def write_token_config(token_text, config):
    try:
        token = json.loads(token_text)
        valid = isinstance(token, dict) and all(isinstance(token.get(k), str) and token[k]
            for k in ('access_token', 'refresh_token', 'token_type', 'expiry'))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError('Expected the JSON token from rclone authorize drive; token contents omitted.') from None
    config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.parent.chmod(0o700)
    with os.fdopen(os.open(config, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as out:
        out.write('[hfh_storage]\ntype = drive\nscope = drive\nroot_folder_id = '+STORAGE_FOLDER_ID+
            '\ntoken = '+json.dumps(token)+'\n')
    config.chmod(0o600)


def install_rclone(private, compute_service):
    from googleapiclient.http import MediaIoBaseDownload
    binary = private/'rclone'
    if not binary.exists() or sha(binary) != RCLONE_BINARY_SHA:
        archive = private/'rclone.zip'
        print('Loading verified storage helper from Drive...', flush=True)
        with archive.open('wb') as out:
            download = MediaIoBaseDownload(out, compute_service.files().get_media(
                fileId=RCLONE_DRIVE_FILE_ID), chunksize=8*1024**2)
            done = False
            while not done:
                _, done = download.next_chunk(num_retries=6)
        if sha(archive) != RCLONE_ZIP_SHA:
            raise ValueError('rclone release checksum mismatch')
        temporary = private/'rclone.installing'
        with zipfile.ZipFile(archive) as z:
            temporary.write_bytes(z.read('rclone-v'+RCLONE_VERSION+'-linux-amd64/rclone'))
        if sha(temporary) != RCLONE_BINARY_SHA:
            raise ValueError('rclone binary checksum mismatch')
        temporary.chmod(0o700)
        temporary.replace(binary)
        archive.unlink()
    # This rclone build needs fusermount3 even when Colab has FUSE2.
    if not shutil.which('fusermount3'):
        log = private/'fuse-install.log'
        for attempt in range(3):
            with log.open('a') as output:
                update = subprocess.run(['apt-get', '-o', 'Acquire::Retries=2',
                    '-o', 'Acquire::http::Timeout=20', 'update'],
                    stdout=output, stderr=subprocess.STDOUT, timeout=180)
                install = subprocess.run(['apt-get', '-o', 'Acquire::Retries=2',
                    '-o', 'Acquire::http::Timeout=20', 'install', '-y', 'fuse3'],
                    stdout=output, stderr=subprocess.STDOUT, timeout=180)
            if install.returncode == 0 and shutil.which('fusermount3'):
                break
            print('FUSE3 install retry', attempt+1, flush=True)
        else:
            print('\n'.join(log.read_text().splitlines()[-15:]), flush=True)
            raise RuntimeError('FUSE3 installation failed; inspect the printed package-manager error.')
    return binary


def storage_bootstrap(kind, *, campaign='v1', transport_name='transport_manifest.json',
                      transport_sha=TRANSPORT_SHA):
    from google.colab import auth, drive, userdata, files
    import google.auth
    from googleapiclient.discovery import build
    import requests

    # Colab billing follows the browser/runtime account. API identity is an extra
    # guard, not a query of Colab's billing balance.
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    compute_service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    compute = compute_service.about().get(fields='user(emailAddress)').execute()
    if compute['user']['emailAddress'].lower() != COMPUTE_EMAIL:
        raise RuntimeError('Open Colab and authenticate compute as '+COMPUTE_EMAIL)

    private = Path('/content/.hfh_storage_auth')
    private.mkdir(mode=0o700, exist_ok=True)
    private.chmod(0o700)
    binary = install_rclone(private, compute_service)
    config = private/'rclone.conf'
    if not config.exists():
        try:
            token_text = userdata.get('HFH_SECOND_DRIVE_TOKEN')
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
            print('Select hfh-second-drive-token.json created by the local authorization helper.')
            uploaded = files.upload(target_dir=str(private))
            if len(uploaded) != 1:
                raise RuntimeError('Select exactly one storage authorization JSON file.')
            filename, token_bytes = next(iter(uploaded.items()))
            token_text = token_bytes.decode()
            uploaded_path = private/Path(filename).name
            if uploaded_path.is_file():
                uploaded_path.unlink()
            del uploaded, token_bytes
        write_token_config(token_text, config)
        del token_text

    def rc(*args, binary_output=False):
        result = subprocess.run([str(binary), '--config', str(config), *map(str, args)],
            capture_output=True, timeout=600)
        if result.returncode:
            # Never propagate OAuth responses or token configuration into saved outputs.
            raise RuntimeError('Storage command failed ('+str(args[0])+'); reauthorize the second account if expired.')
        return result.stdout if binary_output else result.stdout.decode()

    rc('about', 'hfh_storage:', '--json')  # Refresh OAuth before API identity checks.
    def api_get(path, **params):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(config)
        token = json.loads(parser['hfh_storage']['token'])['access_token']
        for attempt in range(6):
            response = requests.get('https://www.googleapis.com/drive/v3/'+path,
                headers={'Authorization': 'Bearer '+token}, params=params, timeout=60)
            if response.status_code == 200 or response.status_code not in (403, 429, 500, 502, 503, 504):
                break
            time.sleep(2*(attempt+1))
        if response.status_code != 200:
            raise RuntimeError('Storage verification API failed with HTTP '+str(response.status_code))
        return response.json()

    about = api_get('about', fields='user(emailAddress),storageQuota')
    require_storage_account(about)
    require_owner(api_get('files/'+STORAGE_FOLDER_ID, fields='id,owners(emailAddress)'))
    mount = Path('/content/hfh_storage')
    mount.mkdir(exist_ok=True)
    if not os.path.ismount(mount):
        # No delayed VFS writeback: recovery reads must come from the remote.
        process = subprocess.Popen([str(binary), '--config', str(config), 'mount', 'hfh_storage:', str(mount),
            '--vfs-cache-mode', 'off', '--dir-cache-time', '5m', '--poll-interval', '30s',
            '--tpslimit', '2', '--tpslimit-burst', '2',
            '--drive-chunk-size', '16M', '--buffer-size', '8M', '--file-perms', '0600',
            '--dir-perms', '0700', '--log-file', str(private/'mount.log')],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(60):
            if os.path.ismount(mount):
                break
            if process.poll() is not None:
                raise RuntimeError('Second-account Drive mount failed; training has not started.')
            time.sleep(1)
        else:
            process.terminate()
            raise RuntimeError('Storage mount timed out; training has not started.')

    # Exercise the same create/fsync/rename/read/delete operations as recovery.
    # Check with independent Drive API calls so an unrelated existing mount fails closed.
    name = 'storage_probe_'+uuid.uuid4().hex
    temporary, final = mount/(name+'.tmp'), mount/name
    payload = os.urandom(4096)
    with temporary.open('wb') as out:
        out.write(payload); out.flush(); os.fsync(out.fileno())
    temporary.replace(final)
    try:
        listing = api_get('files', q="'"+STORAGE_FOLDER_ID+"' in parents and name='"+name+"' and trashed=false",
            fields='files(id,size,md5Checksum,owners(emailAddress))')['files']
        if len(listing) != 1:
            raise RuntimeError('Mounted storage did not publish the verification file.')
        require_owner(listing[0])
        if listing[0]['md5Checksum'] != hashlib.md5(payload).hexdigest() or final.read_bytes() != payload:
            raise RuntimeError('Remote storage write/read verification failed.')
        if rc('cat', 'hfh_storage:/'+name, binary_output=True) != payload:
            raise RuntimeError('Independent remote read failed.')
    finally:
        final.unlink(missing_ok=True)

    # Check campaign identity BEFORE staging large inputs. A changed baseline
    # recipe deliberately starts in a new namespace; v1 results are never reused.
    if campaign == 'v1':
        existing = compute_service.files().list(q="'"+os.environ['HIPPO_LEGACY_DRIVE_ROOT']+"' in parents and name='results_"+
            kind+"_v1' and trashed=false", fields='files(id)').execute()['files']
        for previous in existing:
            contents = compute_service.files().list(q="'"+previous['id']+"' in parents and trashed=false",
                fields='nextPageToken,files(id,name,mimeType)', pageSize=100).execute()
            if contents.get('nextPageToken') or any(item['name'] != 'setup.log' or
                    item['mimeType'] == 'application/vnd.google-apps.folder'
                    for item in contents.get('files', [])):
                raise RuntimeError('Existing third-account experiment artifacts detected. Migrate and audit recovery before switching storage.')
            print('Earlier third-account folder contains setup logs only; no experiment to replay. Original logs retained.', flush=True)

    # Copy inputs once into second-account ownership. No results are written to
    # the third account; its existing immutable source copy is only read.
    manifest = mount/transport_name
    if not manifest.exists():
        drive.mount('/content/compute_drive', readonly=True)
        source = Path('/content/compute_drive/MyDrive')/FOLDER_NAME/transport_name
        if sha(source) != transport_sha:
            raise ValueError('Original input manifest changed')
        rc('copyto', source, 'hfh_storage:/'+transport_name, '--checksum')
    if sha(manifest) != transport_sha:
        raise ValueError('Storage transport manifest changed')
    transport = json.loads(manifest.read_text())
    def mount_source():
        if not os.path.ismount('/content/compute_drive'):
            drive.mount('/content/compute_drive', readonly=True)
    assets = required_assets(transport, kind)
    total = sum(asset['bytes'] for asset in assets.values())
    print(f'Preparing {kind} inputs: {total/1024**2:.1f} MiB. '
          'SHA256 verification runs once while staging to local SSD.', flush=True)
    started = time.monotonic()
    for asset in assets.values():
        for part in asset['parts']:
            if Path(part['name']).name != part['name']:
                raise ValueError('Unsafe transport filename')
            target = mount/part['name']
            if not target.exists():
                mount_source()
                source = Path('/content/compute_drive/MyDrive')/FOLDER_NAME/part['name']
                if source.stat().st_size != part['bytes']:
                    raise ValueError('Source input size differs: '+part['name'])
                print('Copying missing second-account input:', part['name'], flush=True)
                rc('copyto', source, 'hfh_storage:/'+part['name'], '--checksum')
            for _ in range(10):
                if target.exists():
                    break
                time.sleep(1)
            if target.stat().st_size != part['bytes']:
                raise ValueError('Stored input size differs: '+part['name'])
            print('Input available (content verified in next cell):', part['name'], flush=True)

    rc('about', 'hfh_storage:', '--json')
    about = api_get('about', fields='user(emailAddress),storageQuota')
    require_storage_account(about)
    results = mount/('results_'+kind+'_'+campaign)
    results.mkdir(exist_ok=True)
    record = {'compute_account': COMPUTE_EMAIL, 'storage_account': STORAGE_EMAIL,
        'storage_folder_id': STORAGE_FOLDER_ID, 'remote_write_owner_verified': True,
        'quota': about['storageQuota'], 'input_transport_sha256': transport_sha,
        'campaign': campaign, 'required_input_bytes': total}
    (results/'storage_verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Compute account:', COMPUTE_EMAIL, '| Storage owner:', STORAGE_EMAIL)
    print('Results and recovery:', results)
    print(f'Storage preparation finished in {time.monotonic()-started:.1f}s; training has not started.', flush=True)
    return mount, results


def required_assets(transport, kind):
    """Baselines never access SAM3 weights, including during metadata checks."""
    if kind not in ('baselines', 'medsam3'):
        raise ValueError('Unknown HFH notebook kind')
    return {name: spec for name, spec in transport['assets'].items()
            if kind == 'medsam3' or name != 'sam3.pt'}


def notebook_cell(kind):
    source = Path(__file__).read_text().split('\ndef notebook_cell(')[0]
    return source+'\nKIND = '+repr(kind)+'''\nHOME, RESULTS = storage_bootstrap(KIND)
ROOT = Path('/content/hippo_hfh')
ROOT.mkdir(exist_ok=True)
BUNDLE = ROOT/'bundle'
SESSION_HOURS = 8
CANDIDATE_COUNT = 32
'''
