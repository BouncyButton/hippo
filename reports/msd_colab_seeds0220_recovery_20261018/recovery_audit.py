"""CPU-only cloud audit. Never changes source results or launches training."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import threading
import time
import zipfile

ACCOUNT = 'focacciafilippo@gmail.com'
CAMPAIGN_ROOT = '1ipls98K9dmRQ5rZwfbepw1JyWVL3Yq0P'
RUNS_ROOT = '1TmkvYx2ah8ba_fzmQkuyQzhCGy__0ZK_'
RECOVERY_ROOT = '1YSxdkYD8CYMArbK3wC9ZolWIgOHMEfcN'
SNAPSHOT_ID = '1v3Er1Sh-WStGXx3hB07NtKsoR4aKt3lJ'
API = 'https://www.googleapis.com/drive/v3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rate_limited(response):
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    try:
        reasons = {e.get('reason') for e in response.json().get('error', {}).get('errors', [])}
        return bool(reasons & {'rateLimitExceeded', 'userRateLimitExceeded'})
    except (ValueError, TypeError):
        return False


def safe_name(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe archive member: ' + name)
    return p


def verify_blob(name, data, expected):
    if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
        raise ValueError('Size/hash mismatch: ' + name)
    if name.endswith('.json'):
        json.loads(data)
    elif name.endswith('.jsonl'):
        for line in data.decode().splitlines():
            json.loads(line)
    elif not name.endswith('.png'):
        data.decode()


def assess_files(manifest, blobs):
    """Exact duplicate bytes can repair a staged copy, never the source."""
    failures, candidates = [], {}
    for name, expected in manifest['metadata'].items():
        safe_name(name)
        data = blobs.get(name)
        if data is None:
            matches = [(k, v) for k, v in blobs.items()
                       if len(v) == expected['bytes'] and sha(v) == expected['sha256']]
            if matches:
                source, data = matches[0]
                candidates[name] = {'source_file': source, 'sha256': sha(data)}
            else:
                failures.append('Missing: ' + name)
                continue
        try:
            verify_blob(name, data, expected)
        except Exception as exc:
            failures.append(str(exc))
    return failures, candidates


class Cloud:
    def __init__(self, credentials):
        self.credentials = credentials
        self.local = threading.local()

    def session(self):
        from google.auth.transport.requests import AuthorizedSession
        if not hasattr(self.local, 'session'):
            self.local.session = AuthorizedSession(self.credentials)
        return self.local.session

    def get(self, path, **params):
        for attempt in range(7):
            response = self.session().get(API + path, params=params, timeout=(15, 90))
            if not rate_limited(response) and response.status_code not in (500, 502, 503, 504):
                response.raise_for_status()
                return response
            if attempt >= 2:
                print('Drive rate limit/transient error; retrying safely.', flush=True)
            time.sleep(min(2 ** (attempt + 1), 60))
        response.raise_for_status()

    def blob(self, file_id):
        return self.get('/files/' + file_id, alt='media').content

    def children(self, folder):
        result, token = [], None
        while True:
            args = dict(q="'" + folder + "' in parents and trashed = false",
                        fields='nextPageToken,files(id,name,size,mimeType,modifiedTime,md5Checksum)',
                        pageSize=1000)
            if token:
                args['pageToken'] = token
            page = self.get('/files', **args).json()
            result.extend(page.get('files', []))
            token = page.get('nextPageToken')
            if not token:
                return result

    def revisions(self, file_id):
        result, token = [], None
        while True:
            args = dict(pageSize=1000, fields='nextPageToken,revisions(id,size,modifiedTime)')
            if token:
                args['pageToken'] = token
            page = self.get('/files/' + file_id + '/revisions', **args).json()
            result.extend(page.get('revisions', []))
            token = page.get('nextPageToken')
            if not token:
                return result

    def create_backup(self, path):
        """Create only a new file; validate Google's stored-file checksum."""
        metadata = {'name': path.name, 'parents': [CAMPAIGN_ROOT],
                    'description': 'Read-only recovery audit; original results unchanged.'}
        mime = 'application/json' if path.suffix == '.json' else 'application/zip'
        response = self.session().post('https://www.googleapis.com/upload/drive/v3/files',
            params={'uploadType': 'resumable', 'fields': 'id,name,md5Checksum,webViewLink'},
            json=metadata, headers={'X-Upload-Content-Type': mime,
                                   'X-Upload-Content-Length': str(path.stat().st_size)},
            timeout=(15, 60))
        response.raise_for_status()
        location = response.headers['Location']
        with path.open('rb') as stream:
            response = self.session().put(location, data=stream,
                headers={'Content-Type': mime, 'Content-Length': str(path.stat().st_size)},
                timeout=(15, 300))
        response.raise_for_status()
        uploaded = response.json()
        check = self.get('/files/' + uploaded['id'],
                         fields='id,name,size,md5Checksum,webViewLink').json()
        h = hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        if check.get('md5Checksum') != h.hexdigest() or int(check['size']) != path.stat().st_size:
            raise RuntimeError('Backup upload checksum did not verify; original results untouched')
        return check


def audit_run(cloud, item, folder, target):
    record = {'id': item['id'], 'status': 'needs_recovery', 'issues': [],
              'exact_repair_candidates': {}}
    if folder is None:
        record['issues'] = ['Missing run folder']
        return record
    files = cloud.children(folder['id'])
    names = [f['name'] for f in files]
    if len(names) != len(set(names)):
        record['issues'] = ['Duplicate filenames; no automatic selection']
        return record
    by_name = {f['name']: f for f in files}
    blobs = {}
    destination = target / 'runs' / item['id']
    destination.mkdir(parents=True)
    for name, file in by_name.items():
        safe_name(name)
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            record['issues'].append('Unexpected subfolder: ' + name)
            continue
        if int(file.get('size', 0)) > 32 * 1024**2:
            record['issues'].append('Large file deferred: ' + name)
            continue
        try:
            data = cloud.blob(file['id'])
            p = destination / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            blobs[name] = data
        except Exception as exc:
            record['issues'].append('Read failed: ' + name + ': ' + str(exc))
    if 'archive_manifest.json' not in blobs:
        record['issues'].append('Missing archive manifest')
        return record
    manifest = json.loads(blobs['archive_manifest.json'])
    if manifest['task']['id'] != item['id']:
        record['issues'].append('Manifest task identity differs')
        return record
    for key in ('model', 'arm', 'fold', 'seed'):
        if manifest['task'][key] != item[key]:
            record['issues'].append('Task identity differs: ' + key)
    failures, candidates = assess_files(manifest, blobs)
    record['issues'].extend(failures)
    record['exact_repair_candidates'] = candidates
    for name, provenance in candidates.items():
        p = target / 'exact_repair_candidates' / item['id'] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blobs[provenance['source_file']])
    for marker in ('archive_acknowledgment.json', 'cleanup.json'):
        if marker not in blobs:
            record['issues'].append('Missing: ' + marker)
    if 'archive_acknowledgment.json' in blobs:
        ack = json.loads(blobs['archive_acknowledgment.json'])
        if ack.get('status') != 'verified' or ack.get('manifest_sha256') != sha(blobs['archive_manifest.json']):
            record['issues'].append('Archive acknowledgment mismatch')
    if 'cleanup.json' in blobs and json.loads(blobs['cleanup.json']).get('status') != 'complete':
        record['issues'].append('Cleanup receipt incomplete')
    if not record['issues'] and not candidates:
        record['status'] = 'cloud_archive_hashes_verified'
    elif not record['issues']:
        record['status'] = 'exact_repair_available_not_applied'
    return record


def audit_recovery(cloud, folder, target):
    record = {'id': folder['name'], 'files': cloud.children(folder['id']),
              'status': 'requires_checkpoint_and_budget_review', 'small_files': {}, 'revisions': {}}
    dest = target / 'recovery_metadata' / folder['name']
    dest.mkdir(parents=True)
    for file in record['files']:
        name = file['name']
        safe_name(name)
        if not name.endswith('.json'):
            continue
        data = cloud.blob(file['id'])
        (dest / name).write_bytes(data)
        try:
            record['small_files'][name] = json.loads(data)
        except (ValueError, UnicodeError):
            record['small_files'][name] = {'invalid_or_empty': True}
        versions = cloud.revisions(file['id'])
        record['revisions'][name] = versions
        for version in versions[-10:]:
            if int(version.get('size', 0)) <= 0 or int(version.get('size', 0)) > 1024**2:
                continue
            try:
                old = cloud.get('/files/' + file['id'] + '/revisions/' + version['id'], alt='media').content
                (dest / (name + '.' + version['id'] + '.revision')).write_bytes(old)
            except Exception as exc:
                version['download_error'] = str(exc)
    return record


def run(credentials, work='/content/msd_recovery_audit'):
    cloud = Cloud(credentials)
    email = cloud.get('/about', fields='user(emailAddress)').json()['user']['emailAddress']
    if email.lower() != ACCOUNT:
        raise RuntimeError('Wrong account: ' + email + '. Sign in as ' + ACCOUNT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path(work) / stamp
    target.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(cloud.blob(SNAPSHOT_ID))
    (target / 'original_reported_results.json').write_text(json.dumps(snapshot, indent=2))
    items = snapshot['runs']
    if len({r['id'] for r in items}) != len(items) or any(r['seed'] not in (0, 2) for r in items):
        raise ValueError('Unexpected task identities in snapshot')
    for item in items:
        if '/' in item['id'] or safe_name(item['id']).name != item['id']:
            raise ValueError('Invalid task identifier')
    folders = cloud.children(RUNS_ROOT)
    mapping = {}
    for f in folders:
        if f['name'] in mapping:
            raise ValueError('Duplicate run folders: ' + f['name'])
        mapping[f['name']] = f
    report = {'created_at_utc': stamp, 'account': email, 'training_launched': False,
              'source_files_modified': False, 'restart_approved': False, 'runs': [], 'recovery': []}
    print('Reading actual cloud bytes. No Drive mount, GPU, training or deletions.', flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(audit_run, cloud, r, mapping.get(r['id']), target): r for r in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {'id': item['id'], 'status': 'audit_failed', 'issues': [str(exc)]}
            report['runs'].append(record)
            print(f"CHECKED {len(report['runs'])}/{len(items)}: {item['id']} — {record['status']}", flush=True)
    completed = {r['id'] for r in items}
    for folder in cloud.children(RECOVERY_ROOT):
        name = folder['name']
        if name in completed or not name.endswith(('s0', 's2')):
            continue
        safe_name(name)
        print('RECOVERY METADATA:', name, flush=True)
        try:
            report['recovery'].append(audit_recovery(cloud, folder, target))
        except Exception as exc:
            report['recovery'].append({'id': name, 'status': 'audit_failed', 'error': str(exc)})
    report['counts'] = {status: sum(r['status'] == status for r in report['runs'])
                        for status in sorted({r['status'] for r in report['runs']})}
    report['note'] = 'Archive byte integrity audit, not GPU preflight or permission to reset/replay charged updates. Large recovery checkpoints are preserved in Drive and need separate budget/state inspection.'
    (target / 'recovery_report.json').write_text(json.dumps(report, indent=2))
    archive = target.parent / ('MSD_RECOVERY_AUDIT_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(target))
    report_path = target.parent / ('MSD_RECOVERY_REPORT_' + stamp + '.json')
    report_path.write_text(json.dumps(report, indent=2))
    print('AUDIT FINISHED:', json.dumps(report['counts']), flush=True)
    print('Uploading a new audit backup; existing campaign files remain unchanged.', flush=True)
    try:
        receipt = cloud.create_backup(archive)
        print('BACKUP VERIFIED:', receipt.get('webViewLink', receipt['id']), flush=True)
        report_receipt = cloud.create_backup(report_path)
        print('REPORT VERIFIED:', report_receipt.get('webViewLink', report_receipt['id']), flush=True)
    except Exception as exc:
        receipt = {'upload_status': 'not_fully_verified', 'error': str(exc)}
        print('DRIVE BACKUP NOT FULLY VERIFIED:', str(exc), flush=True)
        print('Downloading the report and backup to your computer instead. Keep both files.', flush=True)
        from google.colab import files
        files.download(str(report_path))
        files.download(str(archive))
    print('Training remains stopped. Share this notebook’s final output with Codex for the recovery decision.', flush=True)
    return report, receipt
