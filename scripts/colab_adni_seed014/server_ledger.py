"""One server-acknowledged update per charge; no per-update Drive-FUSE rename.

The first publication retains the existing mounted path so the frozen runner's
existence checks remain valid. Subsequent writes update that SAME server ID.
Only this writer's acknowledged bytes may satisfy its immediate readback; a
different process always reads the server. Training/snapshot payloads are opaque.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time


FIELDS = 'id,name,size,md5Checksum,parents'


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def matches(data, metadata):
    return (int(metadata.get('size', -1)) == len(data)
            and metadata.get('md5Checksum') == hashlib.md5(data).hexdigest())


class ServerLedger:
    def __init__(self, transport, root_id, api=None):
        self.transport = transport
        self.root = transport.drive_root
        self.root_id = root_id
        self._api = api
        self._local = threading.local()
        self.folders = {(): root_id}
        self.ids = {}
        self.acknowledged = {}

    def api(self):
        if self._api is not None:
            return self._api
        if not hasattr(self._local, 'api'):
            import google.auth
            import google_auth_httplib2
            import httplib2
            from googleapiclient.discovery import build
            http = google_auth_httplib2.AuthorizedHttp(google.auth.default()[0],
                                                     http=httplib2.Http(timeout=120))
            self._local.api = build('drive', 'v3', http=http, cache_discovery=False)
        return self._local.api

    def execute(self, request, label, mutation=False):
        return self.transport.call(label, lambda: request.execute(num_retries=0), mutation=mutation)

    def child(self, parent, name):
        escaped = name.replace('\\', '\\\\').replace("'", "\\'")
        result = self.execute(self.api().files().list(
            q=f"'{parent}' in parents and trashed=false and name='{escaped}'",
            fields='files('+FIELDS+')', pageSize=100), 'resolve ledger path')
        found = result.get('files', [])
        if len(found) > 1:
            raise RuntimeError('Duplicate Drive ledger path; preserve and inspect: '+name)
        return found[0] if found else None

    def resolve(self, path):
        relative = Path(os.path.abspath(path)).relative_to(self.root)
        parent_parts = ()
        parent = self.root_id
        for part in relative.parts[:-1]:
            parent_parts += (part,)
            if parent_parts not in self.folders:
                child = self.child(parent, part)
                if child is None:
                    return None
                self.folders[parent_parts] = child['id']
            parent = self.folders[parent_parts]
        return self.child(parent, relative.name)

    def read(self, path):
        if Path(path).name != 'charged_ledger.json':
            raise ValueError('Expected charge file')
        key = str(path)
        if key in self.acknowledged:
            return json.loads(self.acknowledged[key])
        metadata = self.resolve(path)
        if metadata is None:
            raise RuntimeError('Charge file is not visible on the Drive server; no training allowed')
        data = self.execute(self.api().files().get_media(fileId=metadata['id']), 'read server charge')
        if not matches(data, metadata):
            raise RuntimeError('Server charge bytes/checksum differ; no training allowed')
        self.ids[key] = metadata['id']
        # Do NOT cache another process's read: only our own completed upload may
        # be used as immediate readback under the exclusive experiment owner.
        return json.loads(data)

    def write(self, path, value):
        from googleapiclient.http import MediaIoBaseUpload
        if Path(path).name != 'charged_ledger.json':
            raise ValueError('Expected charge file')
        key, data = str(path), encoded(value)
        if self.acknowledged.get(key) == data:
            self.transport.add(server_duplicate_charges_avoided=1)
            return
        if key not in self.ids:
            metadata = self.resolve(path)
            if metadata is None:
                # Create the visible mounted path ONCE, then wait for independent
                # server checksum confirmation before authorizing any update.
                self.transport.call('publish initial charge path',
                    lambda: self.transport.raw_write(path, value), mutation=True)
                for attempt in range(12):
                    metadata = self.resolve(path)
                    if metadata and matches(data, metadata):
                        break
                    self.transport.wait(5)
                else:
                    raise RuntimeError('Initial charge has not reached the Drive server; local work retained')
            self.ids[key] = metadata['id']
            if matches(data, metadata):
                self.acknowledged[key] = data
                return
        file_id = self.ids[key]
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype='application/json', resumable=False)
        request = self.api().files().update(fileId=file_id, media_body=media, fields=FIELDS)
        try:
            result = self.execute(request, 'acknowledge FLOP charge on server', mutation=True)
        except Exception:
            # An ambiguous response may have committed. Reconcile the same ID;
            # never create another ledger or assume success from mounted bytes.
            result = self.execute(self.api().files().get(fileId=file_id, fields=FIELDS),
                                  'reconcile server charge')
            if not matches(data, result):
                raise RuntimeError('Drive charge failed; next optimizer update is blocked') from None
        if not matches(data, result):
            raise RuntimeError('Drive did not acknowledge the exact FLOP charge')
        self.acknowledged[key] = data
        self.transport.add(server_charge_updates=1)

    def confirm_file(self, source, target):
        """Confirm mounted archive bytes reached Google before any retirement."""
        size = Path(source).stat().st_size
        h = hashlib.md5()
        with Path(source).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        for attempt in range(12):
            metadata = self.resolve(target)
            if (metadata and int(metadata.get('size', -1)) == size
                    and metadata.get('md5Checksum') == h.hexdigest()):
                return metadata
            self.transport.wait(5)
        raise RuntimeError('Archive bytes are not confirmed on the Drive server; '
                           'retain checkpoints and local staging: '+str(target))


def saving_probe(root_id, expected_account):
    """One tiny persistent probe, reused across sessions; no GPU work."""
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    api = build('drive', 'v3', credentials=google.auth.default()[0], cache_discovery=False)
    account = api.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']
    if account.lower() != expected_account.lower():
        raise RuntimeError('Authenticate '+expected_account)
    name = 'colab_saving_probe.json'
    found = api.files().list(q=f"'{root_id}' in parents and trashed=false and name='{name}'",
                            fields='files(id)').execute().get('files', [])
    if len(found) > 1:
        raise RuntimeError('Duplicate saving probes')
    data = encoded({'checked_at': time.time(), 'nonce': os.urandom(16).hex(), 'account': account})
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype='application/json', resumable=False)
    if found:
        result = api.files().update(fileId=found[0]['id'], media_body=media, fields=FIELDS).execute()
    else:
        result = api.files().create(body={'name':name, 'parents':[root_id]},
                                    media_body=media, fields=FIELDS).execute()
    if not matches(data, result) or api.files().get_media(fileId=result['id']).execute() != data:
        raise RuntimeError('Server save/readback probe failed; training remains blocked')
    print('DRIVE SAVE CHECK PASSED: exact bytes acknowledged and independently downloaded.', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root-id', required=True)
    p.add_argument('--account', required=True)
    a = p.parse_args()
    saving_probe(a.root_id, a.account)
