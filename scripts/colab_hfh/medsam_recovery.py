"""Verified HFH uploads through rclone, avoiding Drive FUSE write failures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from scripts.colab_medsam3 import runtime


MOUNT = Path('/content/hfh_storage')
AUTH = Path('/content/.hfh_storage_auth')
LOCAL_COPY = runtime.copy_verified
REMOTE_FOLDERS = {}


def remote_path(path):
    try:
        relative = Path(path).relative_to(MOUNT)
    except ValueError:
        return None
    return 'hfh_storage:' + relative.as_posix()


def rclone(*args):
    if os.environ.get('HFH_DRIVE_TRANSPORT') == 'colab_adc':
        from scripts.colab_hfh.medsam_constraints_drive import client
        return client().command(*args)
    command = [str(AUTH/'rclone'), '--config', str(AUTH/'rclone.conf'),
               '--tpslimit', '2', '--tpslimit-burst', '2', '--retries', '5',
               '--low-level-retries', '10', '--retries-sleep', '5s', *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise IOError('Verified HFH upload failed: ' + result.stderr[-1500:])
    return result.stdout


def copy_verified(source, destination):
    remote = remote_path(destination)
    if remote is None:
        return LOCAL_COPY(source, destination)
    if os.environ.get('HFH_DRIVE_TRANSPORT') == 'colab_adc':
        from scripts.colab_hfh.medsam_constraints_drive import client
        return client().upload(source, remote)
    source = Path(source)
    parent, name = remote.rsplit('/', 1)
    if parent not in REMOTE_FOLDERS:
        rclone('mkdir', parent)
        folder = json.loads(rclone('lsjson', parent, '--stat'))
        # Some rclone versions omit ID for a directory addressed as the
        # filesystem root. Fall back to its exact path, retaining all checks.
        folder_id = folder.get('ID') if folder.get('IsDir') else None
        if not folder_id:
            container, folder_name = parent.rsplit('/', 1)
            children = json.loads(rclone('lsjson', container, '--dirs-only'))
            matches = [row.get('ID') for row in children
                       if isinstance(row, dict) and row.get('Name') == folder_name and row.get('ID')]
            if len(matches) == 1:
                folder_id = matches[0]
        REMOTE_FOLDERS[parent] = folder_id
    # Reuse the exact directory ID: each new rclone process otherwise walks
    # the same multi-level Drive path again for every upload and hash check.
    root = []
    if REMOTE_FOLDERS[parent]:
        root = ['--drive-root-folder-id', REMOTE_FOLDERS[parent]]
        remote = 'hfh_storage:' + name
    checksum = hashlib.md5()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            checksum.update(block)
    rclone('copyto', str(source), remote, '--checksum', *root)
    metadata = json.loads(rclone('lsjson', remote, '--stat', '--hash', *root))
    hashes = {key.lower(): value for key, value in metadata.get('Hashes', {}).items()}
    if metadata['Size'] != source.stat().st_size or hashes.get('md5') != checksum.hexdigest():
        raise IOError('Drive upload size/checksum mismatch: ' + str(destination))


def restore(output, recovery):
    """Stage the exact remote checkpoint before the audited restore routine."""
    if os.environ.get('HFH_DRIVE_TRANSPORT') == 'colab_adc':
        from scripts.colab_hfh.medsam_constraints_drive import client
        api = client()
        remote = remote_path(recovery)
        try:
            api.resolve(remote)
        except FileNotFoundError:
            return
        # An existing recovery folder with a missing pointer is not silently
        # treated as a new run: its uploaded snapshots need an explicit audit.
        api.download(remote+'/latest.json', recovery/'latest.json')
        marker = json.loads((recovery/'latest.json').read_text())
        name = marker['file']
        if Path(name).name != name or not name.startswith('recovery-') or not name.endswith('.tar'):
            raise ValueError('Unsafe recovery pointer')
        api.download(remote+'/'+name, recovery/name)
    return runtime.restore(output, recovery)


class RecoveryWriter(runtime.RecoveryWriter):
    def persist(self, snapshot):
        archive = snapshot.with_suffix('.tar')
        successful = False
        try:
            runtime.write(snapshot/'snapshot_manifest.json',
                          {p.name: runtime.digest(p) for p in snapshot.iterdir()})
            with tarfile.open(archive, 'w') as tar:
                for path in snapshot.iterdir():
                    tar.add(path, arcname=path.name, recursive=False)
            sha = runtime.digest(archive)
            name = 'recovery-'+sha+'.tar'
            copy_verified(archive, self.durable/name)
            # Publish the pointer only after the archive is verified on Drive.
            marker = snapshot/'latest.json'
            runtime.write(marker, dict(file=name, sha256=sha))
            copy_verified(marker, self.durable/'latest.json')
            successful = True
        finally:
            # Failed snapshots are retained locally for recovery/inspection.
            if successful:
                archive.unlink(missing_ok=True)
                shutil.rmtree(snapshot)


def install(optimized):
    optimized.RecoveryWriter = RecoveryWriter
    if os.environ.get('HFH_CONTROLLER_ARCHIVES') == '1':
        # The compact controller validates all local metadata, uploads one
        # complete archive, and verifies it before retiring any checkpoint.
        # RecoveryWriter still uploads periodic and final snapshots itself.
        optimized.copy_verified = lambda source, destination: None
    else:
        optimized.copy_verified = copy_verified
