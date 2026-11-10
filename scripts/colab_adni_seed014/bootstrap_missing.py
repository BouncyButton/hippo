"""Import only immutable gate receipts and four interrupted recoveries, never reset state."""
import argparse
import json
from pathlib import Path

from missing_scope import digest, read, verify_package


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing state differs; refusing overwrite: ' + str(path))
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
    if read(path) != value:
        raise IOError('Imported metadata did not read back: ' + str(path))


def bootstrap(package, drive_root):
    verify_package(package)
    scope = read(package/'continuation.json')
    identity = {'continuation_sha256': digest(package/'continuation.json'),
                'target_account': scope['target_account'], 'historical_archives_imported': 0}
    marker = drive_root/'missing_only_bootstrap.json'
    for folder in ('results_noaug_v1', 'results_seeds014_noaug_v1'):
        for lock in ('.adni_controller_lock', '.adni_seeds014_controller_lock'):
            if (drive_root/folder/lock).exists():
                raise RuntimeError('Controller ownership exists; preserve it and inspect before setup')
    # Static scientific receipts must be exactly the original ones, even on rerun.
    for name, expected in scope['static_receipts'].items():
        source = package/'import_state'/name
        if digest(source) != expected:
            raise ValueError('Imported scientific receipt changed')
        target = drive_root/name
        if marker.exists() and not target.is_file():
            raise RuntimeError('Previously imported receipt is missing; do not recreate state')
        write_once(target, read(source))
    if marker.exists():
        if read(marker) != identity:
            raise ValueError('Different continuation already installed')
        print('BOOTSTRAP: already installed; no checkpoint or ledger overwritten.', flush=True)
        return
    for task, record in scope['recovery'].items():
        folder = drive_root/'results_noaug_v1/recovery'/task
        pointer, ledger = record['pointer'], record['ledger']
        # Large files were copied server-side to this account. Reading their
        # complete bytes occurs in the original restore gate before any update.
        path = folder/pointer['file']
        if not path.is_file() or path.stat().st_size != record['size']:
            raise RuntimeError('Copied recovery is missing/incomplete: ' + str(path))
        for name, value in [('charged_ledger.json', ledger), ('latest.json', pointer)]:
            write_once(folder/name, value)
        print('IMPORTED RECOVERY:', task, 'charged updates:', ledger['charged_attempts'], flush=True)
    write_once(marker, identity)
    print('BOOTSTRAP COMPLETE: four recoveries; zero old result archives. '
          'Full recovery hashes/readability are still required before training.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive-root', type=Path, required=True)
    args = parser.parse_args()
    bootstrap(Path(__file__).resolve().parent, args.drive_root)
