"""Store verified model exports on the cluster without a laptop connection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def store_ready(root, ready):
    run_id = ready['run_id']
    if not re.fullmatch(r'\d{2}_(nnunet|swinunetr|unetr_vit_b16|basic_unet16)_dice(?:_(?:bands|edge|sdf|depth))*', run_id):
        raise ValueError('Unexpected export identity')
    if run_id != f"{ready['index']:02d}_{ready['model']}_{ready['objective']}" or not 1 <= ready['index'] <= 44:
        raise ValueError('Export identity fields differ')
    if ready['archive_name'] != run_id + '.tar':
        raise ValueError('Unexpected archive name')
    archive = root / 'outbox' / ready['archive_name']
    if archive.stat().st_size != ready['archive_bytes'] or digest(archive) != ready['archive_sha256']:
        raise ValueError('Archive checksum or size differs')
    destination = root / ready['model'] / ready['objective'] / 'fold0/seed0/run'
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or any(not member.isfile() or Path(member.name).name != member.name for member in members):
            raise ValueError('Unsafe export members')
        raw = tar.extractfile('export_manifest.json').read()
        if hashlib.sha256(raw).hexdigest() != ready['export_manifest_sha256']:
            raise ValueError('Export manifest checksum differs')
        manifest = json.loads(raw)
        if set(names) != set(manifest['files']) | {'export_manifest.json'}:
            raise ValueError('Export file membership differs')
        for member in members:
            target = destination / member.name
            temporary = target.with_suffix(target.suffix + '.tmp')
            with temporary.open('wb') as stream:
                shutil.copyfileobj(tar.extractfile(member), stream, length=1024 * 1024)
            expected = ready['export_manifest_sha256'] if member.name == 'export_manifest.json' else manifest['files'][member.name]
            if digest(temporary) != expected:
                raise ValueError('Stored member checksum differs: ' + member.name)
            temporary.replace(target)
    receipt = dict(run_id=run_id, archive_sha256=ready['archive_sha256'],
        all_files_sha256_verified=True, local_model_directory=str(destination),
        storage_kind='verified_cluster_model_only',
        model_tensor_sha256=manifest['model_tensor_sha256'], stored_at_unix=time.time())
    write_json(root / 'acks' / (run_id + '.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if not re.fullmatch(r'/mnt/beegfsstudents/home/3160552/capped_depth_delay_31976_\d+', str(root)):
        raise ValueError('Unexpected run root')
    while True:
        ready_path = root / 'ready.json'
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            ack = root / 'acks' / (ready['run_id'] + '.json')
            if not ack.exists():
                print(json.dumps({'verified_cluster_store': store_ready(root, ready)['run_id']}), flush=True)
        summary_path = root / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary['status'] == 'failed':
                raise RuntimeError('Training worker failed: ' + repr(summary['failures']))
            if summary['status'] == 'complete':
                if len(summary['runs']) != 44 or len(list((root / 'acks').glob('*.json'))) != 44:
                    raise ValueError('Completed allocation has missing verified exports')
                print(json.dumps({'cluster_storage_complete': 44}), flush=True)
                return
        time.sleep(2)


if __name__ == '__main__':
    main()
