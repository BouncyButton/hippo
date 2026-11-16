"""Commit MedSAM results as one verified archive, with local scientific checks."""
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile
import zipfile

from scripts.colab_hfh import runtime as original
from scripts.colab_hfh.medsam_recovery import copy_verified, rclone, remote_path

CONTEXT = {}


def configure(bundle, work, durable):
    cache = Path(work)/'compact_archives'
    cache.mkdir(parents=True, exist_ok=True)
    remote = Path(durable)/'compact_archives'
    rclone('mkdir', remote_path(remote))
    inventory = json.loads(rclone('lsjson', remote_path(remote), '--files-only', '--hash'))
    CONTEXT.update(bundle=Path(bundle), work=Path(work), cache=cache, remote=remote,
                   inventory={row['Name']: row for row in inventory})


def has_archive(durable):
    return Path(durable).name+'.zip' in CONTEXT['inventory']


def verify_zip(path):
    with zipfile.ZipFile(path) as zipped:
        for item in zipped.infolist():
            parts = Path(item.filename).parts
            if item.is_dir() or len(parts) != 2 or parts[0] not in ('results', 'source') or '..' in parts:
                raise ValueError('Unsafe compact archive member')
        manifest = json.loads(zipped.read('results/archive_manifest.json'))
        for name, expected in manifest.items():
            if Path(name).name != name or hashlib.sha256(zipped.read('results/'+name)).hexdigest() != expected:
                raise ValueError('Compact result archive checksum mismatch')
        return json.loads(zipped.read('results/result.json'))


def verify_archive(durable):
    name = Path(durable).name+'.zip'
    local = CONTEXT['cache']/name
    row = CONTEXT['inventory'][name]
    expected = {k.lower(): v for k, v in row.get('Hashes', {}).items()}.get('md5')
    if not expected:
        raise ValueError('Remote compact archive has no checksum')
    if not local.exists() or hashlib.md5(local.read_bytes()).hexdigest() != expected:
        rclone('copyto', remote_path(CONTEXT['remote']/name), str(local), '--checksum')
    if hashlib.md5(local.read_bytes()).hexdigest() != expected:
        raise ValueError('Downloaded compact archive differs from Drive')
    result = verify_zip(local)
    expected_name = f"medsam3_f{result['fold']}_s{result['seed']}.zip"
    if name != expected_name or result.get('updates') != 200:
        raise ValueError('Archived run identity differs')
    protocol = original.read(CONTEXT['bundle']/'medsam_protocols'/f"f{result['fold']}_s{result['seed']}.json")
    with zipfile.ZipFile(local) as zipped:
        if json.loads(zipped.read('results/protocol.json')) != {**protocol, 'arm': 'dice'}:
            raise ValueError('Archived scientific protocol differs')
    return result


def archive(output, durable, kind, model, fold, seed, protocol):
    if kind != 'medsam3':
        raise ValueError('Compact MedSAM controller cannot run baselines')
    output, durable = Path(output), Path(durable)
    done = original.read(output/'completion.json')
    if done.get('fold') != fold or done.get('seed') != seed:
        raise ValueError('Completion identity differs from requested run')
    name = durable.name+'.zip'
    archive_path = CONTEXT['cache']/name
    with tempfile.TemporaryDirectory(prefix='medsam-archive-', dir=CONTEXT['work']) as tmp:
        stage, final = Path(tmp)/'input', Path(tmp)/'results'
        stage.mkdir(); final.mkdir()
        for path in output.iterdir():
            if path.is_file() and path.suffix in ('.json', '.csv', '.log', '.jsonl', '.png'):
                shutil.copyfile(path, stage/path.name)
        # Reuse the existing 200-update, split, evaluation-hash and case-count
        # checks on SSD. No original checkpoint is deleted by this staged call.
        result = original.archive(stage, final, kind, model, fold, seed, protocol)
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(final.iterdir()):
                if path.is_file():
                    zipped.write(path, 'results/'+path.name)
            for filename in ('manifest.json', 'runtime_manifest.json', 'gpu_overlay.json', 'medsam_storage_patch_manifest.json'):
                path = CONTEXT['bundle']/filename
                if path.exists():
                    zipped.write(path, 'source/'+filename)
            for filename in ('medsam.py', 'medsam_recovery.py', 'medsam_archive.py', 'medsam_search.py', 'search.py', 'runtime.py'):
                path = Path(__file__).parent/filename
                if path.exists():
                    zipped.write(path, 'source/'+filename)
        if verify_zip(archive_path) != result:
            raise ValueError('Local result archive differs')
        copy_verified(archive_path, CONTEXT['remote']/name)
        md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
        CONTEXT['inventory'][name] = dict(Name=name, Hashes={'md5': md5})
        # Retire original local checkpoints only after the complete result
        # archive has been verified on Drive. Remote recovery remains intact
        # until this commit; it is retired separately with an audit receipt.
        deleted = []
        for checkpoint in output.glob('checkpoint_*.pt'):
            if original.digest(checkpoint) != done['checkpoint_sha256']:
                raise ValueError('Local checkpoint changed; preserved')
            deleted.append(dict(path=str(checkpoint), bytes=checkpoint.stat().st_size))
            checkpoint.unlink()
        receipt = dict(remote=remote_path(CONTEXT['remote']/name), md5=md5,
                       local_checkpoints_retired=deleted, status='verified')
        receipt_path = output/'compact_archive_receipt.json'
        original.write(receipt_path, receipt)
        try:
            recovery = remote_path(durable/'recovery')
            entries = json.loads(rclone('lsjson', recovery, '--files-only', '--hash'))
            known = [row for row in entries if row['Name'] == 'latest.json' or
                     re.fullmatch(r'recovery-[0-9a-f]{64}\.tar', row['Name'])]
            # Only generated recovery files for this verified completed run.
            # Unknown files and failed partial uploads are deliberately retained.
            if known:
                filters = [arg for row in known for arg in ('--include', '/'+row['Name'])]
                rclone('delete', recovery, '--max-depth', '1', *filters)
            receipt['remote_recovery_retired'] = known
        except (OSError, ValueError) as error:
            receipt['recovery_cleanup_pending'] = str(error)
            print('Recovery cleanup pending; verified results are safe:', str(error), flush=True)
        original.write(receipt_path, receipt)
        copy_verified(receipt_path, CONTEXT['remote']/(durable.name+'.retirement.json'))
        print('COMPACT ARCHIVE VERIFIED', name, 'bytes', archive_path.stat().st_size, flush=True)
        return result
