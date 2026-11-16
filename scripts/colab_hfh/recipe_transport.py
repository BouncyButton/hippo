"""Restore verified diagnostic archives and record the current charged attempt."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def command():
    private = Path('/content/.hfh_storage_auth')
    return [str(private/'rclone'), '--config', str(private/'rclone.conf'),
            '--tpslimit', '2', '--tpslimit-burst', '2', '--retries', '3']


def restore(root, suffix):
    remote = 'hfh_storage:results_recipe_diagnostics_v3'+suffix
    cache = root/('imported_recipe_archives'+suffix)
    cache.mkdir(exist_ok=True)
    result = subprocess.run(command()+['copy', remote, str(cache), '--include', '*.zip',
        '--include', 'campaign_status.json', '--transfers', '2', '--checkers', '2'],
        capture_output=True, text=True, timeout=600)
    if result.returncode:
        if result.returncode == 3 and 'directory not found' in result.stderr.lower():
            return
        raise RuntimeError('Cannot inspect existing diagnostic results: '+result.stderr[-1000:])
    for archive in sorted(cache.glob('*/*.zip')):
        label, name = archive.parent.name, archive.stem
        work = root/('recipe_diagnostics'+suffix)/label
        with zipfile.ZipFile(archive) as zipped:
            manifest = json.loads(zipped.read('results/archive_manifest.json'))
            for filename, expected in manifest.items():
                if Path(filename).name != filename or hashlib.sha256(zipped.read('results/'+filename)).hexdigest() != expected:
                    raise ValueError('Remote archive failed SHA256 verification: '+str(archive))
            for item in zipped.infolist():
                parts = Path(item.filename).parts
                if item.is_dir() or len(parts) != 2 or parts[0] not in ('results','protocol'):
                    continue
                if parts[1] in ('.','..') or Path(item.filename).is_absolute():
                    raise ValueError('Unsafe diagnostic archive member')
                target = work/('archives/'+name if parts[0]=='results' else 'protocol')/parts[1]
                payload = zipped.read(item)
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError('Existing local result/protocol differs from Drive: '+str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        shutil.copyfile(archive, work/archive.name)
        (work/(name+'.remote_verified.json')).write_text(json.dumps(dict(
            remote=remote+'/'+label+'/'+archive.name, md5=hashlib.md5(archive.read_bytes()).hexdigest(),
            verification='rclone download checksum and per-member archive SHA256')))
    status_file = cache/'campaign_status.json'
    if status_file.exists():
        status = json.loads(status_file.read_text())
        current = status.get('current')
        if status['status'] in ('running','runtime_error') and current:
            c = current
            label = f"{c['model']}_lr{c['lr']:g}_poly_cap"
            work = root/('recipe_diagnostics'+suffix)/label
            name = f"{c['model']}_f{c['fold']}_s{c['seed']}"
            if not (work/'archives'/name/'result.json').exists() and not (work/name/'budget_ledger.json').exists():
                raise RuntimeError('Previous VM has an unfinished charged attempt. Its recovery must be audited before replay: '+name)


def publish_status(path, suffix):
    subprocess.run(command()+['copyto', str(path),
        'hfh_storage:results_recipe_diagnostics_v3'+suffix+'/campaign_status.json'],
        check=True, timeout=180)
