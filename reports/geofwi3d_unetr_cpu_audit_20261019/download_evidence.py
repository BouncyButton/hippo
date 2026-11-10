"""Materialize connector-authorized small result files; never fetch checkpoints."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
plan = json.loads(Path('/private/tmp/geofwi_audit_downloads.json').read_text())

def fetch(item):
    dst = ROOT/'evidence'/item['run']/item['name']
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rtk', 'proxy', 'curl', '--fail', '--silent', '--show-error',
                    '--retry', '2', item['download_url'], '--output', str(dst)], check=True)
    if item.get('size') is not None:
        assert dst.stat().st_size == int(item['size']), str(dst)
    return {k:v for k,v in item.items() if k!='download_url'} | {
        'local_path':str(dst.relative_to(ROOT)), 'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    receipts = list(pool.map(fetch, plan))
(ROOT/'download_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(f'Downloaded and size-checked {len(receipts)} small evidence files. No checkpoints downloaded.')
