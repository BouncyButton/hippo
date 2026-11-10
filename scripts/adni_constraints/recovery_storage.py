"""Retire only the 22 checkpoint copies whose local backups were verified."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recovery',type=Path,required=True)
    args=parser.parse_args()
    stage=args.recovery
    for name,sha in json.loads((stage/'manifest.json').read_text()).items():
        if digest(stage/name)!=sha:
            raise ValueError('Frozen recovery changed')
    control=json.loads((stage/'recovery_plan.json').read_text())
    root=Path(control['result_root'])
    archive=json.loads((stage/'checkpoint_archive.json').read_text())
    receipt_path=stage/'cleanup_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['status']!='complete' or receipt['archive_catalog_sha256']!=digest(stage/'checkpoint_archive.json'):
            raise ValueError('Existing cleanup receipt differs')
        print(json.dumps(receipt),flush=True)
        return
    bound_catalog=root/'checkpoint_archive.json'
    previously_verified=bound_catalog.exists() and digest(bound_catalog)==digest(stage/'checkpoint_archive.json')
    summary=json.loads((root/'summary.json').read_text())
    expected={str(Path(r['model'])/r['objective']/'fold0/seed0/run/checkpoint_best.pt') for r in summary['runs']}
    if len(expected)!=22 or {r['relative_path'] for r in archive['files']}!=expected:
        raise ValueError('Cleanup must exactly match the 22 verified completed runs')
    # Verify every candidate before any file is removed.
    for item in archive['files']:
        p=root/item['relative_path']
        if not p.exists() and previously_verified:
            continue
        if (p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes']
                or digest(p)!=item['sha256']
                or digest(p.parent/'completion.json')!=item['completion_sha256']):
            raise ValueError('Checkpoint differs from verified local backup: '+str(p))
    interrupted=root/control['interrupted_relative_path']
    destination=root/control['interrupted_archive_relative_path']
    if not interrupted.exists() and destination.exists() and previously_verified:
        interrupted=destination
    if digest(interrupted/'checkpoint_latest.pt')!=control['checkpoint_sha256']:
        raise ValueError('Interrupted checkpoint changed')
    for name,sha in control['interrupted_metadata_sha256'].items():
        if digest(interrupted/name)!=sha:
            raise ValueError('Interrupted metadata changed')
    if destination.exists() and destination!=interrupted:
        raise FileExistsError('Interrupted archive already exists')
    (root/'checkpoint_archive.json').write_bytes((stage/'checkpoint_archive.json').read_bytes())
    for item in archive['files']:
        (root/item['relative_path']).unlink(missing_ok=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if interrupted!=destination:
        interrupted.rename(destination)
    receipt=dict(status='complete',completed_models_preserved_locally=22,
        released_file_bytes=sum(i['bytes'] for i in archive['files']),timestamp_unix=time.time(),
        archive_catalog_sha256=digest(stage/'checkpoint_archive.json'),
        interrupted_checkpoint_archive=str(destination),original_ledger_preserved=True)
    (stage/'cleanup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
