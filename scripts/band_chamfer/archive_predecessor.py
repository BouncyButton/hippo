"""Record the completed predecessor before explicitly authorized cleanup.

Archive is read-only with respect to the original run. Cleanup is a separate
invocation requiring a verified local-backup receipt and copied warmup hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

EXPECTED = Path('/mnt/beegfsstudents/home/3160552/edge_ablation_fold0_seed0_5pct_20261008_01')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def archive(source, destination):
    if source != EXPECTED or source.is_symlink():
        raise ValueError('Only the named predecessor is in cleanup scope')
    status = json.loads((source/'status.json').read_text())
    results = json.loads((source/'results.json').read_text())
    protocol = json.loads((source/'protocol.json').read_text())
    if status['state'] != 'complete' or status['completed_runs'] != 20 or results['completed'] != 20:
        raise ValueError('The experiment has not completed all accepted runs')
    # Integrity is verified before deleting any artifact, including checkpoints.
    for entry in protocol['runs']:
        folder = source/'runs'/entry['id']
        record = json.loads((folder/'completion.json').read_text())
        if record['identity']['protocol_sha256'] != protocol['identity_sha']:
            raise ValueError('Run protocol mismatch')
        for name, expected in record['artifacts'].items():
            if sha(folder/name) != expected:
                raise ValueError('Completed artifact mismatch: '+str(folder/name))
    destination.mkdir(parents=True, exist_ok=False)
    kept, discarded = {}, {}
    extensions = {'.json', '.csv', '.md', '.log', '.out', '.err', '.py', '.sh', '.txt'}
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError('Unexpected symlink inside predecessor')
        relative = str(path.relative_to(source))
        if path.suffix in extensions:
            output = destination/relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            kept[relative] = dict(bytes=path.stat().st_size, sha256=sha(output))
        else:
            discarded[relative] = dict(bytes=path.stat().st_size)
    record = dict(source=str(source), recorded_at_unix=time.time(), completed_runs=20,
                  retained_files=kept, removed_artifacts=discarded,
                  source_bytes=sum(v['bytes'] for v in kept.values())+sum(v['bytes'] for v in discarded.values()),
                  note='All aggregate/per-case metrics, histories, calibration/configuration, provenance, source snapshots and logs retained. Checkpoints and compressed prediction/geometry arrays omitted at user request.')
    (destination/'archive_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    bundle = destination.with_suffix('.tar.gz')
    with tarfile.open(bundle, 'w:gz') as tar:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(destination)))
    receipt = dict(archive=str(bundle), archive_sha256=sha(bundle), archive_bytes=bundle.stat().st_size,
                   source=str(source), source_bytes=record['source_bytes'], completed_runs=20)
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


def cleanup(source, destination, followup):
    receipt = json.loads(destination.with_suffix('.receipt.json').read_text())
    local = json.loads(destination.with_suffix('.local_verified.json').read_text())
    if source != EXPECTED or source.is_symlink() or receipt['source'] != str(source):
        raise ValueError('Cleanup path mismatch')
    if local['archive_sha256'] != receipt['archive_sha256'] or not local['verified']:
        raise ValueError('No verified local backup')
    if sha(receipt['archive']) != receipt['archive_sha256']:
        raise ValueError('Archive changed')
    prefix = json.loads((followup/'shared_warmup/provenance.json').read_text())
    for name, expected in prefix['artifacts'].items():
        if sha(followup/'shared_warmup'/name) != expected:
            raise ValueError('Follow-up warmup not safely copied')
    queue = subprocess.check_output(['squeue', '-u', '3160552', '-h', '-o', '%i|%Z|%o'], text=True)
    if str(source) in queue or '690632|' in queue or '690510|' in queue:
        raise ValueError('A scheduler job still references the predecessor')
    if json.loads((source/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Predecessor is no longer complete')
    shutil.rmtree(source)
    value = dict(removed=str(source), removed_bytes=receipt['source_bytes'],
                 archive_sha256=receipt['archive_sha256'], local_backup=local,
                 warmup_preserved_at=str(followup/'shared_warmup'), finished_unix=time.time())
    destination.with_suffix('.cleanup.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('archive', 'cleanup'))
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--followup', type=Path)
    args = parser.parse_args()
    if args.mode == 'archive':
        archive(EXPECTED, args.destination)
    else:
        cleanup(EXPECTED, args.destination, args.followup)


if __name__ == '__main__':
    main()
