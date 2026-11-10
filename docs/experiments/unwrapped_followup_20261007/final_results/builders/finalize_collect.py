"""Verify completed campaign artifacts and package small evidence; never delete."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

HOME = Path('/mnt/beegfsstudents/home/3160552')
ROOTS = [HOME / name for name in (
    'unwrapped_followup_fold0_20261007_01',
    'unwrapped_swin_fold0_seed0_20261006_02',
    'unwrapped_swin_fold0_seed0_20261006_01')]


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root, prior, initial = ROOTS
    protocol = json.loads((root / 'protocol.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert status['state'] == 'complete' and status['completed_main_runs'] == 24
    assert all(v['returncode'] == 0 for v in status['stages'])
    scheduler = command(['sacct', '-j', '687700,686515,686405',
                         '--format=JobID,State,Elapsed,Start,End,ExitCode', '-P'])
    assert any(s.startswith('687700|COMPLETED|') and s.endswith('|0:0') for s in scheduler.splitlines())
    assert any(s.startswith('686515|COMPLETED|') and s.endswith('|0:0') for s in scheduler.splitlines())
    expected = {e['id'] for e in protocol['runs']}
    assert expected == {p.parent.name for p in (root / 'runs').glob('*/completion.json')}
    for e in protocol['runs']:
        d = json.loads((root / 'runs' / e['id'] / 'completion.json').read_text())
        assert d['identity']['run'] == e
        assert d['identity']['protocol_sha256'] == protocol['identity_sha']
        assert d['identity']['train'] == protocol['splits'][e['fraction']]['train']
        assert d['identity']['val'] == protocol['splits'][e['fraction']]['val']
    assert len(list((prior / 'runs').glob('*/*/completion.json'))) == 20
    assert len(list((root / 'diagnostics').glob('*/*/completion.json'))) == 4
    for fraction, arms in protocol['pairs'].items():
        assert json.loads((root / ('cuda_preflight_' + fraction + '.json')).read_text())['passed']
        for arm in arms:
            d = json.loads((root / 'diagnostics' / fraction / arm / 'completion.json').read_text())
            assert d['protocol_sha256'] == protocol['identity_sha']
            assert d['checkpoint_sha256'] == sha(prior / 'runs' / fraction / arm / 'best.pt')
    assert json.loads((root / 'unit_checks.json').read_text())['passed']
    checked, hashes = [], {}

    def verify(path, expected_hash, origin):
        path = Path(path)
        assert any(path.resolve().is_relative_to(r.resolve()) for r in ROOTS), path
        actual = hashes.setdefault(str(path), None)
        if actual is None:
            actual = hashes[str(path)] = sha(path)
        assert actual == expected_hash, (str(path), origin, actual, expected_hash)
        checked.append({'path': str(path), 'sha256': actual, 'record': str(origin)})

    inventory, preserve, links = [], [], []
    for base in ROOTS:
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [n for n in dirs if n != '_finalization']
            for name in dirs + names:
                path = Path(directory) / name
                st = path.lstat()
                if path.is_symlink():
                    links.append({'path': str(path), 'target': os.readlink(path), 'resolved': str(path.resolve())})
                    continue
                if not path.is_file():
                    continue
                rel = path.relative_to(base)
                inventory.append({'path': str(path), 'bytes': st.st_size, 'allocated_bytes': st.st_blocks * 512,
                                  'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino, 'nlink': st.st_nlink})
                if rel.parts[0] == 'deps' or '__pycache__' in rel.parts:
                    continue
                if path.name in ('completion.json', 'manifest.json'):
                    record = json.loads(path.read_text())
                    for n, value in record.get('artifacts', {}).items():
                        verify(path.parent / n, value, path)
                if (rel.parts[0] == 'source' or path.suffix in ('.json', '.csv', '.log', '.out', '.err', '.md', '.txt', '.sh', '.py')
                        or (path.suffix == '.npz' and rel.parts[0] in ('runs', 'guides', 'diagnostics'))):
                    assert st.st_size < 64 * 1024 * 1024, path
                    preserve.append(path)
        manifest = base / 'payload_manifest.json'
        for n, value in json.loads(manifest.read_text()).items():
            verify(base / n, value, manifest)
    reference = root / 'reference_manifest.json'
    for n, value in json.loads(reference.read_text())['files'].items():
        verify(Path(n), value, reference)
    # Record all source digests even when no older manifest references a file.
    file_manifest, duplicate_files, selected = {}, {}, []
    for path in sorted(preserve):
        key = str(path.relative_to(HOME))
        digest = hashes.get(str(path)) or sha(path)
        if path.is_relative_to(initial) and path.relative_to(initial).parts[0] in ('runs', 'guides'):
            canonical = prior / path.relative_to(initial)
            if canonical.is_file() and sha(canonical) == digest:
                duplicate_files[key] = {'canonical': str(canonical.relative_to(HOME)), 'sha256': digest}
                continue
        selected.append(path)
        file_manifest[key] = {'sha256': digest, 'bytes': path.stat().st_size}
    queue = command(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j'])
    active = {}
    for row in queue.splitlines():
        job = row.split('|')[0]
        active[job] = {'details': command(['scontrol', 'show', 'job', job]),
                       'batch_script': command(['scontrol', 'write', 'batch_script', job, '-'])}
    report = {'verified_unix': time.time(), 'completion_passed': True, 'main_runs': 24,
              'original_runs': 20, 'checkpoint_audits': 4, 'cuda_preflights': 2,
              'scheduler': scheduler, 'queue': queue, 'active_jobs': active,
              'quota': command(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home']),
              'verified_artifact_references': checked, 'inventory': inventory, 'symlinks': links,
              'preserved_files': file_manifest, 'deduplicated_files': duplicate_files,
              'notes': 'Cancelled initial campaign completed baselines/guides are deduplicated against repaired campaign. Small prediction arrays preserved; checkpoints and geometry arrays verified but not transferred.'}
    output = root / '_finalization'
    output.mkdir(exist_ok=True)
    (output / 'collection_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    archive = output / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in selected:
            tar.add(path, arcname=str(path.relative_to(HOME)), recursive=False)
        tar.add(output / 'collection_manifest.json', arcname='collection_manifest.json')
    receipt = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'preserved_files': len(selected), 'deduplicated_files': len(duplicate_files),
               'verified_artifact_references': len(checked), 'inventory_bytes': sum(x['bytes'] for x in inventory)}
    (output / 'collection_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
