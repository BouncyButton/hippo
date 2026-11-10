"""Archive evidence and clean only generated outputs of completed job 688313."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path('/mnt/beegfsstudents/home/3160552/tissue_profiles_fold0_seed0_20261007_01')
WORK = Path('/tmp/tissue_688313_finalization_20261008')
GENERATED_DIRS = {'runs', 'logs', 'diagnostic_geometry'}
GENERATED_FILES = {'comparison.json', 'status.json', 'environment.json', 'local_checks.json', 'unit_checks.json', 'slurm-688313.out', 'slurm-688313.err'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, check=True)
    return p.stdout + p.stderr


def save(name, value):
    (WORK / name).write_text(json.dumps(value, indent=2) + '\n')


def selected(path):
    rel = path.relative_to(ROOT)
    return rel.parts[0] in GENERATED_DIRS or (len(rel.parts) == 1 and (rel.name in GENERATED_FILES or rel.name.endswith('_cuda_preflight.json')))


def dependencies():
    queue = command(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j|%Z'])
    assert not queue.strip(), ('Active/pending jobs need dependency review before cleanup', queue)
    links = []
    excluded = {'.conda', '.cache', '.local', '.git', 'datasets', 'deps', 'node_modules', 'imagesTr', 'labelsTr', ROOT.name}
    for directory, dirs, files in os.walk(ROOT.parent, followlinks=False):
        dirs[:] = [n for n in dirs if n not in excluded]
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink():
                target = p.resolve()
                if target == ROOT or ROOT in target.parents:
                    links.append({'link': str(p), 'target': str(target)})
    return {'queue': queue, 'external_links': links, 'excluded_from_link_scan': sorted(excluded), 'checked_unix': time.time()}


def collect():
    assert ROOT.resolve() == ROOT and not ROOT.is_symlink()
    WORK.mkdir(exist_ok=True)
    status = json.loads((ROOT / 'status.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert status['state'] == 'complete' and status['completed_main_runs'] == 36
    assert len(protocol['runs']) == 36 and all(s['returncode'] == 0 for s in status['stages'])
    accounting = command(['sacct', '-j', '688313', '--format=JobID,State,Elapsed,Start,End,ExitCode', '-P'])
    assert any(line.startswith('688313|COMPLETED|') and line.endswith('|0:0') for line in accounting.splitlines())
    checked = []
    for spec in protocol['runs']:
        run = ROOT / 'runs' / spec['id']
        c = json.loads((run / 'completion.json').read_text())
        assert c['identity']['run'] == spec
        assert c['identity']['protocol_sha256'] == status['protocol_sha256']
        val = json.loads((run / 'validation_best.json').read_text())
        assert len(val['cases']) == 52
        assert abs(sum(x['macro_dice'] for x in val['cases']) / 52 - c['validation']['macro_dice']) < 1e-12
        for name, expected in c['artifacts'].items():
            p = run / name
            assert not p.is_symlink() and p.resolve().parent == run
            actual = digest(p)
            assert actual == expected, str(p)
            checked.append({'path': str(p.relative_to(ROOT)), 'sha256': actual})
    comparison = json.loads((ROOT / 'comparison.json').read_text())
    assert len(comparison['rows']) == 42
    dep = dependencies()
    inventory, evidence = [], []
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [n for n in dirs if n != '__pycache__']
        for name in files:
            p = Path(directory) / name
            assert not p.is_symlink(), ('Review file symlink', str(p))
            st = p.lstat()
            if selected(p):
                for link in dep['external_links']:
                    target = Path(link['target'])
                    assert not (p == target or target in p.parents), ('Externally shared output', str(p), link)
                inventory.append({'path': str(p), 'relative': str(p.relative_to(ROOT)), 'bytes': st.st_size,
                    'allocated_bytes': st.st_blocks * 512, 'inode': st.st_ino, 'mtime_ns': st.st_mtime_ns})
            # No checkpoint, prediction array, geometry array, cache or deployment binary copies.
            if p.suffix in {'.json', '.csv', '.md', '.py', '.sh', '.log', '.out', '.err', '.txt'}:
                evidence.append({'relative': str(p.relative_to(ROOT)), 'bytes': st.st_size, 'sha256': digest(p)})
    verification = {'job': '688313', 'complete': True, 'runs': 36, 'conditions': 42,
        'artifact_hashes_verified': checked, 'accounting': accounting, 'dependencies': dep,
        'quota_before': command(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home']),
        'comparison_sha256': digest(ROOT / 'comparison.json')}
    plan = {'root': str(ROOT), 'job': '688313', 'created_unix': time.time(), 'delete': inventory,
        'delete_bytes': sum(x['bytes'] for x in inventory), 'delete_allocated_bytes': sum(x['allocated_bytes'] for x in inventory),
        'preserve': 'Source, protocol/deployment records, shared atlas link and all paths outside the exact root.'}
    save('verification.json', verification)
    save('cleanup_plan.json', plan)
    save('evidence_manifest.json', {'root': str(ROOT), 'files': evidence})
    archive = WORK / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        for entry in evidence:
            t.add(ROOT / entry['relative'], arcname='evidence/' + entry['relative'], recursive=False)
        for name in ('verification.json', 'cleanup_plan.json', 'evidence_manifest.json'):
            t.add(WORK / name, arcname=name, recursive=False)
    receipt = {'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
        'files': len(evidence), 'verified_artifacts': len(checked), 'cleanup_plan_sha256': digest(WORK / 'cleanup_plan.json'),
        'delete_bytes': plan['delete_bytes'], 'delete_files': len(inventory)}
    save('collection_receipt.json', receipt)
    print(json.dumps(receipt))


def execute():
    plan = json.loads((WORK / 'cleanup_plan.json').read_text())
    preserved = json.loads((WORK / 'preservation_receipt.json').read_text())
    assert preserved['local_archive_verified'] and preserved['all_evidence_hashes_verified'] and preserved['report_verified']
    assert preserved['cleanup_plan_sha256'] == digest(WORK / 'cleanup_plan.json')
    assert preserved['archive_sha256'] == digest(WORK / 'evidence.tar.gz')
    assert preserved['report_sha256'] == digest(WORK / 'RESULTS.md')
    assert plan['root'] == str(ROOT) and ROOT.resolve() == ROOT
    dep = dependencies()
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'complete'
    for entry in plan['delete']:
        p = Path(entry['path'])
        assert ROOT in p.parents and selected(p)
        assert not any(a.is_symlink() for a in [p] + list(p.parents))
        st = p.lstat()
        assert (st.st_ino, st.st_size, st.st_mtime_ns) == (entry['inode'], entry['bytes'], entry['mtime_ns'])
        for link in dep['external_links']:
            target = Path(link['target'])
            assert not (p == target or target in p.parents)
    receipt = {'job': '688313', 'root': str(ROOT), 'started_unix': time.time(), 'preservation': preserved,
        'dependencies': dep, 'deleted': [], 'success': False,
        'quota_before': command(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'])}
    try:
        for entry in plan['delete']:
            Path(entry['path']).unlink()
            receipt['deleted'].append(entry)
        for name in GENERATED_DIRS:
            for directory, dirs, files in os.walk(ROOT / name, topdown=False, followlinks=False):
                p = Path(directory)
                if not p.is_symlink() and not list(p.iterdir()):
                    p.rmdir()
        assert all(not os.path.lexists(x['path']) for x in plan['delete'])
        assert not any((ROOT / n).exists() for n in GENERATED_DIRS)
        receipt['success'] = True
    finally:
        receipt['finished_unix'] = time.time()
        receipt['deleted_bytes'] = sum(x['bytes'] for x in receipt['deleted'])
        receipt['deleted_allocated_bytes'] = sum(x['allocated_bytes'] for x in receipt['deleted'])
        receipt['quota_immediate'] = command(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'])
        save('cleanup_receipt.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k not in {'deleted', 'preservation', 'dependencies'}}))


if __name__ == '__main__':
    {'collect': collect, 'execute': execute}[sys.argv[1]]()
