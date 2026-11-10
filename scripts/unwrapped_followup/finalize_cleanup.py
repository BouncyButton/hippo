"""Plan or perform allowlisted cleanup after verified local preservation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HOME = Path('/mnt/beegfsstudents/home/3160552')
ROOTS = [HOME / n for n in ('unwrapped_followup_fold0_20261007_01',
    'unwrapped_swin_fold0_seed0_20261006_02', 'unwrapped_swin_fold0_seed0_20261006_01')]
WORK = ROOTS[0] / '_finalization'


def cmd(args):
    r = subprocess.run(args, capture_output=True, text=True, check=True)
    return r.stdout + r.stderr


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dependencies():
    queue = cmd(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j'])
    active, required = {}, {}
    for row in queue.splitlines():
        job = row.split('|')[0]
        details = cmd(['scontrol', 'show', 'job', job])
        script = cmd(['scontrol', 'write', 'batch_script', job, '-'])
        active[job] = dict(details=details, batch_script=script)
        workdir = next((s.split('=', 1)[1] for s in details.split() if s.startswith('WorkDir=')), None)
        assert workdir, 'Missing active job working directory'
        directory = Path(workdir)
        assert not any(directory.resolve().is_relative_to(r.resolve()) for r in ROOTS), 'Active job inside cleanup scope'
        base = directory.parent if directory.name == 'source' else directory
        for filename in ('reference_manifest.json', 'branch_controls.json'):
            manifest = base / filename
            if manifest.exists():
                for path, digest in json.loads(manifest.read_text()).get('files', {}).items():
                    p = Path(path)
                    if any(p.resolve().is_relative_to(r.resolve()) for r in ROOTS):
                        required[str(p.resolve())] = {'sha256':digest, 'consumer':str(manifest)}
    links = []
    skip = {'.conda', '.cache', '.local', '.git', 'datasets', 'deps', 'node_modules', 'imagesTr', 'labelsTr'}
    for d, dirs, files in os.walk(HOME, followlinks=False):
        dirs[:] = [n for n in dirs if n not in skip and n not in {r.name for r in ROOTS}]
        for n in dirs + files:
            p = Path(d) / n
            if p.is_symlink() and any(p.resolve().is_relative_to(r.resolve()) for r in ROOTS):
                links.append({'path':str(p), 'target':str(p.resolve())})
    return dict(queue=queue, active_jobs=active, required_files=required,
                external_symlinks=links, scan_excluded_directories=sorted(skip))


def protected(path, dep):
    base = next(r for r in ROOTS if path.is_relative_to(r))
    rel = path.relative_to(base)
    if rel.parts[0] in ('source', 'deps', '_finalization'):
        return 'source, runtime dependencies or finalization receipt'
    resolved = str(path.resolve())
    if resolved in dep['required_files']:
        return 'active-job frozen reference'
    for link in dep['external_symlinks']:
        target = Path(link['target'])
        if path.resolve() == target or path.resolve().is_relative_to(target):
            return 'externally shared symlink target'
    return None


def make_plan():
    assert json.loads((ROOTS[0] / 'status.json').read_text())['state'] == 'complete'
    manifest = json.loads((WORK / 'collection_manifest.json').read_text())
    assert manifest['completion_passed'] and manifest['main_runs'] == 24 and manifest['checkpoint_audits'] == 4
    dep = dependencies()
    remove, keep = [], []
    for root in ROOTS:
        for d, dirs, files in os.walk(root, followlinks=False):
            for n in dirs + files:
                p = Path(d) / n
                if not p.is_symlink() and not p.is_file():
                    continue
                st = p.lstat()
                entry = dict(path=str(p), bytes=st.st_size, allocated_bytes=st.st_blocks*512,
                             inode=st.st_ino, mtime_ns=st.st_mtime_ns, symlink=p.is_symlink())
                reason = protected(p, dep)
                if reason:
                    entry['reason'] = reason
                    keep.append(entry)
                else:
                    remove.append(entry)
    value = dict(created_unix=time.time(), roots=[str(r) for r in ROOTS], dependencies=dep,
                 delete=remove, retain=keep, bytes_to_delete=sum(e['bytes'] for e in remove),
                 allocated_bytes_to_delete=sum(e['allocated_bytes'] for e in remove),
                 bytes_to_retain=sum(e['bytes'] for e in keep))
    (WORK / 'cleanup_plan.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


def execute():
    plan = json.loads((WORK / 'cleanup_plan.json').read_text())
    preservation = json.loads((WORK / 'preservation_receipt.json').read_text())
    assert preservation['local_hash_verification_passed'] and preservation['pdf_visual_inspection_passed']
    assert preservation['analysis_validation_passed'] and preservation['scientific_runs'] == 44
    assert len(preservation['pdf_sha256']) == 64 and len(preservation['preservation_manifest_sha256']) == 64
    assert preservation['cleanup_plan_sha256'] == sha(WORK / 'cleanup_plan.json')
    assert preservation['archive_sha256'] == sha(WORK / 'evidence.tar.gz')
    assert plan['roots'] == [str(r) for r in ROOTS]
    current = dependencies()
    # All new references and consumers are checked again just before the first deletion.
    for e in plan['delete']:
        p = Path(e['path'])
        assert any(p.is_relative_to(r) for r in ROOTS) and p not in ROOTS
        assert not any(a.is_symlink() for a in p.parents if any(a.is_relative_to(r) for r in ROOTS))
        assert protected(p, current) is None, ('New shared dependency', str(p))
        st = p.lstat()
        assert (st.st_ino, st.st_size, st.st_mtime_ns, p.is_symlink()) == (e['inode'], e['bytes'], e['mtime_ns'], e['symlink']), ('File changed', str(p))
    for path, entry in current['required_files'].items():
        assert sha(path) == entry['sha256'], ('Active job reference changed', path)
    quota_before = cmd(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'])
    receipt = dict(started_unix=time.time(), preservation=preservation, dependencies=current,
                   quota_before=quota_before, deleted=[], errors=[])
    try:
        for e in plan['delete']:
            Path(e['path']).unlink()
            receipt['deleted'].append(e)
        for root in ROOTS:
            for d, dirs, files in os.walk(root, topdown=False, followlinks=False):
                p = Path(d)
                if p != root and not p.is_symlink() and not list(p.iterdir()):
                    if p.relative_to(root).parts[0] not in ('source','deps','_finalization'):
                        p.rmdir()
        for e in plan['delete']:
            assert not os.path.lexists(e['path'])
        for path, entry in current['required_files'].items():
            assert sha(path) == entry['sha256']
        for link in current['external_symlinks']:
            assert Path(link['path']).exists()
    except Exception as error:
        receipt['errors'].append(repr(error))
        raise
    finally:
        receipt.update(finished_unix=time.time(), deleted_bytes=sum(e['bytes'] for e in receipt['deleted']),
                       deleted_allocated_bytes=sum(e['allocated_bytes'] for e in receipt['deleted']),
                       success=not receipt['errors'], retained_required_files_verified=not receipt['errors'])
        (WORK / 'cleanup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    execute() if parser.parse_args().execute else make_plan()
