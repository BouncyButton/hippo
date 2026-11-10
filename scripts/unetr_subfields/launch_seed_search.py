"""Deploy a separate search, preserving and resuming the interrupted attempt."""
import json
from pathlib import Path
import subprocess
import shutil
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.unetr_subfields.seed_search import ROOT, NAME, PARENT, verify


def main():
    stage, report = ROOT / 'experiments' / NAME, ROOT / 'reports' / NAME
    destination, parent = REMOTE / NAME, REMOTE / PARENT
    plan = verify(stage)
    report.mkdir(parents=True, exist_ok=True)
    if (report / 'submission.json').exists():
        raise RuntimeError('Already submitted; inspect the existing search rather than relaunching')
    python = ROOT / '.venv/bin/python'
    subprocess.run([str(python), '-c', 'import inspect,tarfile; assert "filter" in inspect.signature(tarfile.TarFile.extractall).parameters'], check=True)
    if remote(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j']).strip():
        raise RuntimeError('Wait for the interrupted sweep to leave the queue')
    old_control = json.loads(remote(['cat', parent / 'control.json']))
    old_status = json.loads(remote(['cat', parent / 'status.json']))
    if not old_control['pause'] or old_status['status'] != 'paused':
        raise RuntimeError('Original campaign is not safely paused')
    quota = remote(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'])
    import re
    match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
    if not match or float(match[2]) - float(match[1]) < plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient verified quota headroom')
    write(report / 'prelaunch.json', dict(parent_status=old_status, quota=quota, checked_at=time.time()))
    code = ('import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); m=p/"manifest.json"; '
            'assert not m.exists() or hashlib.sha256(m.read_bytes()).hexdigest()==sys.argv[2], "Stage differs"; '
            'assert not (p/"submission.json").exists(), "Already submitted"; p.mkdir(parents=True,exist_ok=True)')
    remote(['python3', '-c', code, destination, digest(stage / 'manifest.json')])
    code = '''import hashlib,shutil,sys
from pathlib import Path
parent,destination=map(Path,sys.argv[1:3])
assert hashlib.sha256((parent/"manifest.json").read_bytes()).hexdigest()==sys.argv[3]
for name in ("data","source"):
 shutil.copytree(parent/name,destination/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))'''
    remote(['python3', '-c', code, parent, destination, digest(ROOT / 'experiments' / PARENT / 'manifest.json')])
    subprocess.run(['rsync', '-az', '--exclude=__pycache__', str(stage) + '/', f'bocconi-cluster:{destination}/'], check=True)
    # Atomic move within BeeGFS: no checkpoint copy, deletion, reset or extra updates.
    code = '''import hashlib,json,sys,time
from pathlib import Path
old,new=map(Path,sys.argv[1:]); name="COBRA_unetr_vit_b16_dice_f0s1"
src,dst=old/"runs"/name,new/"runs"/name
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024**2),b""): h.update(block)
 return h.hexdigest()
receipt=new/"migration.json"
if receipt.exists():
 assert dst.is_dir() and not src.exists()
 print(receipt.read_text()); sys.exit()
assert src.is_dir() and not dst.exists()
assert json.loads((old/"control.json").read_text())["pause"]
assert not (src/"completion.json").exists()
ledger=json.loads((src/"budget_ledger.json").read_text())
assert ledger["charged_attempts"]==115
before={str(f.relative_to(src)):digest(f) for f in src.rglob("*") if f.is_file()}
src.rename(dst)
assert before=={str(f.relative_to(dst)):digest(f) for f in dst.rglob("*") if f.is_file()}
record=dict(status="verified_migration",task=name,source=str(src),destination=str(dst),sha256=before,
 charged_attempts=ledger["charged_attempts"],training_flops=ledger["training_flops"],time=time.time(),
 reason="User interrupted the fixed sweep and authorized seed search; preserve and resume this attempt under its original cap.")
receipt.write_text(json.dumps(record,indent=2)+"\\n")
(old/"migration_to_seed_search.json").write_text(json.dumps(record,indent=2)+"\\n")
print(json.dumps(record))'''
    migration = json.loads(remote(['python3', '-c', code, parent, destination], timeout=300))
    write(report / 'migration.json', migration)
    write(ROOT / 'reports' / PARENT / 'migration_to_seed_search.json', migration)
    environment = ':'.join(str(destination / name) for name in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    command = ['env', 'PYTHONPATH=' + environment, plan['conventional_python'], '-m',
               'scripts.unetr_subfields.seed_search']
    print(remote(command + ['preflight', '--stage', destination], timeout=600), flush=True)
    # Imported result manifests include empty run.lock files. The generic
    # collector's copy ignores *.lock, so explicitly retain frozen manifest members.
    frozen = report / 'frozen'
    if not frozen.exists():
        shutil.copytree(stage, frozen, ignore=lambda path, names:
            [name for name in names if name in ('runs', 'submissions')] if Path(path) == stage else [])
    for name, sha in read(frozen / 'manifest.json').items():
        target = frozen / name
        if not target.exists():
            if digest(stage / name) != sha:
                raise ValueError('Cannot repair a changed frozen archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / name, target)
        if digest(target) != sha:
            raise ValueError('Frozen collector archive differs: ' + name)
    log = (report / 'collector.log').open('a')
    collector = subprocess.Popen(['caffeinate', '-i', str(python), '-u', '-m',
        'scripts.cross_dataset_campaign.collect', '--stage', str(destination),
        '--local-stage', str(stage), '--destination', str(report)], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write(report / 'collector_process.json', dict(pid=collector.pid, python=str(python), started_at=time.time()))
    receipt = json.loads(remote(command + ['submit', '--stage', destination]))
    write(report / 'submission.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
