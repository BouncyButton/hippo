"""Deploy the Geo recipe search and first retire the abandoned Dice+CE checkpoint."""
import json
import re
import shutil
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.unetr_subfields.geo_search import ROOT, NAME, PARENT, ABANDONED, verify


def main():
    stage, report = ROOT / 'experiments' / NAME, ROOT / 'reports' / NAME
    destination, parent = REMOTE / NAME, REMOTE / PARENT
    plan = verify(stage)
    if (report / 'submission.json').exists(): raise RuntimeError('Search already submitted')
    python = ROOT / '.venv/bin/python'
    subprocess.run([str(python), '-c', 'import inspect,tarfile; assert "filter" in inspect.signature(tarfile.TarFile.extractall).parameters'], check=True)
    if remote(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j']).strip():
        raise RuntimeError('Old allocation must exit before replacement submission')
    old = json.loads(remote(['cat', parent / 'status.json']))
    if old['status'] != 'paused' or old['active'] != ABANDONED or not json.loads(remote(['cat', parent / 'control.json']))['pause']:
        raise RuntimeError('Unexpected original campaign state')
    quota = remote(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'])
    match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
    if not match or float(match[2]) - float(match[1]) < plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient storage headroom')
    write(report / 'prelaunch.json', dict(parent_status=old, quota=quota, checked_at=time.time()))
    code = '''import hashlib,shutil,sys
from pathlib import Path
parent,destination=map(Path,sys.argv[1:3]); manifest=destination/"manifest.json"
assert not (destination/"submission.json").exists()
assert not manifest.exists() or hashlib.sha256(manifest.read_bytes()).hexdigest()==sys.argv[3]
assert hashlib.sha256((parent/"manifest.json").read_bytes()).hexdigest()==sys.argv[4]
destination.mkdir(parents=True,exist_ok=True)
for name in ("data","source"):
 shutil.copytree(parent/name,destination/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))'''
    remote(['python3', '-c', code, parent, destination, digest(stage / 'manifest.json'),
            digest(ROOT / 'experiments' / PARENT / 'manifest.json')])
    subprocess.run(['rsync', '-az', '--exclude=__pycache__', str(stage) + '/', f'bocconi-cluster:{destination}/'], check=True)
    code = '''import hashlib,json,sys,time
from pathlib import Path
old,new=map(Path,sys.argv[1:3]); name=sys.argv[3]
src,dst=old/"runs"/name,new/"runs"/("abandoned_"+name); receipt=new/"migration.json"
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024**2),b""): h.update(block)
 return h.hexdigest()
if receipt.exists():
 assert dst.is_dir() and not src.exists(); print(receipt.read_text()); sys.exit()
assert src.is_dir() and not dst.exists() and not (src/"completion.json").exists()
assert json.loads((old/"control.json").read_text())["pause"]
ledger=json.loads((src/"budget_ledger.json").read_text()); assert ledger["charged_attempts"]==180
before={str(f.relative_to(src)):digest(f) for f in src.rglob("*") if f.is_file()}
dst.parent.mkdir(parents=True,exist_ok=True); src.rename(dst)
assert before=={str(f.relative_to(dst)):digest(f) for f in dst.rglob("*") if f.is_file()}
record=dict(status="verified_migration_for_retirement",source=str(src),destination=str(dst),sha256=before,
 charged_attempts=ledger["charged_attempts"],training_flops=ledger["training_flops"],time=time.time(),
 reason="User abandoned ordinary Dice+CE sweep. Evaluate and archive stopped checkpoint, then delete generated artifacts; initialize Geo recipe runs from fresh random weights.")
receipt.write_text(json.dumps(record,indent=2)+"\\n")
(old/"migration_to_geo_retirement.json").write_text(json.dumps(record,indent=2)+"\\n")
print(json.dumps(record))'''
    migration = json.loads(remote(['python3', '-c', code, parent, destination, ABANDONED], timeout=300))
    write(report / 'migration.json', migration)
    write(ROOT / 'reports' / PARENT / 'migration_to_geo_retirement.json', migration)
    environment = ':'.join(str(destination / name) for name in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    command = ['env', 'PYTHONPATH=' + environment, plan['conventional_python'], '-m', 'scripts.unetr_subfields.geo_search']
    print(remote(command + ['preflight', '--stage', destination], timeout=600), flush=True)
    log = (report / 'collector.log').open('a')
    collector = subprocess.Popen(['caffeinate', '-i', str(python), '-u', '-m', 'scripts.cross_dataset_campaign.collect',
        '--stage', str(destination), '--local-stage', str(stage), '--destination', str(report)], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write(report / 'collector_process.json', dict(pid=collector.pid, started_at=time.time(), python=str(python)))
    log = (report / 'summary_watcher.log').open('a')
    watcher = subprocess.Popen(['caffeinate', '-i', str(python), '-u', '-m', 'scripts.summarize_unetr_seed_search',
        '--report', str(report), '--watch'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write(report / 'summary_process.json', dict(pid=watcher.pid, started_at=time.time()))
    receipt = json.loads(remote(command + ['submit', '--stage', destination]))
    write(report / 'submission.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
