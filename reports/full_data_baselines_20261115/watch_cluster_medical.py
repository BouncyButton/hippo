"""Collect complete ADNI controls and admit them after the 36 primary MNI scratch runs."""
import argparse
import fcntl
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from cluster_medical import KEYS
from watch_matched_mni import D, HOST, STAGE, ssh

sys.path.insert(0, str(D.parents[1]))
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from comparisons import medical_comparison

REMOTE = STAGE / 'cluster_medical'
LOCAL = D / 'archive/cluster_medical'
PACKAGE = D / 'cluster_adni_medical.zip'
SUFFIX = '_coverage_control_results.zip'
WORKER_FILES = ('cluster_medical.py','covered_medical.py')


def collect(key):
    out = REMOTE / 'work' / key
    dest = LOCAL / key
    if not dest.is_dir():
        with tempfile.TemporaryDirectory(dir=LOCAL) as tmp:
            stage = Path(tmp) / key
            stage.mkdir()
            subprocess.run(['rsync', '-rt', '--exclude=*.pt', '--exclude=*.tmp', '--exclude=*.nii.gz',
                            f'{HOST}:{out}/', str(stage) + '/'], check=True)
            receipt = read(stage / 'archive_manifest.json')
            verify_archive(stage, receipt)
            assert receipt['task']['id'] == key
            stage.rename(dest)
    manifest = read(dest / 'archive_manifest.json')
    verify_archive(dest, manifest)
    identity = read(dest / 'cluster_protocol.json')
    assert identity['input_manifest'] == ssh('sha256sum', REMOTE/'SHA256SUMS').split()[0]
    with zipfile.ZipFile(PACKAGE) as source:
        assert hashlib.sha256(source.read('SHA256SUMS')).hexdigest() == identity['input_manifest']
    cell = read(D/'manifest.json')[key]
    split = read(D/'splits.json')[cell['split']]
    result = medical_comparison(key, cell, split)
    assert result and abs(result['dice'] - read(dest/'completion.json')['validation_dice']) < 1e-12
    archive = LOCAL / (key + SUFFIX)
    if not archive.is_file():
        with zipfile.ZipFile(archive.with_suffix('.partial'), 'w', zipfile.ZIP_DEFLATED) as z:
            for name in sorted(manifest['metadata']):
                z.write(dest / name, name)
            z.write(dest / 'archive_manifest.json', 'archive_manifest.json')
            with zipfile.ZipFile(PACKAGE) as source:
                for name in (*WORKER_FILES, 'SHA256SUMS', 'protocols/'+key+'.json', 'references/'+key+'.json'):
                    z.writestr(name, source.read(name))
        archive.with_suffix('.partial').replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and json.loads(z.read('completion.json')) == read(dest/'completion.json')
    ack = dict(status='verified', source_verified=True, durable_archive=str(dest),
               manifest_sha256=digest(dest/'archive_manifest.json'), input_manifest_sha256=identity['input_manifest'])
    write(dest/'archive_acknowledgment.json', ack)
    code = """import json,sys
from pathlib import Path
from scripts.cross_dataset_campaign.common import write,digest
from scripts.cross_dataset_campaign.storage import retire
p=Path(sys.argv[1]);ack=json.loads(sys.argv[2]);assert digest(p/'archive_manifest.json')==ack['manifest_sha256']
write(p/'archive_acknowledgment.json',ack);retire(p)"""
    ssh('env', 'PYTHONPATH='+str(STAGE/'cluster_medical/source'), 'python3', '-c', code, out, json.dumps(ack))
    subprocess.run(['rsync', '-rt', f'{HOST}:{out}/cleanup.json', str(dest)], check=True)
    assert read(dest/'cleanup.json')['status'] == 'complete'
    subprocess.run([sys.executable, str(D/'comparisons.py')], check=True)
    print('Archived', key, result['dice'], flush=True)


ADMIT = """import fcntl,json,pathlib,subprocess,sys,time
from scripts.cross_dataset_campaign.storage import quota
p=pathlib.Path(sys.argv[1]);s=p.parent
def queue(): return dict(x.split() for x in subprocess.check_output(['squeue','--me','-h','-o','%i %T'],text=True).splitlines())
with (s/'campaign_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if len(list((s/'matched_MNI/work').glob('*/durable_ack.json')))!=36:sys.exit()
 jobs=queue(); receipt=p/'scheduler.json'
 if len(jobs)>1 or any(st not in ('RUNNING','COMPLETING','CONFIGURING') for st in jobs.values()):sys.exit()
 if receipt.exists():
  prev=json.loads(receipt.read_text())['job']
  if prev in jobs:sys.exit()
  result=subprocess.check_output(['sacct','-X','-j',prev,'-nPo','State,ExitCode'],text=True).strip()
  if result!='FAILED|75:0' and not result.startswith('TIMEOUT|'):print('Previous job',prev,result);sys.exit()
 q=quota(); assert q['free_gib']>=24
 cmd=['sbatch','--parsable','--output='+str(p/'slurm-%j.out')]
 if jobs:cmd.append('--dependency=afterany:'+next(iter(jobs)))
 batch=p/('cluster_hfh_gain.sbatch' if p.name=='cluster_hfh_gain' else 'cluster_medical.sbatch')
 job=subprocess.check_output(cmd+[str(batch)],cwd=p,text=True).strip().split(';')[0]; assert job.isdigit()
 actual=queue()
 if len(actual)>2:
  if actual.get(job)=='PENDING':subprocess.run(['scancel',job],check=True)
  raise RuntimeError('Concurrent submission')
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(dict(job=job,time=time.time(),quota=q)));tmp.replace(receipt)
 print('Admitted',p.name,job)"""


def once():
    status = """import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'work';print(json.dumps({d.name:(d/'cleanup.json').is_file() for d in p.iterdir() if (d/'archive_manifest.json').is_file()}))"""
    ready = json.loads(ssh('python3','-c',status,REMOTE))
    assert set(ready) <= set(KEYS)
    for key, cleaned in ready.items():
        if not cleaned or not (LOCAL/key/'cleanup.json').is_file():
            collect(key)
    if all((LOCAL/key/'cleanup.json').is_file() for key in KEYS):
        return True
    msg = ssh('env','PYTHONPATH='+str(STAGE/'cluster_medical/source'),'python3','-c',ADMIT,REMOTE)
    if msg.strip(): print(msg.strip(), flush=True)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--hfh-gain', action='store_true')
    args = parser.parse_args()
    if args.hfh_gain:
        from cluster_hfh_gain import KEYS
        REMOTE = STAGE / 'cluster_hfh_gain'
        LOCAL = D / 'archive/cluster_hfh_gain'
        PACKAGE = D / 'cluster_hfh_gain.zip'
        SUFFIX = '_gain_coverage_control_results.zip'
        WORKER_FILES = ('cluster_hfh_gain.py','cluster_hfh_gain.sbatch','original_inputs.json')
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'monitor.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if once() or args.once: break
            except (OSError,subprocess.CalledProcessError) as exc:
                if args.once: raise
                print('ADNI collection will retry', type(exc).__name__, flush=True)
            time.sleep(60)
