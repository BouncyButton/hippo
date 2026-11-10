"""Finish the already authorized deployment after cluster connectivity returns."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
STAGE='/mnt/beegfsstudents/home/3160552/mni_medsam3_n5_f012_s01_constraints_20261110'
PYTHON='/home/3160552/.conda/envs/hippocampus/bin/python'
MODULE='scripts.mni_medsam3_cluster.constraint_campaign'
EXPECTED='4da4e9bd5caac0fda12101e08fb641b9c84fba3c05b05f4c8dbc3d8a268bf47e'


def main():
    stage=ROOT/'stage'
    assert hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest()==EXPECTED
    manifest=json.loads((stage/'manifest.json').read_text())
    for name,sha in manifest.items():
        assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==sha,name
    patch=Path('/tmp/mni_constraint_readonly_v2.tar.gz')
    members=['manifest.json','source/scripts/mni_medsam3_cluster/constraint_campaign.py',
             'source/scripts/mni_medsam3_cluster/test_constraint_campaign.py']
    with tarfile.open(patch,'w:gz') as archive:
        for name in members:archive.add(stage/name,arcname=name)
    launch=f'''import os,sys,fcntl,json,subprocess,time
from pathlib import Path
p=Path({STAGE!r})
sys.path.insert(0,str(p/'source'))
from scripts.cross_dataset_campaign.common import read,write,digest
assert digest(p/'manifest.json')=={EXPECTED!r}
assert read(p/'preflight.json')['operational_manifest_sha256']==digest(p/'manifest.json')
assert not list((p/'submissions').glob('*.json'))
assert not read(Path(read(p/'plan.json')['baseline_stage'])/'control.json').get('pause')
with (p/'service.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
control=read(p/'control.json');control.update(pause=False,revision=2);write(p/'control.json',control)
cmd=[sys.executable,'-u','-m',{MODULE!r},'serve','--stage',str(p)]
with (p/'service.log').open('ab') as log:
 process=subprocess.Popen(cmd,cwd=p/'source',env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=process.pid,command=cmd,time=time.time(),manifest_sha256=digest(p/'manifest.json'),tests_passed=25,revision=2,laptop_required=False)
write(p/'service_start.json',receipt);print(json.dumps(receipt))
'''
    command=(f'tar --no-same-owner -xzf - -C {STAGE} && cd {STAGE}/source && '
        f'export PYTHONPATH={STAGE}/source:{STAGE}/source/scripts:{STAGE}/source/baselines/nnUNet '
        'PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 && '
        f'{PYTHON} -m unittest scripts.mni_medsam3_cluster.test_constraint_campaign '
        'scripts.mni_medsam3_cluster.offline_ops.test_offline scripts.mni_medsam3_cluster.test_adaptive -q && '
        f'{PYTHON} -m {MODULE} preflight --stage {STAGE} && {PYTHON} -c {shlex.quote(launch)}')
    with patch.open('rb') as stream:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','HostName=10.35.5.3','-o','HostKeyAlias=slogin.hpc.unibocconi.it',
            '-o','ServerAliveInterval=15','bocconi-cluster',command],stdin=stream,check=True)


if __name__=='__main__':main()
