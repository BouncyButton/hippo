"""Copy finished seed-2 ZIPs to reports, verify their contents, then retire their checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.colab_hfh.medsam_archive import verify_zip
from scripts.colab_hfh.medsam_constraints_policy import ARMS

STAGE = 'hfh_medsam3_seed2_20261114'
WORK = STAGE+'/medsam_gain_adaptive_matrix125_seed2_v1'
DEST = Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111/medsam_adaptive_matrix_replication/seed2'
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',os.environ.get('HIPPO_CLUSTER_HOST', 'hippo-cluster')]


def on_cluster(*args):
    return subprocess.check_output([*SSH,shlex.join([str(a) for a in args])],timeout=150)


def retire(name, sha, destination):
    if not destination.is_absolute() or destination.parent!=DEST:
        raise ValueError('Expected local project report archive')
    output = Path(WORK)/'runs'/name
    archive = Path(WORK)/'archives'/(name+'.zip')
    if digest(archive)!=sha:raise ValueError('Remote archive changed')
    verify_zip(archive)
    checkpoint = output/'checkpoint_latest.pt'
    done=read(output/'completion.json')
    log=output/'archive_cleanup.json'
    if done['status']!='complete' or done['updates']!=125 or done['seed']!=2:raise ValueError('Incomplete experiment')
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('results/archive_manifest.json'))
        if any(digest(output/f)!=checksum for f,checksum in manifest.items()):
            raise ValueError('Source metadata changed since local archival')
    receipt=dict(durable_archive=str(destination),archive_sha256=sha,checkpoint_sha256=done['checkpoint_sha256'],
                 removed='checkpoint_latest.pt')
    if checkpoint.is_symlink():raise ValueError('Checkpoint must belong to completed run')
    if checkpoint.exists():
        if digest(checkpoint)!=done['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        write(log,dict(receipt,status='retiring',bytes=checkpoint.stat().st_size))
        checkpoint.unlink()
    previous=read(log)
    assert all(previous[k]==v for k,v in receipt.items())
    write(log,dict(previous,status='verified_and_retired'))
    print(log.read_text())


def self_resume_if_timed_out():
    # A completed matrix, a user pause, or a scientific error must never be relaunched.
    status=json.loads(on_cluster('cat',WORK+'/status.json'))
    if status['status'] in ('complete','error') or status['completed']==42:return
    command="""set -eu
cd hfh_medsam3_seed2_20261114
exec 9>submit.lock
flock -n 9 || exit 0
test ! -f medsam_gain_adaptive_matrix125_seed2_v1/STOP || exit 0
job=$(cat job.id)
test -z "$(squeue -h --me)" || exit 0
state=$(sacct -X -nP -j "$job" -o State | head -1 | cut -d'|' -f1)
case "$state" in
  TIMEOUT|PREEMPTED|NODE_FAIL) ;;
  *) exit 0 ;;
esac
sbatch --parsable --time-min=03:00:00 seed2.sbatch > job.next
mv job.next job.id
cat job.id"""
    on_cluster('bash','-c',command)


def receive():
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected={f'matrix125__{arm}__f{f}_s2' for arm in ARMS for f in range(3)}
        while True:
            done={n for n in expected if (DEST/(n+'.receipt.json')).exists()}
            if done==expected:return
            try:
                # Wait on the remote filesystem. Only finished, atomically published ZIPs cross SSH.
                wait_code="""import pathlib,sys,time,json
p=pathlib.Path(sys.argv[1]); names=set(json.loads(sys.argv[2]))
for i in range(20):
 ready=sorted(f.stem for f in p.glob('*.zip') if f.stem in names)
 if ready:break
 time.sleep(5)
print(json.dumps(ready))"""
                ready=json.loads(on_cluster('python3','-c',wait_code,WORK+'/archives',json.dumps(sorted(expected-done))))
                if not ready:
                    self_resume_if_timed_out()
                for name in ready:
                    if name not in expected:raise ValueError('Unexpected archive')
                    local=DEST/(name+'.zip')
                    path=WORK+'/archives/'+local.name
                    tmp=local.with_suffix('.partial')
                    tmp.write_bytes(on_cluster('cat',path))
                    sha=on_cluster('sha256sum',path).decode().split()[0]
                    row=verify_zip(tmp)
                    if digest(tmp)!=sha or (row['seed'],row['updates'])!=(2,125):raise ValueError('Archive mismatch')
                    if local.exists() and digest(local)!=sha:raise ValueError('Existing local result differs')
                    tmp.replace(local)
                    receipt=json.loads(on_cluster('env','HIPPO_ARCHIVE_ROOT='+str(DEST),'PYTHONPATH='+STAGE+'/bundle/source',
                        os.environ.get('HIPPO_REMOTE_PYTHON', 'python3'),STAGE+'/archive_seed2.py','--retire',name,sha,str(local)))
                    write(DEST/(name+'.receipt.json'),receipt)
                    print('ARCHIVED',name,flush=True)
            except (OSError,subprocess.SubprocessError) as e:
                print('Archive transfer will retry:',e,flush=True);time.sleep(120)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--retire',nargs=3)
    args=p.parse_args()
    if args.retire:
        name,sha,location=args.retire
        if name not in {f'matrix125__{a}__f{f}_s2' for a in ARMS for f in range(3)} or not re.fullmatch('[a-f0-9]{64}',sha):
            raise ValueError('Invalid completed-run identity')
        # The collector supplies its canonical off-cluster project reports path.
        DEST=Path(os.environ['HIPPO_ARCHIVE_ROOT'])
        retire(name,sha,Path(location))
    else:receive()
