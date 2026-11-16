"""Pull compact diagnostic records and verify them against the held-out splits."""
import os
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REMOTE=os.environ.get('HIPPO_CLUSTER_HOST', 'hippo-cluster')
CLUSTER_RESULTS=os.environ['HIPPO_REMOTE_HOME'].rstrip('/') + '/reports/paper_spatial_all_models_20261116/'
STAGE=os.environ['HIPPO_CLUSTER_ROOT'].rstrip('/') + '/paper_spatial_all_models_20261116/'
LOCAL=HERE/'results'


def command(args):
    return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=90)


def check():
    jobs=json.loads((HERE/'jobs.json').read_text())
    done=[]
    for marker in sorted(LOCAL.glob('*/complete.json')):
        key=marker.parent.name
        rec=json.loads(marker.read_text())
        assert rec['key']==key and rec['recipe_sha256']==jobs[key]['recipe_sha256']
        for n,v in rec['checks'].items():
            p=marker.parent/n
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==v
        for t in (5,50):
            with (marker.parent/f'u{t}.csv').open() as f:rows=list(csv.DictReader(f))
            assert [r['case'] for r in rows]==jobs[key]['val']
            assert all(int(r['step'])==t and r['model']==jobs[key]['model'] for r in rows)
        done.append(key)
    return done


def main(job_id):
    assert job_id.isdigit()
    LOCAL.mkdir(exist_ok=True)
    previous=None
    hiccups=0
    while True:
        try:
            command(['rsync','-rlt','--exclude=*.tmp',REMOTE+':'+CLUSTER_RESULTS,str(LOCAL)+'/'])
            done=check()
            sched=command(['ssh','-o','BatchMode=yes',REMOTE,
                          'squeue --noheader -j '+job_id+' -o "%T|%R|%M"; tail -5 '+STAGE+'slurm-'+job_id+'.log 2>/dev/null || true']).strip()
            errors=[]
            for f in LOCAL.glob('*/log.txt'):
                if f.parent.name in done:
                    continue
                s=f.read_text(errors='replace')[-4096:]
                if 'Traceback (most recent call last)' in s or 'srun: error' in s:
                    errors.append((f.parent.name,s.splitlines()[-3:]))
            status=(len(done),len(errors),sched.split('|')[0].split('\n')[0])
            if status!=previous:
                if previous is None or len(done)!=previous[0]:
                    print(command([sys.executable, str(HERE/'summarize_models.py')]).strip(),flush=True)
                print(time.strftime('%H:%M:%S'),status,str(errors[:1]),sched.splitlines()[:1],flush=True)
                previous=status
            if len(done)==48:return
            if not sched.split('|')[0] in ('PENDING','RUNNING','COMPLETING','CONFIGURING'):
                return
            hiccups=0
        except (subprocess.SubprocessError,OSError,ValueError,AssertionError) as exc:
            print(str(exc)[:240],flush=True);hiccups+=1
            if hiccups>=4:raise
        time.sleep(45)


if __name__=='__main__':main(sys.argv[1])
