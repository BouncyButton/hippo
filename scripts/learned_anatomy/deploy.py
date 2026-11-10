"""Read-only reference extraction and quota-aware idempotent job submission."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[3]
    p=json.loads((root/'protocol.json').read_text())
    assert os.getuid()==1395 and root.name==p['name']
    assert root.name in ('learned_anatomy_fold0_seed0_5pct_20261008_01','learned_anatomy_fold0_seed0_5pct_20261008_02')
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'submission.json').exists():print((root/'submission.json').read_text());return
        assert not (root/'submission_attempt.json').exists(), 'Unresolved submission attempt'
        for name,expected in json.loads((root/'payload_manifest.json').read_text()).items():
            assert sha(root/name)==expected,name
        checks=json.loads((root/'local_checks.json').read_text())
        assert checks['passed'] and checks['source_sha256']==p['source_sha256']
        assert sha(p['pkl'])==p['pkl_sha256']
        if p.get('exploratory_audit_acceptance'):
            manifest=json.loads((root/'roi_manifest.json').read_text())
            assert manifest['train_only'] and set(manifest['files'])=={n+'.npz' for n in p['splits']['5pct']['train']}
            assert all(sha(root/'roi'/n)==h for n,h in manifest['files'].items())
        else:
            atlas=Path(p['guide_atlas'])
            manifest=json.loads((atlas/'manifest.json').read_text())
            assert manifest['complete'] and manifest['protocol_sha256']==p['atlas_protocol_sha256']
            (root/'roi').mkdir(exist_ok=True)
            refs={}
            for name in p['splits']['5pct']['train']:
                src=atlas/(name+'.npz'); assert sha(src)==manifest['artifacts'][src.name]
                with np.load(src,allow_pickle=False) as a:
                    roi=np.zeros(64**3,np.uint8);roi[a['roi_indices'].astype(np.int64)]=1
                with (root/'roi'/src.name).open('wb') as f:np.savez_compressed(f,roi=roi.reshape(64,64,64))
                refs[src.name]=sha(root/'roi'/src.name)
            (root/'roi_manifest.json').write_text(json.dumps(dict(files=refs,source_atlas=str(atlas),
                source_manifest_sha256=sha(atlas/'manifest.json'),train_only=True,guide_training='Previously frozen out-of-fold guides'),indent=2)+'\n')
        quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        assert match and float(match[2])-float(match[1])>=p['minimum_free_gib'],quota
        queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i|%T|%j'],text=True)
        jobs=[line.split('|') for line in queue.splitlines() if line.strip()]
        assert len(jobs)<=1 and all(j[1]=='RUNNING' for j in jobs),queue
        cmd=['sbatch','--parsable',f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
        if jobs:cmd.append('--dependency=afterany:'+jobs[0][0])
        cmd.extend([str(root/'source/scripts/learned_anatomy/run.sh'),str(root)])
        request=dict(command=cmd,requested_unix=time.time(),quota=quota,queue=queue,protocol_sha256=p['identity_sha'],
            main_runs=1,encoder_train_cases=10,gradient_gate_required=not bool(p.get('exploratory_audit_acceptance')),
            exploratory_audit_acceptance=p.get('exploratory_audit_acceptance'),root=str(root))
        attempt=root/'submission_attempt.json';attempt.write_text(json.dumps(request,indent=2)+'\n')
        with attempt.open('rb') as f:os.fsync(f.fileno())
        result=subprocess.run(cmd,capture_output=True,text=True,cwd=root/'source')
        request.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        (root/'submission_response.json').write_text(json.dumps(request,indent=2)+'\n')
        job=result.stdout.strip().split(';')[0]
        assert result.returncode==0 and job.isdigit(),request
        request['job_id']=job
        (root/'submission.json').write_text(json.dumps(request,indent=2)+'\n')
        print(json.dumps(request,indent=2))


if __name__=='__main__':main()
