"""Quota-aware, idempotent submission of the authorized edge-ablation allocation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[3]
    p=json.loads((root/'protocol.json').read_text())
    if os.getuid()!=1395 or root.name!='edge_ablation_fold0_seed0_5pct_20261008_01' or p['name']!=root.name:
        raise RuntimeError('Incorrect staging location')
    receipt=root/'submission.json'
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists():print(receipt.read_text());return
        attempt=root/'submission_attempt.json'
        if attempt.exists():raise RuntimeError('Unresolved prior submission attempt; inspect scheduler')
        for name,expected in json.loads((root/'payload_manifest.json').read_text()).items():
            if sha(root/name)!=expected:raise ValueError('Payload changed: '+name)
        for name,expected in p['source_sha256'].items():
            if sha(root/'source'/name)!=expected:raise ValueError('Source changed: '+name)
        assert p['main_runs']==len(p['runs'])==20 and p['fold']==0 and p['max_epochs']==75
        assert set(p['splits']) == {'5pct'} and all(e['seed']==0 and e['fraction']=='5pct' and e['arm']=='ordinary_swinunetr' for e in p['runs'])
        checks=json.loads((root/'local_checks.json').read_text())
        assert checks['passed'] and checks['protocol_sha256']==sha(root/'protocol.json')
        assert sha(p['pkl'])==p['pkl_sha256']
        quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not match:raise RuntimeError('Cannot parse quota: '+quota)
        free=float(match[2])-float(match[1])
        if free<p['minimum_free_gib']:raise RuntimeError('Insufficient free quota; preserve results before authorized cleanup: '+quota)
        queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i|%T|%j'],text=True).splitlines()
        jobs=[line.split('|') for line in queue if line.strip()]
        if len(jobs)>1 or any(j[1]!='RUNNING' for j in jobs):
            raise RuntimeError('Keep one running allocation and at most one queued allocation: '+repr(queue))
        cmd=['sbatch','--parsable',f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
        deps=[j[0] for j in jobs]
        if deps:cmd.append('--dependency=afterany:'+':'.join(deps))
        cmd.extend([str(root/'source/scripts/edge_ablation/run.sh'),str(root)])
        request=dict(command=cmd,root=str(root),requested_at_unix=time.time(),protocol_sha256=sha(root/'protocol.json'),
             protocol_identity=p['identity_sha'],runs=p['main_runs'],objective='Twenty ordinary SwinUNETR edge ablations against Dice+bands',depth_balancing=False,
             fractions=['5pct'],fold=0,seeds=[0],max_epochs=75,queue_before_submission=queue,
             dependencies=deps,quota_before_submission=quota,free_gib=free,minimum_free_gib=p['minimum_free_gib'],
             cleanup_required=False,fresh_matched_baseline=True,auxiliary_branch=False)
        attempt.write_text(json.dumps(request,indent=2)+'\n')
        with attempt.open('rb') as stream:os.fsync(stream.fileno())
        result=subprocess.run(cmd,cwd=root/'source',capture_output=True,text=True)
        request.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        (root/'submission_response.json').write_text(json.dumps(request,indent=2)+'\n')
        job=result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():raise RuntimeError('Submission not confirmed; inspect response')
        request['job_id']=job
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(request,indent=2)+'\n');tmp.replace(receipt)
        print(json.dumps(request,indent=2))


if __name__=='__main__':main()
