"""Submit one bounded inference-only job for missing pre-retirement metrics."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
STAGE = '/mnt/beegfsstudents/home/3160552/storage_retirement_audit_20261014'
worker = (BASE / 'audit_missing_metrics.py').read_bytes()
script = '''#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export nnUNet_compile=false
export MPLCONFIGDIR="/tmp/hippo-storage-audit-${SLURM_JOB_ID}"
stage=/mnt/beegfsstudents/home/3160552/storage_retirement_audit_20261014
for experiment in capped_depth_delay_31976_687658 budget_baselines_94603_684354 budget_bands_94603_684431; do
 /home/3160552/.conda/envs/hippocampus/bin/python "$stage/audit_missing_metrics.py" \\
  --root "/mnt/beegfsstudents/home/3160552/$experiment" \\
  --output "$stage/results" \\
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl
done
'''
(BASE / 'run_metrics_audit.sh').write_text(script)
payload = dict(stage=STAGE, worker=base64.b64encode(worker).decode(), script=script)
remote = '''
import base64, json, os, re, subprocess
from pathlib import Path
stage=Path(PAYLOAD['stage'])
queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j'],text=True)
assert not queue.strip(), 'An account job is already running or queued: '+queue
quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
m=re.search(r'(\\d+(?:\\.\\d+)?)GiB/(\\d+(?:\\.\\d+)?)GiB',quota)
assert m and float(m[2])-float(m[1])>1.0, 'Need 1 GiB reserve for small inference outputs'
stage.mkdir(exist_ok=True)
if (stage/'submission.json').exists():
    raise RuntimeError('Submission receipt already exists; inspect before resubmitting')
(stage/'audit_missing_metrics.py').write_bytes(base64.b64decode(PAYLOAD['worker']))
(stage/'run.sh').write_text(PAYLOAD['script'])
args=['sbatch','--parsable','--job-name=hippo-storage-audit','--partition=stud',
      '--gres=gpu:4g.40gb:1','--cpus-per-task=4','--mem=24G','--time=02:00:00',
      '--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err'),str(stage/'run.sh')]
job=subprocess.check_output(args,cwd=stage,text=True).strip()
receipt=dict(job_id=job,stage=str(stage),queue_before=queue,quota_before=quota,
             purpose='Inference only: preserve missing Dice/FN/FP/A-P-swap metrics and hard predictions before approved deletion',
             expected_endpoints=100,estimated_peak_output_gib=0.5,command=args)
(stage/'submission.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
code = 'PAYLOAD = ' + repr(payload) + '\n' + remote
out = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster','python3 -'],input=code.encode())
receipt = json.loads(out)
receipt['local_worker_sha256'] = hashlib.sha256(worker).hexdigest()
(BASE/'evaluation_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
