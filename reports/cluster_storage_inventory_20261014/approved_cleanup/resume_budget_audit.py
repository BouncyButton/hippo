"""Resume only the two budget diagnostics with their frozen U-Net factory."""
import base64,json,subprocess,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
payload={name:base64.b64encode((BASE/name).read_bytes()).decode() for name in ('audit_missing_metrics.py','audit_missing_metrics_v3.py')}
remote=r"""
import base64,json,os,subprocess
from pathlib import Path
stage=Path('/mnt/beegfsstudents/home/3160552/storage_retirement_audit_20261014')
queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j'],text=True)
assert not queue.strip(),queue
previous=subprocess.check_output(['sacct','-j','700218','--format=State,ExitCode','-n','-X','-P'],text=True).strip()
assert previous.startswith('FAILED|'),previous
receipt=stage/'budget_architecture_submission.json'
assert not receipt.exists()
for name,data in PAYLOAD.items():(stage/name).write_bytes(base64.b64decode(data))
old=(stage/'run.sh').read_text()
(stage/'run_v3.sh').write_text(old)
(stage/'run.sh').write_text(old.replace('capped_depth_delay_31976_687658 budget_baselines_94603_684354','budget_baselines_94603_684354'))
args=json.loads((stage/'submission.json').read_text())['command']
args.insert(-1,'--nodelist=gnode02')
job=subprocess.check_output(args,cwd=stage,text=True).strip()
record=dict(job_id=job,previous_job='700218',previous_state=previous,reason='Use original SHA-verified width-48 linear U-Net factory and original budget node gnode02. Resume verified outputs; no training.',command=args)
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
"""
code='PAYLOAD='+repr(payload)+'\n'+remote
result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster','python3 -'],input=code.encode())
record=json.loads(result);record['local_worker_sha256']=hashlib.sha256((BASE/'audit_missing_metrics.py').read_bytes()).hexdigest()
(BASE/'evaluation_budget_architecture_submission.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
