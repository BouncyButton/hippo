from pathlib import Path
import datetime,hashlib,json,subprocess,sys
root=Path('/mnt/beegfsstudents/home/3160552/pcgrad_50cases_seed0_20260929_01')
assert not (root/'SUBMISSION.json').exists(), 'Already submitted'
assert hashlib.sha256((root/'source/PAYLOAD.json').read_bytes()).hexdigest()=='4f79ea06a173de27073ac17330551a75c890c3e582d280b2a7fb7985e56c42a8'
assert hashlib.sha256((root/'run.sh').read_bytes()).hexdigest()=='52a6fa76410c7df2b8a81a20eae40c81d766df81c04c71c0e1824ad2a2c6bf0b'
preflight=json.loads((root/'PREFLIGHT.json').read_text())
assert preflight['status']=='passed' and preflight['training_cases']==50 and preflight['validation_cases']==52
assert preflight['seed']==preflight['fold']==0 and not preflight['training']
assert '2 passed' in (root/'CPU_TESTS.txt').read_text()
sys.path[:0]=[str(root/'source/scripts'),str(root/'source')]
from train_separated_edge import verify_payload
from run_degree_bands_ab import check_quota
verify_payload()
quota=check_quota(root,'pre_submit',1.6)
(root/'logs').mkdir(exist_ok=True)
command=['sbatch','--parsable','--output='+str(root/'logs/%j.out'),'--error='+str(root/'logs/%j.err'),str(root/'run.sh'),str(root)]
job=subprocess.check_output(command,text=True).strip().split(';')[0]
assert job.isdigit()
record=dict(job_id=job,status='submitted',fold=0,seeds=[0],arms=['pcgrad'],new_models=1,training_cases=50,validation_cases=52,epoch_cap=75,early_stopping=dict(min_epochs=60,patience=8,min_delta=.0005),lr_policy=dict(name='ReduceLROnPlateau',initial_lr=1e-4,factor=.5,patience=3,min_lr=1e-6),quota_free_gib=quota['free_bytes']/1024**3,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
(root/'SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
