"""One-time submission after provenance and CPU tests pass."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
root=Path('/mnt/beegfsstudents/home/3160552/pcgrad_edge_20260929_01')
assert not (root/'SUBMISSION.json').exists(),'Already submitted; do not duplicate.'
tests=(root/'CPU_TESTS.txt').read_text()
assert ('8 passed' in tests or '9 passed' in tests) and ' failed' not in tests
for seed in range(3):assert (root/f'fold0/seed{seed}/reference_binding.json').exists()
expected={'source/PAYLOAD.json':'6655b23f385fc46b9c9e0892fa478440601c5ddc84c0f2b3d451500c61536828',
          'PROTOCOL.md':'7e3602d636fa2a8e9557688d446d8c3d26649b9566e7fdaab7becf118ce98aa3'}
for name,value in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
sys.path[:0]=[str(root/'source'),str(root/'source/scripts')]
from train_separated_edge import verify_payload
from run_degree_bands_ab import check_quota
verify_payload()
quota=check_quota(root,'pre_submit',1.6)
(root/'logs').mkdir(exist_ok=True)
command=['sbatch','--parsable','--output='+str(root/'logs/%j.out'),'--error='+str(root/'logs/%j.err'),
         str(root/'source/scripts/run_pcgrad_edge.sh'),str(root)]
job=subprocess.check_output(command,text=True).strip().split(';')[0]
assert job.isdigit(),job
record=dict(status='submitted',job_id=job,folds=[0],seeds=[0,1,2],arms=['sum','pcgrad'],new_models=6,
    epoch_cap=75,automatic_fold_expansion=False,quota_free_gib=quota['free_bytes']/1024**3,
    submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,hashes=expected,
    training_gate='GPU tests pass, then inner/cross conflict in at least 6/10 training cases in every seed.')
(root/'SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
