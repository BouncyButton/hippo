"""Run on the login node after the approved archive cleanup, exactly once."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys

root=Path('/mnt/beegfsstudents/home/3160552/separated_edge_20260929_01')
assert not (root/'SUBMISSION.json').exists(), 'A submission receipt already exists; do not duplicate the job.'
assert json.loads((root/'ARCHIVE_CLEANUP.json').read_text())['status']=='archived_and_removed'
assert '15 passed' in (root/'PREFLIGHT_TESTS.txt').read_text()
sys.path[:0]=[str(root/'source'),str(root/'source/scripts')]
from train_separated_edge import verify_payload
from run_degree_bands_ab import check_quota
verify_payload()
expected={'source/PAYLOAD.json':'0aeeb1dc6f6fad33fa61c9eeb61a06d4afa6404abfc10c5a85ca19902bb041f9',
          'original_coherence.json':'ef4b2cf6ecaf95b313ec5419bee4077c6abaceb2f858ae279d7759d749291fa7',
          'PROTOCOL.md':'9366e4e1ca9e24a181134063e9739f28a5b436a964cc45fe3158ab96fa499824'}
for name,value in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
quota=check_quota(root,'pre_submit',1.6)
(root/'logs').mkdir(exist_ok=True)
command=['sbatch','--parsable','--dependency=afterany:675308',
    '--output='+str(root/'logs/%j.out'),'--error='+str(root/'logs/%j.err'),
    str(root/'source/scripts/run_separated_edge.sh'),str(root)]
job=subprocess.check_output(command,text=True).strip().split(';')[0]
assert job.isdigit(),job
result=dict(status='submitted',job_id=job,dependency='afterany:675308',
    folds=[0],seeds=[0,1,2],arms=['pooled','separated'],new_training_runs=6,
    epoch_cap=75,automatic_fold_expansion=False,quota_free_gib=quota['free_bytes']/1024**3,
    submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,hashes=expected)
(root/'SUBMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
