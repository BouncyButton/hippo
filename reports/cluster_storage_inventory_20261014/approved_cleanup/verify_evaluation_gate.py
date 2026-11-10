"""Verify all required supplemental evaluations in the local archive."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
retention=BASE/'evaluation_retention'
verification=json.loads((retention/'results_verification.json').read_text())
assert verification['status']=='verified' and len(verification['archives'])==1
row=verification['archives'][0];archive=Path(row['archive'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive_sha256']
manifest=json.loads((retention/'results_manifests/storage_retirement_audit_20261014.json').read_text())
metrics={};completions={};jobs={}
with tarfile.open(archive,'r|gz') as tar:
 for member in tar:
  if member.isfile() and member.name.endswith(('voxel_metrics.json','completion.json')):
   value=json.load(tar.extractfile(member))
   if member.name.endswith('voxel_metrics.json'):metrics[member.name]=value
   else:completions[member.name]=value
assert len(metrics)==100
assert sorted(d['endpoints'] for d in completions.values())==[6,6,88]
for path,d in metrics.items():
 assert d['status']=='verified' and d['maximum_case_dice_difference']<2e-5
 prediction=str(Path(path).with_name('validation_predictions.npz'))
 assert manifest['files'][prediction]['sha256']==d['prediction_sha256']
 assert all(k in d['summary'] for k in ('macro_hard_dice','foreground_false_negative','foreground_false_positive','anterior_to_posterior','posterior_to_anterior','ap_swaps'))
for d in completions.values():
 assert d['status']=='verified'
 assert all('storage_retirement_audit_20261014/results/'+p in metrics for p in d['outputs'])
geo=json.loads((BASE/'geofwi3d_voxel_metrics.json').read_text())
assert geo['status']=='verified' and len(geo['runs'])==9
record=dict(status='verified',hippocampus_endpoints=100,geofwi3d_runs=9,
 evaluation_archive=str(archive.resolve()),evaluation_archive_sha256=row['archive_sha256'],
 maximum_case_dice_difference=max(d['maximum_case_dice_difference'] for d in metrics.values()),
 endpoint_summaries=[dict(path=p,endpoint=d['endpoint'],summary=d['summary'],node=d.get('node'),checkpoint_sha256=d['checkpoint_sha256']) for p,d in sorted(metrics.items())])
(BASE/'evaluation_gate.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='endpoint_summaries'}),flush=True)
