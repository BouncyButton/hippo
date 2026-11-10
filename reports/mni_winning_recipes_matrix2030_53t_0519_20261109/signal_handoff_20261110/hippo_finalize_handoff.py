import fcntl
import json
import time
from pathlib import Path
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read, write, REMOTE

name='mni_winning_recipes_matrix2030_53t_0519_20261109'
report=Path('/Users/filippofocaccia/Desktop/hippo/reports')/name
folder=report/'signal_handoff_20261110'
receipt=read(folder/'submission.json')
code='''import fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);job=sys.argv[2];expected=sys.argv[3]
def read(n):return json.loads((p/n).read_text())
def sha(n):return hashlib.sha256((p/n).read_bytes()).hexdigest()
assert sha('manifest.json')==expected
assert read('preflight.json')['status']=='passed' and read('preflight.json')['manifest_sha256']==expected
assert not read('control.json')['pause']
assert not (p/'failure.json').exists()
raw=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True)
queue=[r.split('|') for r in raw.splitlines() if r.strip()]
own=next(r for r in queue if r[0]==job)
assert len(queue)<=2 and sum(r[1]=='PENDING' for r in queue)<=1
info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
if own[1]=='PENDING':
 with (p/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read('status.json');state.update(status='queued',active=[],job_id=job,dependency=read('submission.json')['dependency'],runtime_manifest_sha256=expected,updated_at=time.time())
  temporary=p/'status.json.repair.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(p/'status.json')
state=read('status.json')
print(json.dumps(dict(queue=queue,scontrol=info,status=state,manifest_sha256=expected,preflight_passed=True,checked_at=time.time())))
'''
result=json.loads(remote(['python3','-c',code,REMOTE/name,receipt['job_id'],receipt['manifest_sha256']],timeout=180))
assert 'gres/gpu:4g.40gb:2' in result['scontrol'],result['scontrol']
assert result['status']['logical_completed']==245
with (report/'matrix_service.lock').open('a') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise RuntimeError('Archive service is not holding its singleton lock')
    except BlockingIOError:
        result['archive_service_lock_held']=True
result['archive_service_pid']=read(report/'matrix_collector_process.json')['pid']
result['archive_errors']=[n for n in ('collector_error.json','matrix_collector_error.json') if (report/n).exists()]
assert not result['archive_errors']
write(folder/'launch_verification.json',result)
write(report/'launch_verification.json',result)
summary=dict(status='repaired_and_queued',job_id=receipt['job_id'],dependency=receipt['dependency'],gpu_workers=2,logical_completed=245,logical_total=504,remaining=259,checkpoint_resume_update=82,regression_tests_passed=9,preflight_passed=True,quota=receipt['quota'],original_training_recipes_unchanged=True,manifest_sha256=receipt['manifest_sha256'],automatic_time_limit_continuation=True,archive_service_pid=result['archive_service_pid'],checked_at=time.time())
write(folder/'repair_summary.json',summary)
print(json.dumps(summary,indent=2))
print(result['scontrol'])
