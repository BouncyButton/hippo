"""Submit the remaining frozen study jobs when this account has a Slurm slot."""
import json, subprocess, time
from pathlib import Path
root=Path('/mnt/beegfsstudents/home/3160552/medsam3_multiclass_replication_20260930')
study=json.loads((root/'study.json').read_text())
record=json.loads((root/'SUBMISSION.json').read_text())
def save():
 tmp=root/'SUBMISSION.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(root/'SUBMISSION.json')
def submit(arguments):
 p=subprocess.run(['sbatch','--parsable',*arguments],capture_output=True,text=True)
 if p.returncode:
  if 'QOSMaxSubmitJobPerUserLimit' in p.stderr:
   return None
  raise RuntimeError(p.stderr)
 job=p.stdout.strip().split(';')[0]
 assert job.isdigit(),job
 return job
def capacity():
 rows=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i'],text=True).splitlines()
 return len(rows)<3
def broadcast():
 with (root/'logs/broadcast_controller.log').open('ab') as log:
  p=subprocess.Popen(['python3',str(root/'source/scripts/medsam3_broadcast_controller.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record['broadcast_controller_pid']=p.pid;save()
try:
 while True:
  missing=[f for f in study['folds'] if str(f) not in record['jobs_by_fold']]
  if missing and capacity():
   fold=missing[0]
   job=submit(['--job-name=medsam3-ap-f'+str(fold),'--output='+str(root/'logs/%j.out'),'--error='+str(root/'logs/%j.err'),str(root/'source/scripts/run_medsam3_multiclass_replication.sh'),str(root),str(fold)])
   if job:
    record['jobs_by_fold'][str(fold)]=job;save();broadcast()
    print('Submitted fold',fold,'job',job,flush=True)
  elif not missing and 'aggregate_job_id' not in record and capacity():
   job=submit(['--dependency=afterok:'+':'.join(record['jobs_by_fold'].values()),'--output='+str(root/'logs/%j.out'),'--error='+str(root/'logs/%j.err'),str(root/'source/scripts/run_medsam3_multiclass_aggregate.sh'),str(root)])
   if job:
    record['aggregate_job_id']=job;record['status']='submitted';save()
    print('Submitted aggregate job',job,flush=True)
  if 'aggregate_job_id' in record:
   break
  record['status']='waiting_for_account_job_slot';save()
  time.sleep(60)
except Exception as exc:
 record['submission_controller_error']=repr(exc);save();raise
