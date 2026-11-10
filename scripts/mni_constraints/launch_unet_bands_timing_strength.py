"""Deploy pilot, temporarily pause matrix, and restore its original control state."""
import argparse
import fcntl
import json
import subprocess
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints import unet_bands_timing_strength as study
from scripts.mni_constraints.stage_seed5_remote import assemble

STAGE=study.ROOT/'experiments'/study.NAME
REPORT=study.ROOT/'reports'/study.NAME
DEST=REMOTE/study.NAME
PARENT=REMOTE/study.PARENT


def command(stage,module):
    env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    return ['env','PYTHONPATH='+env,read(STAGE/'plan.json')['conventional_python'],'-m',module]


def collector(stage,local_stage,report,module):
    with (report/'collector.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(already_running=True)
        fcntl.flock(lock,fcntl.LOCK_UN)
    with (report/'collector.log').open('a') as log:
        p=subprocess.Popen(['caffeinate','-i',str(study.ROOT/'.venv/bin/python'),'-u','-m',module,
            '--stage',str(stage),'--local-stage',str(local_stage),'--destination',str(report)],
            cwd=study.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=p.pid,started_at=time.time(),module=module)
    write(report/'collector_process.json',receipt)
    return receipt


def deploy():
    study.verify(STAGE);REPORT.mkdir(parents=True,exist_ok=True)
    assert not (REPORT/'submission.json').exists()
    code='''import json,sys,subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);own={f.stem for f in (p/'submissions').glob('*.json')}
q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T'],text=True).strip().splitlines()
assert len(q)<=2 and all(x.split('|')[0] in own for x in q),q
usage=quota();assert usage['free_gib']>=16;print(json.dumps(dict(quota=usage,queue=q)))'''
    env=':'.join(str(PARENT/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    write(REPORT/'initial_cluster_check.json',json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',code,PARENT])))
    remote(['python3','-c','import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)',DEST])
    print(json.dumps(assemble(STAGE,DEST,REPORT,read(STAGE/'deployment_known_donors.json'))),flush=True)
    output=remote(command(DEST,study.MODULE)+['preflight','--stage',DEST],timeout=900)
    write(REPORT/'remote_preflight.json',json.loads(output.strip().splitlines()[-1]))
    print('Remote preflight passed; matrix has not yet been paused.',flush=True)


def pause_and_submit():
    assert read(REPORT/'remote_preflight.json')['status']=='passed'
    assert not (REPORT/'submission.json').exists(),'Already submitted; inspect existing pilot.'
    code='''import json,sys,subprocess,os,time
from pathlib import Path
p=Path(sys.argv[1]);owner=sys.argv[2];read=lambda f:json.loads(f.read_text())
assert not (Path(sys.argv[3])/'submission.json').exists()
old=read(p/'control.json');assert not old.get('pause'),old
own={f.stem for f in (p/'submissions').glob('*.json')}
q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T'],text=True).strip().splitlines()
assert len(q)<=2 and all(x.split('|')[0] in own for x in q),q
record=dict(original_control=old,status_before=read(p/'status.json'),queue_before=q,time=time.time())
receipt=p/('pilot_pause_'+owner+'.json');assert not receipt.exists();receipt.write_text(json.dumps(record))
new=dict(old,pause=True,pilot_pause_owner=owner)
tmp=p/'control.pilot_tmp';tmp.write_text(json.dumps(new));os.replace(tmp,p/'control.json')
for row in q:
 job,state=row.split('|')
 if state=='PENDING':subprocess.run(['scancel',job],check=True)
print(json.dumps(record))'''
    receipt=json.loads(remote(['python3','-c',code,PARENT,study.NAME,DEST]))
    write(REPORT/'matrix_pause_receipt.json',receipt)
    print('Matrix pause requested; pending continuation cancelled; active recoveries retained.',flush=True)
    previous=None
    for _ in range(60):
        code='''import json,sys,subprocess
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'status.json').read_text());q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T'],text=True).strip()
print(json.dumps(dict(status=s['status'],active=s.get('active'),queue=q)))'''
        state=json.loads(remote(['python3','-c',code,PARENT]))
        if state!=previous:print(json.dumps(state),flush=True);previous=state
        if state['status']=='paused' and not state['queue']:break
        time.sleep(5)
    else:raise RuntimeError('Matrix has not safely paused; pilot not submitted.')
    code='''import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]);items=[]
for f in (p/'runs').glob('*/*.pt'):
 items.append(dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print(json.dumps(items))'''
    write(REPORT/'matrix_preserved_checkpoints.json',json.loads(remote(['python3','-c',code,PARENT],timeout=180)))
    collector(DEST,STAGE,REPORT,'scripts.cross_dataset_campaign.collect')
    receipt=json.loads(remote(command(DEST,study.MODULE)+['submit','--stage',DEST],timeout=900))
    write(REPORT/'submission.json',receipt);print(json.dumps(receipt),flush=True)


def resume_matrix():
    assert (REPORT/'pilot_results.json').exists(),'Verify complete local pilot archives first.'
    assert read(REPORT/'pilot_results.json')['status']=='complete'
    assert not (REPORT/'matrix_resume_submission.json').exists(),'Already resumed.'
    code='''import json,sys,subprocess,os
from pathlib import Path
p=Path(sys.argv[1]);pilot=Path(sys.argv[2]);owner=sys.argv[3];read=lambda f:json.loads(f.read_text())
assert read(pilot/'status.json')['status']=='complete'
for task in read(pilot/'plan.json')['tasks']:
 r=pilot/'runs'/task['id'];assert read(r/'cleanup.json')['status']=='complete' and not list(r.glob('*.pt'))
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
c=read(p/'control.json');receipt=read(p/('pilot_pause_'+owner+'.json'))
assert c==dict(receipt['original_control'],pause=True,pilot_pause_owner=owner),'Control changed since pilot pause; do not override.'
failure=p/'failure.json'
if failure.exists():
 import hashlib
 assert sys.argv[4] and hashlib.sha256(failure.read_bytes()).hexdigest()==sys.argv[4]
 failure.rename(p/('pilot_pause_failure_'+owner+'.json'))
tmp=p/'control.pilot_tmp';tmp.write_text(json.dumps(receipt['original_control']));os.replace(tmp,p/'control.json')
# The collector exits on a terminal paused/failed snapshot. Record the actual
# resume request before restarting it, without claiming the job is running yet.
import time
s=read(p/'status.json');(p/('pilot_pause_status_'+owner+'.json')).write_text(json.dumps(s))
s.update(status='resuming_after_pilot',active=[],updated_at=time.time())
tmp=p/'status.pilot_tmp';tmp.write_text(json.dumps(s));os.replace(tmp,p/'status.json')
print(json.dumps(dict(status='restored',control=receipt['original_control'])))'''
    from scripts.cross_dataset_campaign.common import digest
    failure=REPORT/'matrix_failure_preserved.json'
    receipt=json.loads(remote(['python3','-c',code,PARENT,DEST,study.NAME,digest(failure) if failure.exists() else '']))
    write(REPORT/'matrix_control_restored.json',receipt)
    collector(PARENT,study.ROOT/'experiments'/study.PARENT,study.ROOT/'reports'/study.PARENT,'scripts.mni_constraints.collect_verified_matrix')
    receipt=json.loads(remote(command(PARENT,'scripts.mni_constraints.verified_lower_lr_full_matrix')+['submit','--stage',PARENT],timeout=900))
    write(REPORT/'matrix_resume_submission.json',receipt);print(json.dumps(receipt),flush=True)


def submit_verified_recovery():
    """Continue only after separately auditing a stop that was labelled failed."""
    assert read(REPORT/'matrix_recovery_audit.json')['status']=='passed'
    assert not (REPORT/'submission.json').exists()
    expected=read(REPORT/'matrix_recovery_audit.json')['rows']
    code='''import json,sys,hashlib,subprocess
from pathlib import Path
p=Path(sys.argv[1]);owner=sys.argv[2];rows=json.loads(sys.argv[3]);read=lambda f:json.loads(f.read_text())
assert read(p/'control.json')==dict(read(p/('pilot_pause_'+owner+'.json'))['original_control'],pause=True,pilot_pause_owner=owner)
assert read(p/'status.json')['status']=='failed'
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
for row in rows:
 r=p/'runs'/row['id']
 assert hashlib.sha256((r/'checkpoint_latest.pt').read_bytes()).hexdigest()==row['checkpoint_sha256']
 assert hashlib.sha256((r/'budget_ledger.json').read_bytes()).hexdigest()==row['ledger_sha256']
text=(p/'failure.json').read_text()
print(json.dumps(dict(status='verified_stopped_with_recoveries',failure_text=text,failure_sha256=hashlib.sha256(text.encode()).hexdigest())))'''
    receipt=json.loads(remote(['python3','-c',code,PARENT,study.NAME,json.dumps(expected)],timeout=180))
    failure=REPORT/'matrix_failure_preserved.json';failure.write_text(receipt['failure_text'])
    from scripts.cross_dataset_campaign.common import digest
    assert digest(failure)==receipt['failure_sha256']
    write(REPORT/'matrix_verified_stop.json',receipt)
    collector(DEST,STAGE,REPORT,'scripts.cross_dataset_campaign.collect')
    receipt=json.loads(remote(command(DEST,study.MODULE)+['submit','--stage',DEST],timeout=900))
    write(REPORT/'submission.json',receipt);print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['deploy','submit','resume-matrix','submit-verified-recovery'])
    action=parser.parse_args().action
    {'deploy':deploy,'submit':pause_and_submit,'resume-matrix':resume_matrix,'submit-verified-recovery':submit_verified_recovery}[action]()


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:
        print(error.output,flush=True);raise
