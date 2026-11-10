"""Deploy confirmation and preserve the temporarily interrupted full matrix."""
import argparse
import json
import subprocess

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.launch_unetr_subfields import remote
from scripts.mni_constraints import unet_delayed10_confirm as study
from scripts.mni_constraints import launch_unet_bands_timing_strength as launch

STAGE = study.ROOT/'experiments'/study.NAME
REPORT = study.ROOT/'reports'/study.NAME
DEST, PARENT = REMOTE/study.NAME, REMOTE/study.PARENT
launch.study, launch.STAGE, launch.REPORT, launch.DEST, launch.PARENT = study, STAGE, REPORT, DEST, PARENT


def pause():
    assert read(REPORT/'remote_preflight.json')['status'] == 'passed'
    assert not (REPORT/'submission.json').exists()
    code = '''import json,sys,subprocess,os,time
from pathlib import Path
p=Path(sys.argv[1]);owner=sys.argv[2];read=lambda f:json.loads(f.read_text())
receipt=p/('pilot_pause_'+owner+'.json')
if receipt.exists():
 assert read(p/'control.json')==dict(read(receipt)['original_control'],pause=True,pilot_pause_owner=owner)
 print(receipt.read_text());sys.exit(0)
old=read(p/'control.json');assert not old.get('pause'),old
own={f.stem for f in (p/'submissions').glob('*.json')}
q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T'],text=True).strip().splitlines()
assert len(q)<=2 and all(x.split('|')[0] in own for x in q),q
record=dict(original_control=old,status_before=read(p/'status.json'),queue_before=q,time=time.time())
receipt.write_text(json.dumps(record))
tmp=p/'control.confirm_tmp';tmp.write_text(json.dumps(dict(old,pause=True,pilot_pause_owner=owner)));os.replace(tmp,p/'control.json')
for row in q:
 job,state=row.split('|')
 if state=='PENDING':subprocess.run(['scancel',job],check=True)
print(json.dumps(record))'''
    record = json.loads(remote(['python3', '-c', code, PARENT, study.NAME]))
    write(REPORT/'matrix_pause_receipt.json', record)
    print('Pause requested; queued continuation cancelled; checkpoints retained.', flush=True)


def audit():
    code = '''import json,sys,subprocess,hashlib,time
from pathlib import Path
p=Path(sys.argv[1]);owner=sys.argv[2];read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip(),'Wait for allocation to exit.'
assert read(p/'control.json')==dict(read(p/('pilot_pause_'+owner+'.json'))['original_control'],pause=True,pilot_pause_owner=owner)
s=read(p/'status.json');assert s['status'] in ('paused','failed'),s['status']
import torch
rows=[];unstarted=[]
for task in read(p/'plan.json')['tasks']:
 r=p/'runs'/task['id']
 if (r/'completion.json').exists():continue
 if not (r/'budget_ledger.json').exists():
  assert not (r/'checkpoint_latest.pt').exists();continue
 ledger=read(r/'budget_ledger.json');cfg=read(r/'config.json')
 assert ledger['config_sha256']==sha(r/'config.json')
 checkpoint=r/'checkpoint_latest.pt'
 if not checkpoint.exists():
  assert ledger['charged_attempts']==0 and ledger['training_flops']==0
  unstarted.append(task['id']);continue
 cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert cp['configuration']==cfg
 assert cp['step']==cp['charged_attempts']==ledger['charged_attempts']
 assert all(k in cp for k in ('model','optimizer','epoch','permutation','cursor','torch_rng','cuda_rng','augmentation_rng'))
 assert 0<=cp['cursor']<=len(cp['permutation'])
 assert ledger['training_flops']<=cfg['budget_flops']
 if (r/'calibration.json').exists():
  assert read(r/'calibration.json')['config_sha256']==sha(r/'config.json')
  if 'calibration_sha256' in ledger:assert ledger['calibration_sha256']==sha(r/'calibration.json')
 rows.append(dict(id=task['id'],step=cp['step'],charged_attempts=ledger['charged_attempts'],checkpoint_sha256=sha(checkpoint),ledger_sha256=sha(r/'budget_ledger.json'),config_sha256=sha(r/'config.json')))
 del cp
failure=(p/'failure.json').read_text() if (p/'failure.json').exists() else None
print(json.dumps(dict(status='passed',matrix_status=s['status'],rows=rows,unstarted=unstarted,failure_text=failure,checked_at=time.time())))'''
    result = json.loads(remote([read(STAGE/'plan.json')['conventional_python'], '-c', code, PARENT, study.NAME], timeout=240))
    failure = result.pop('failure_text')
    if failure is not None:
        (REPORT/'matrix_failure_preserved.json').write_text(failure)
        result['preserved_failure_sha256'] = digest(REPORT/'matrix_failure_preserved.json')
    write(REPORT/'matrix_recovery_audit.json', result)
    print(json.dumps(result), flush=True)


def submit():
    assert read(REPORT/'remote_preflight.json')['status'] == 'passed'
    audit_result = read(REPORT/'matrix_recovery_audit.json')
    assert audit_result['status'] == 'passed' and not (REPORT/'submission.json').exists()
    code = '''import json,sys,hashlib,subprocess
from pathlib import Path
p=Path(sys.argv[1]);owner=sys.argv[2];audit=json.loads(sys.argv[3]);read=lambda f:json.loads(f.read_text())
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
assert read(p/'control.json')==dict(read(p/('pilot_pause_'+owner+'.json'))['original_control'],pause=True,pilot_pause_owner=owner)
assert read(p/'status.json')['status']==audit['matrix_status']
for row in audit['rows']:
 r=p/'runs'/row['id']
 for file,key in [('checkpoint_latest.pt','checkpoint_sha256'),('budget_ledger.json','ledger_sha256'),('config.json','config_sha256')]:
  assert hashlib.sha256((r/file).read_bytes()).hexdigest()==row[key]
if 'preserved_failure_sha256' in audit:assert hashlib.sha256((p/'failure.json').read_bytes()).hexdigest()==audit['preserved_failure_sha256']
print('Verified stopped matrix recovery states.')'''
    print(remote(['python3', '-c', code, PARENT, study.NAME, json.dumps(audit_result)], timeout=180), flush=True)
    launch.collector(DEST, STAGE, REPORT, 'scripts.cross_dataset_campaign.collect')
    receipt = json.loads(remote(launch.command(DEST, study.MODULE)+['submit', '--stage', DEST], timeout=900))
    write(REPORT/'submission.json', receipt);print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['deploy', 'pause', 'audit', 'submit', 'resume-matrix'])
    action = parser.parse_args().action
    {'deploy': launch.deploy, 'pause': pause, 'audit': audit, 'submit': submit,
     'resume-matrix': launch.resume_matrix}[action]()


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.output, flush=True)
        raise
