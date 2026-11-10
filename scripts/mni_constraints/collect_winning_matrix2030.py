"""Archive the dynamically selected matrix without making GPU training depend on it."""
import io
import fcntl
import json
from pathlib import Path
import shutil
import tarfile
import time
import traceback

from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE

# Archiving only needs these identities; importing the training module here
# would unnecessarily require PyTorch on the laptop.
NAME='mni_winning_recipes_matrix2030_53t_0519_20261109'
PARENT='mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'

ROOT=Path(__file__).resolve().parents[2]
STAGE=REMOTE/NAME
LOCAL=ROOT/'experiments'/NAME
REPORT=ROOT/'reports'/NAME
BUNDLE=ROOT/'experiments/mni_matrix2030_handoff_bundle_20261109'


def materialize():
    code='''import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not (p/'manifest.json').exists():print('null')
else:print(json.dumps(dict(manifest_text=(p/'manifest.json').read_text(),reused=json.loads((p/'reused_runs.json').read_text()),control=json.loads((p/'control.json').read_text()))))'''
    value=json.loads(collect.remote(['python3','-c',code,STAGE]))
    if value is None:return False
    manifest=json.loads(value['manifest_text']);origins={e['task']['id']:Path(e['source']) for e in value['reused']}
    missing=[];LOCAL.mkdir(parents=True,exist_ok=True)
    for name,sha in manifest.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe frozen member')
        target=LOCAL/relative
        if target.exists() and digest(target)==sha:continue
        candidates=[ROOT/'experiments'/PARENT/relative]
        if relative.parts[:1]==('source',):candidates.append(BUNDLE/relative.name)
        if relative.parts[:1]==('reused_runs',):
            origin=origins[relative.parts[1]].relative_to(REMOTE)
            tail=Path(*relative.parts[2:])
            candidates.extend((ROOT/'reports'/origin/tail,ROOT/'experiments'/origin/tail,
                ROOT/'reports'/origin.parts[0]/'frozen'/Path(*origin.parts[1:])/tail))
        donor=next((f for f in candidates if f.is_file() and digest(f)==sha),None)
        if donor:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor,target)
        else:missing.append(name)
    if missing:
        payload=collect.remote(['tar','-czf','-','-C',STAGE,*missing])
        with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
            assert set(archive.getnames())==set(missing) and all(m.isfile() for m in archive.getmembers())
            archive.extractall(LOCAL,filter='data')
    for name,sha in manifest.items():assert digest(LOCAL/name)==sha,name
    (LOCAL/'manifest.json').write_text(value['manifest_text']);write(LOCAL/'control.json',value['control'])
    write(REPORT/'materialization.json',dict(verified_at=time.time(),manifest_sha256=digest(LOCAL/'manifest.json'),files=len(manifest),downloaded_files=len(missing)))
    return True


def finish_archive():
    code='''import fcntl,json,sys,time
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write
from scripts.cross_dataset_campaign.storage import retire,quota
from scripts.mni_constraints.winning_matrix_runner import retire_recovery
p=Path(sys.argv[1]);cleaned=[]
with (p/'worker.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for d in (p/'runs').iterdir():
  if (d/'completion.json').exists() and (d/'archive_acknowledgment.json').exists() and read(d/'completion.json')['status']=='complete':
   if not (d/'cleanup.json').exists() or read(d/'cleanup.json')['status']!='complete':retire(d)
   cleaned.append(d.name)
 retire_recovery(p)
 state=read(p/'status.json')
 if state['status']=='complete_pending_archive' and len(cleaned)==read(p/'plan.json')['num_runs']:
  state.update(status='complete',completed=len(cleaned),pending_durable_archive=[],updated_at=time.time());write(p/'status.json',state)
 print(json.dumps(dict(status=state['status'],cleaned=cleaned,quota=quota())))'''
    env=':'.join(str(STAGE/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    runtime=read(LOCAL/'plan.json')['conventional_python']
    result=json.loads(collect.remote(['env','PYTHONPATH='+env,runtime,'-c',code,STAGE]))
    write(REPORT/'final_archive_cleanup.json',result)


def serve():
    REPORT.mkdir(parents=True,exist_ok=True)
    ready=(LOCAL/'manifest.json').exists()
    while True:
        try:
            if not ready:ready=materialize()
            if ready:
                collect.serve(STAGE,LOCAL,REPORT,once=True)
                status=read(REPORT/'live.json').get('status.json',{}).get('status')
                if status in ('complete','complete_pending_archive','failed','paused'):
                    finish_archive();collect.serve(STAGE,LOCAL,REPORT,once=True);return
        except Exception:
            write(REPORT/'matrix_collector_error.json',dict(time=time.time(),traceback=traceback.format_exc()))
        time.sleep(20 if ready else 60)


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'matrix_service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        serve()


if __name__=='__main__':main()
