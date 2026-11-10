"""Local durable archive service; the GPU worker waits for verified receipts."""
import argparse
import fcntl
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
import traceback

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, NAME
from scripts.cross_dataset_campaign.storage import verify_archive


def remote(arguments):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','bocconi-cluster',shlex.join([str(v) for v in arguments])],timeout=180)


def upload_json(value, path):
    code='import json,sys,os; from pathlib import Path; p=Path(sys.argv[1]); t=p.with_name(p.name+".tmp"); t.write_text(sys.stdin.read()); os.replace(t,p)'
    subprocess.run(['ssh','bocconi-cluster',shlex.join(['python3','-c',code,str(path)])],input=json.dumps(value).encode(),check=True,timeout=60)


def snapshot(stage):
    code='''import json,sys
from pathlib import Path
p=Path(sys.argv[1]); result={}
for name in ('status.json','failure.json','submission.json','successor.json','base_request.json'):
 f=p/name
 if f.exists(): result[name]=json.loads(f.read_text())
result['ready']=[str(f.parent.relative_to(p)) for f in (p/'runs').glob('*/archive_manifest.json') if not (f.parent/'archive_acknowledgment.json').exists()]
result['cleanup']=[str(f.relative_to(p)) for f in (p/'runs').glob('*/cleanup.json')]
print(json.dumps(result))'''
    return json.loads(remote(['python3','-c',code,stage]))


def collect_one(stage,relative,destination,source_sha):
    output=stage/relative
    manifest_bytes=remote(['cat',output/'archive_manifest.json'])
    manifest=json.loads(manifest_bytes)
    names=list(manifest['metadata'])+['archive_manifest.json']
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe archive member')
    blob=remote(['tar','-czf','-','-C',output,*names])
    target=destination/relative;target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
        if set(archive.getnames())!=set(names) or any(not m.isfile() for m in archive.getmembers()):raise ValueError('Archive inventory differs')
        archive.extractall(target,filter='data')
    if (target/'archive_manifest.json').read_bytes()!=manifest_bytes:raise ValueError('Archive manifest differs')
    verify_archive(target,manifest)
    ack=dict(status='verified',manifest_sha256=digest(target/'archive_manifest.json'),source_verified=True,
             source_manifest_sha256=source_sha,durable_archive=str(target.resolve()),verified_at=time.time())
    write(target/'archive_acknowledgment.json',ack)
    upload_json(ack,output/'archive_acknowledgment.json')
    return manifest['task']['id']


def deliver_base(stage,plan,request):
    job=str(request['job_id']);destination=request['destination']
    if not job.isdigit() or destination!=f'/tmp/hippo-sam3-1395-{job}.pt':raise ValueError('Unexpected checkpoint destination')
    expected=plan['base_sha256']
    if request['expected_sha256']!=expected:raise ValueError('Checkpoint request differs')
    marker=stage/f'base_ready_{job}.json'
    if remote(['test','-f',marker])==b'':return


def resume_after_archive(stage,plan,state):
    """Resume only this opted-in campaign after durable archive has returned."""
    module=plan.get('archive_resume_module')
    if not module or state.get('status.json',{}).get('status') not in ('waiting_for_archive_offline','waiting_for_base_offline'):
        return None
    # Never use manual 'resume' here: it clears pause. Automatic continuation
    # must leave any user pause intact and wait for the old allocation to exit.
    control=json.loads(remote(['cat',stage/'control.json']))
    if control.get('pause'):
        return None
    if control.get('automatic_continuation_owner',str(stage))!=str(stage):
        return None
    code='import sys; from pathlib import Path; p=Path(sys.argv[1]); print(sum(not (f.parent/"archive_acknowledgment.json").exists() for f in (p/"runs").glob("*/archive_manifest.json")))'
    if int(remote(['python3','-c',code,stage]).decode().strip()):
        return None
    for dependency in plan.get('archive_dependency_stages',[]):
        dependency_control=json.loads(remote(['cat',str(Path(dependency)/'control.json')]))
        if dependency_control.get('pause') or int(remote(['python3','-c',code,dependency]).decode().strip()):
            return None
    env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='import importlib,sys; from pathlib import Path; from scripts.cross_dataset_campaign import controller; m=importlib.import_module(sys.argv[1]); p=Path(sys.argv[2]); controller.verify=m.verify; own={f.stem for f in (p/"submissions").glob("*.json")}; active=controller.queue(); print("" if any(row[0] in own for row in active) else controller.submit(p))'
    return remote(['env','PYTHONPATH='+env,'python3','-c',code,module,stage]).decode().strip()


def serve(stage,local_stage,destination,once=False):
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=destination/'frozen'
        if not frozen.exists():shutil.copytree(local_stage,frozen,ignore=shutil.ignore_patterns('runs','*.lock','submissions','slurm-*','status.json','failure.json'))
        manifest=read(frozen/'manifest.json')
        for name,sha in manifest.items():
            if digest(frozen/name)!=sha:raise ValueError('Local source archive changed: '+name)
        source_sha=digest(frozen/'manifest.json');plan=read(frozen/'plan.json')
        delivered=set()
        while True:
            try:
                state=snapshot(stage)
                write(destination/'live.json',state)
                request=state.get('base_request.json')
                if request and request['job_id'] not in delivered:
                    job=str(request['job_id']);target=request['destination']
                    if not job.isdigit() or target!=f'/tmp/hippo-sam3-1395-{job}.pt':raise ValueError('Unsafe checkpoint request')
                    try:
                        active=remote(['squeue','-h','-j',job,'-o','%T']).decode().strip()
                    except subprocess.CalledProcessError as exc:
                        # Slurm can return 1 once a finished job has left squeue.
                        # Its old base request must not block result archiving.
                        if exc.returncode != 1:
                            raise
                        active=''
                    if active=='RUNNING':
                        check=remote(['sha256sum',plan['base_staging']]).decode().split()[0]
                        if check!=plan['base_sha256']:raise ValueError('Staged base checkpoint differs')
                        remote(['sbcast','--jobid='+job,'--force',plan['base_staging'],target])
                        upload_json(dict(job_id=job,sha256=check,delivered_at=time.time()),stage/f'base_ready_{job}.json')
                        delivered.add(job)
                for relative in state['ready']:
                    identity=collect_one(stage,relative,destination,source_sha)
                    print(json.dumps(dict(archived=identity,time=time.time())),flush=True)
                for relative in state['cleanup']:
                    if (destination/relative).exists() and read(destination/relative).get('status')=='complete':continue
                    value=json.loads(remote(['cat',stage/relative]))
                    write(destination/relative,value)
                resumed=resume_after_archive(stage,plan,state)
                if resumed:
                    print(json.dumps(dict(resumed_after_archive=resumed,time=time.time())),flush=True)
                status=state.get('status.json',{}).get('status')
                if once or status in ('complete','failed','paused'):
                    print(json.dumps(dict(status=status,collected_at=time.time())),flush=True)
                    return
            except Exception:
                write(destination/'collector_error.json',dict(traceback=traceback.format_exc(),time=time.time()))
                if once:raise
            time.sleep(10)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,default=REMOTE/NAME)
    parser.add_argument('--local-stage',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    serve(args.stage,args.local_stage,args.destination,args.once)


if __name__=='__main__':main()
