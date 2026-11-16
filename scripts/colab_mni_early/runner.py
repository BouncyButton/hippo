"""Bounded Colab controller; original MNI training, verified SDK archives."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parent
CAP=53305109592450


def read(path): return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def worker(task):
    import torch
    from types import SimpleNamespace
    from scripts.mni_constraints.campaign import train
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MNI_GPU_FRACTION','.9')))
    output=ROOT/'runs'/task['id'];output.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(model='nnunet',objective=task['arm'],fold=0,seed=0,
        pkl=ROOT/'data/MNI.pkl',protocol_dir=ROOT/'protocols'/task['id'],output=output,
        device='cuda',threads=4,resume=True,smoke_steps=0)
    try:
        train(args)
        write(output/'resource_usage.json',dict(peak_gpu_allocated=torch.cuda.max_memory_allocated(),
              peak_gpu_reserved=torch.cuda.max_memory_reserved(),threads=torch.get_num_threads(),
              gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    except Exception:
        write(output/'failure.json',dict(traceback=traceback.format_exc()));raise


def archive(task,drive):
    from scripts.cross_dataset_campaign.storage import prepare,verify_archive
    output=ROOT/'runs'/task['id']
    for file in (ROOT/'protocols'/task['id']).glob('*.json'):
        shutil.copy2(file,output/('frozen_'+file.name))
    manifest=prepare(output,task);verify_archive(output,manifest)
    done=read(output/'completion.json');config=read(output/'config.json')
    assert done['budget_flops']==CAP and done['updates']==config['schedule']['max_steps']
    assert done['consumed_training_flops']==config['schedule']['consumed_training_flops']
    assert config['initial_lr']==.02 and config['lr_horizon_updates']==200
    durable=ROOT/'reports'/task['id'];durable.mkdir(parents=True,exist_ok=True)
    for file in [*manifest['metadata'],'archive_manifest.json']:
        target=durable/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(output/file,target)
    verify_archive(durable,manifest)
    zipped=ROOT/'reports'/(task['id']+'.zip')
    with zipfile.ZipFile(zipped,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(durable.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(durable)))
    with zipfile.ZipFile(zipped) as z:
        assert z.testzip() is None
        for name,info in manifest['metadata'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==info['sha256']
    remote='mni_storage:results/'+zipped.name
    drive.upload(zipped,remote)  # Upload verifies remote MD5 and size before returning.
    deleted=[]
    for item in manifest['deletion_candidates']:
        path=output/item['name'];assert path.is_file() and digest(path)==item['sha256']
        path.unlink();deleted.append(item)
    receipt=dict(status='verified_and_retired',remote=remote,archive_sha256=digest(zipped),deleted=deleted)
    write(output/'cleanup.json',receipt);drive.upload(output/'cleanup.json','mni_storage:results/'+task['id']+'_cleanup.json')
    validation=read(output/'validation_latest.json')['summary']['macro_hard_dice']
    return dict(**task,validation_dice=validation,training_dice=done['final_training_hard_dice'],
                updates=done['updates'],flops=done['consumed_training_flops'],remote=remote,
                resources=read(output/'resource_usage.json'))


def controller():
    import psutil
    import torch
    from storage_sdk import Drive
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,sha in read(ROOT/'manifest.json').items(): assert digest(ROOT/name)==sha,name
    drive=Drive()
    quota=drive.quota
    if 'limit' in quota: assert int(quota['limit'])-int(quota['usage'])>8*1024**3
    assert shutil.disk_usage(ROOT).free>10*1024**3
    plan=read(ROOT/'plan.json');pending=list(plan['tasks']);active={};results=[];failed=[]
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert any(n in torch.cuda.get_device_name() for n in ('A100','L4'))
    drive.upload(ROOT/'plan.json','mni_storage:plan.json')
    capacity=1;phase='measuring_peak_resources';last_publish=0
    while pending or active:
        # Both first runs finish before using their measured worst-case reservation.
        if len(results)>=2 and phase!='parallel':
            peak=max(x['resources']['peak_gpu_reserved'] for x in results)
            total=torch.cuda.get_device_properties(0).total_memory
            capacity=max(1,min(12,len(os.sched_getaffinity(0))//4,
                int((total-3*1024**3)//max(2*1024**3,math.ceil(peak*1.4))),
                int((psutil.virtual_memory().available-8*1024**3)//(5*1024**3))))
            phase='parallel'
        while pending and len(active)<capacity and not failed:
            task=pending.pop(0);output=ROOT/'runs'/task['id'];output.mkdir(parents=True,exist_ok=True)
            if (output/'cleanup.json').exists():
                raise RuntimeError('This controller already completed cells; inspect status before restart.')
            log=(output/'worker.log').open('a')
            env=dict(os.environ,MNI_GPU_FRACTION=str(.9/capacity))
            proc=subprocess.Popen([sys.executable,'-u',str(__file__),'worker','--task',task['id']],
                cwd=ROOT/'source',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[task['id']]=(task,proc,log)
        for identity,(task,proc,log) in list(active.items()):
            if proc.poll() is None:continue
            log.close();del active[identity]
            if proc.returncode:
                failed.append(dict(id=identity,returncode=proc.returncode));continue
            try: results.append(archive(task,drive))
            except Exception:
                failed.append(dict(id=identity,archive_error=traceback.format_exc()))
        baseline=next((x['validation_dice'] for x in results if x['arm']=='dice'),None)
        if baseline is not None:
            for row in results:row['gain_pp']=100*(row['validation_dice']-baseline)
        state=dict(status='failed' if failed else 'running' if pending or active else 'complete',
            updated=time.time(),gpu=torch.cuda.get_device_name(),phase=phase,max_workers=capacity,
            active=[dict(id=k,pid=v[1].pid) for k,v in active.items()],pending=len(pending),
            completed=len(results),total=len(plan['tasks']),results=results,failed=failed)
        write(ROOT/'status.json',state)
        if time.time()-last_publish>60 or not active:
            drive.upload(ROOT/'status.json','mni_storage:status.json');last_publish=time.time()
            print(json.dumps({k:v for k,v in state.items() if k!='results'}),flush=True)
        if failed and not active:break
        time.sleep(5)
    if failed:raise RuntimeError('Pilot stopped on failure; recovery files retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['worker','controller']);parser.add_argument('--task')
    args=parser.parse_args()
    if args.mode=='worker':worker(next(t for t in read(ROOT/'plan.json')['tasks'] if t['id']==args.task))
    else:controller()
