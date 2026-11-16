"""Adaptive activation-only search; evaluate three folds before every decision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import time
import uuid
import zipfile
from scripts.colab_hfh.nnunet_confirm13_campaign import validate_result,stage_archive
from scripts.colab_hfh.nnunet_composition_campaign import SDKDrive,check_no_workers,check_prior_attempt,verify_package,retire,zip_archive
from scripts.colab_hfh.nnunet_timing_policy import NAME,CAP,summarize_groups
from scripts.colab_hfh.runtime import read,write,digest


def advance(groups,results,specs):
    """Enqueue only after all folds of the previous timing are archived."""
    queue=[]
    byname={r['name']:r for r in results}
    for key,g in groups.items():
        if g['status']!='searching':continue
        last=g['attempts'][-1]
        if 'run_names' in last:
            last['results']=[byname[n] for n in last['run_names'] if n in byname]
            if len(last['results'])!=3:continue
            last['mean_gain_pp']=mean(r['native_gain_pp'] for r in last['results'])
            if last['mean_gain_pp']>0:
                last['chosen']=True;g['status']='positive';continue
        if not g['remaining_prefixes']:
            g['status']='exhausted_without_positive_mean';continue
        prefix=g['remaining_prefixes'].pop(0)
        names=[f"{g['arm']}__p{prefix:02d}__f{f}_s{g['seed']}" for f in range(3)]
        g['attempts'].append(dict(prefix=prefix,run_names=names,results=[],origin='new_timing_trial'))
        queue.extend(specs[n] for n in names)
    return queue


def archive_one(root,package,spec):
    output,durable=root/'runs'/spec['name']/'run',root/'runs'/spec['name']/'archive'
    row=validate_result(root,package,spec)
    if spec['arm']=='dice':
        reference=root/'native_references'/spec['name']
        for name in ['validation_latest.json','training_final.json']+[f'native_update{n}.json' for n in (20,30,45,60)]:
            if read(output/name)!=read(reference/name):raise ValueError('Native replay differs from original: '+spec['name']+'/'+name)
        for n in range(1,91):
            a=read(output/f'native_update{n}.json')
            if a['after_update']!=n or len(a['model_state_sha256'])!=64:raise ValueError('Missing native timing anchor')
        row['reproduced_previous_native']=True
    else:
        row.update(group=spec['group'],prefix=spec['prefix'])
        prior=next(r for r in read(package/'reference_results.json') if r['arm']==spec['arm'] and (r['fold'],r['seed'])==(spec['fold'],spec['seed']))
        row.update(prior_validation_dice=prior['validation_dice'],gain_vs_prior_recipe_pp=100*(row['validation_dice']-prior['validation_dice']))
    write(output/'comparison.json',row)
    stage_archive(output,durable,package/'protocols'/spec['name'],spec['fold'],spec['seed'])
    drive=SDKDrive('results_'+NAME);zipped=root/(spec['name']+'.zip')
    zip_archive(zipped,durable,package/'protocols'/spec['name'],package);drive.upload(zipped)
    retire(output,durable)
    zip_archive(zipped,durable,package/'protocols'/spec['name'],package);drive.upload(zipped)
    return row,digest(zipped)


def run(root,package,max_workers):
    import psutil
    verify_package(package);design=read(package/'timing_design.json')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_no_workers()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
        safe=min(len(os.sched_getaffinity(0)),int((int(gpu.rsplit(',',1)[1])/1024-8)//4),int((psutil.virtual_memory().available/2**30-16)//8))
        if 'A100' not in gpu or not 1<=max_workers<=safe:raise ValueError('Invalid GPU/worker capacity')
        drive=SDKDrive('results_'+NAME);quota=json.loads(drive.call('about','hfh_storage:','--json'))
        pending_limit=design['max_pending_archives'];peak=(max_workers+pending_limit)*2+8
        if min(quota['free'],shutil.disk_usage(root).free)<peak*2**30:raise ValueError('Insufficient storage headroom')
        write(root/'storage_preflight.json',dict(drive=quota,local_free=shutil.disk_usage(root).free,peak_generated_estimate_gib=peak,max_workers=max_workers,archive_workers=2,gpu=gpu,checked_at=time.time()))
        for rel,sha in read(root/'execution_source_manifest.json').items():
            if digest(root/'bundle/source'/rel)!=sha:raise ValueError('Execution source changed: '+rel)
        previous=read(root/'status.json') if (root/'status.json').exists() else None
        check_prior_attempt(previous,drive.previous_state(root/'previous_remote_status.json'))
        if previous is not None:raise RuntimeError('Existing charged attempt; audit before resume')
        state=dict(run_id=uuid.uuid4().hex,status='verifying_native_replays',results=[],workers=[],archiving=[],
            max_gpu_workers=max_workers,cap=CAP,maximum_new_runs=design['maximum_new_runs'],groups=deepcopy(design['groups']))
        write(root/'status.json',state)
        for name in ('execution_source.zip','launch_provenance.json','storage_preflight.json','reference_archives.zip'):drive.upload(root/name)
        drive.upload(package/'timing_design.json');drive.upload(package/'reference_results.json')
        env=dict(os.environ,PYTHONPATH=':'.join([str(root)]+[str(root/'bundle'/p) for p in ('compatibility','source','source/scripts','source/baselines/nnUNet')]),
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',
            HIPPO_HFH_GPU_CACHE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',HFH_SEARCH_GPU_FRACTION=str(min(.28,.80/max_workers)))
        queue=list(design['native_runs']);active={};pending={};archives={};last=0.
        def publish():
            state['workers']=[]
            for name,(spec,proc,log) in active.items():
                ledger=root/'runs'/name/'run/budget_ledger.json'
                if proc.poll() is None:state['workers'].append(dict(name=name,pid=proc.pid,step=read(ledger)['charged_attempts'] if ledger.exists() else 0))
            summary=summarize_groups(state['groups'])
            state.update(archiving=list(pending),remaining_to_launch=len(queue),updated_at=time.time(),
                positive_groups=sum(r['mean_gain_pp']>0 for r in summary),optimization_flops_spent=sum(r['optimization_flops'] for r in state['results']))
            write(root/'status.json',state);drive.upload(root/'status.json')
            write(root/'summary.json',summary);drive.upload(root/'summary.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                while True:
                    for name,(spec,future) in list(pending.items()):
                        if future.done():
                            row,sha=future.result();archives[name]=sha;state['results'].append(row);del pending[name]
                            print('TIMING_ARCHIVED',name,'gain_pp',round(row['native_gain_pp'],5),flush=True)
                    for name,(spec,proc,log) in list(active.items()):
                        if proc.poll() is None:continue
                        log.close()
                        if proc.returncode:raise RuntimeError('Worker failed; recovery retained; no automatic retry: '+name)
                        if len(pending)<pending_limit:
                            pending[name]=(spec,pool.submit(archive_one,root,package,spec));del active[name]
                    ready=sum(r['arm']=='dice' for r in state['results'])==9
                    if ready:queue.extend(advance(state['groups'],state['results'],design['specs']))
                    while queue and len(active)<max_workers and len(active)+len(pending)<max_workers+pending_limit:
                        spec=queue.pop(0);name=spec['name'];output,durable=root/'runs'/name/'run',root/'runs'/name/'archive'
                        if spec['arm']!='dice' and not ready:raise ValueError('Constraints before verified native controls')
                        output.mkdir(parents=True);durable.mkdir(parents=True)
                        write(output/'controller_attempt.json',dict(run_id=state['run_id'],started_at=time.time()))
                        write(output/'execution_environment.json',read(root/'launch_provenance.json')['hardware'])
                        log=(output/'worker.log').open('a')
                        proc=subprocess.Popen([sys.executable,'-m',design['worker_module'],'--protocol-dir',str(package/'protocols'/name),
                            '--output',str(output),'--durable',str(durable),'--pkl',str(root/'bundle/data_index.json'),'--fold',str(spec['fold']),'--seed',str(spec['seed'])],
                            cwd=root/'bundle/source',env={**env,'HIPPO_DEADLINE':str(time.time()+3600)},stdout=log,stderr=subprocess.STDOUT)
                        active[name]=(spec,proc,log);print('TIMING_START',name,proc.pid,flush=True)
                    state['status']='searching_activation_per_seed' if ready else 'verifying_native_replays'
                    if not queue and not active and not pending:break
                    if time.monotonic()-last>30:publish();last=time.monotonic()
                    time.sleep(1)
                remaining=[str(p) for p in (root/'runs').rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.tar')]
                if remaining:raise ValueError('Completed checkpoints remain')
                summary=summarize_groups(state['groups']);success=all(r['mean_gain_pp']>0 for r in summary)
                state.update(status='complete_all_39_positive' if success else 'complete_feasible_timings_exhausted',storage_after=json.loads(drive.call('about','hfh_storage:','--json')))
                write(root/'verification.json',dict(status='verified_complete',runs=len(state['results']),archives_sha256=archives,
                    remaining_generated_checkpoints=remaining,all_39_positive=success,local_free_gib=shutil.disk_usage(root).free/2**30))
                publish();drive.upload(root/'verification.json');drive.upload(root/'controller.log')
                with zipfile.ZipFile(root/'all_results.zip','w',zipfile.ZIP_DEFLATED) as z:
                    for name in archives:z.write(root/(name+'.zip'),name+'.zip')
                    for name in ('status.json','summary.json','verification.json','launch_provenance.json','storage_preflight.json','execution_source.zip','controller.log','reference_archives.zip'):z.write(root/name,name)
                    z.write(package/'timing_design.json','timing_design.json');z.write(package/'reference_results.json','reference_results.json')
                drive.upload(root/'all_results.zip');print('TIMING_COMPLETE',len(state['results']),'all39positive',success,flush=True)
            except BaseException as error:
                state.update(status='error_no_expansion',error=str(error));print('TIMING_ERROR',str(error),flush=True)
                for name,(spec,proc,log) in list(active.items()):
                    proc.wait();log.close()
                    if proc.returncode==0:pending[name]=(spec,pool.submit(archive_one,root,package,spec))
                    del active[name]
                for name,(spec,future) in list(pending.items()):
                    try:
                        row,sha=future.result();state['results'].append(row);archives[name]=sha;del pending[name]
                    except Exception as failure:print('ARCHIVE_RETAINED',name,str(failure),flush=True)
                publish();drive.upload(root/'controller.log');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--max-workers',type=int,required=True)
    args=parser.parse_args();run(args.root,args.package,args.max_workers)
