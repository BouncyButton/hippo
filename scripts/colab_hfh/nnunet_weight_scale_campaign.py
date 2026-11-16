"""Fixed-timing weight grid; preserve all completed trials and verified controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean,median
import subprocess
import sys
import time
import uuid
import zipfile
from scripts.colab_hfh import runtime
from scripts.colab_hfh.nnunet_confirm13_campaign import validate_result as validate_shared_result
from scripts.colab_hfh.nnunet_composition_campaign import SDKDrive,check_no_workers,check_prior_attempt,verify_package,retire,zip_archive
from scripts.colab_hfh.nnunet_search13_policy import active_coefficients
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.nnunet_weight_scale_policy import NAME,CAP,summarize
read,write,digest=runtime.read,runtime.write,runtime.digest


def stage_archive(output,durable,protocol,fold,seed):
    source=inspect.getsource(runtime.archive)
    old="schedule = p['models'][p['trainer_alias']]['objectives']['dice']"
    if source.count(old)!=1 or source.count('    deleted = []')!=1:raise ValueError('Archive integration changed')
    source=source.replace(old,"schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    source=source[:source.index('    deleted = []')]+'    return result\n'
    ns=dict(runtime.__dict__);exec(compile(source,__file__,'exec'),ns)
    return ns['archive'](output,durable,'baselines','nnunet',fold,seed,protocol)


def validate_result(root,package,spec):
    row=validate_shared_result(root,package,spec)
    output=root/'runs'/spec['name']/'run'
    p=read(package/'protocols'/spec['name']/'protocol.json')
    original=read(package/'references'/spec['reference_name']/'protocol.json')
    anchor=read(package/'references'/spec['reference_name']/'calibration.json')
    cal=read(output/'calibration.json');m=spec['weight_multiplier'];names=components(spec['arm'])
    if p['models']!=original['models'] or p['activity_policy']!=original['activity_policy'] or p['prefix_anchor_updates']!=original['prefix_anchor_updates']:
        raise ValueError('Timing, update count, or activity envelope changed')
    if cal['anchor_model_sha256']!=anchor['anchor_model_sha256'] or cal['cases']!=anchor['cases']:
        raise ValueError('Matched training calibration inputs changed')
    for name in names:
        if not math.isclose(cal['weights'][name],m*anchor['weights'][name],rel_tol=1e-7,abs_tol=1e-15):
            raise ValueError('Requested coefficient multiplier was not achieved')
    pressure=[];per_component={name:[] for name in names};capped=0
    for line in (output/'constraint_activity.jsonl').read_text().splitlines():
        t=json.loads(line)
        if t['scale']!=1.:continue
        old,stats=active_coefficients(t['gradient_gram'],anchor['weights'],names,p['activity_policy'])
        ratio=t['sum_ratio']/stats['sum_ratio'];pressure.append(ratio)
        if ratio<1.-1e-8:raise ValueError('Increasing anchors lowered total effective pressure on identical gradients')
        capped+=int(t['sum_ratio']>=p['activity_policy']['maximum_sum_ratio']-1e-8)
        for name in names:per_component[name].append(t['effective_weights'][name]/old[name])
    reference=next(r for r in read(package/'weight_design.json')['reference_results'] if r['name']==spec['reference_name'])
    row.update(group=spec['group'],prefix=spec['prefix'],weight_multiplier=m,reference_name=spec['reference_name'],
        reference_1x_dice=reference['validation_dice'],gain_vs_1x_pp=100*(row['validation_dice']-reference['validation_dice']),
        coefficient_multiplier_verified=True,effective_strength=dict(full_strength_updates=len(pressure),
            median_sum_pressure_multiplier_vs_1x_on_same_gradients=median(pressure),minimum_sum_pressure_multiplier=min(pressure),
            capped_updates=capped,median_component_weight_multiplier={n:median(v) for n,v in per_component.items()},
            interpretation='Read-only same-state counterfactual using recorded gradient Gram matrices; no added model optimization'))
    return row


def archive_one(root,package,spec):
    output,durable=root/'runs'/spec['name']/'run',root/'runs'/spec['name']/'archive'
    row=validate_result(root,package,spec);write(output/'comparison.json',row)
    stage_archive(output,durable,package/'protocols'/spec['name'],spec['fold'],spec['seed'])
    drive=SDKDrive('results_'+NAME);zipped=root/(spec['name']+'.zip')
    zip_archive(zipped,durable,package/'protocols'/spec['name'],package);drive.upload(zipped)
    retire(output,durable)
    zip_archive(zipped,durable,package/'protocols'/spec['name'],package);drive.upload(zipped)
    return row,digest(zipped)


def run(root,package,max_workers):
    import psutil
    verify_package(package);design=read(package/'weight_design.json')
    reuse=read(root/'reuse_verification.json')
    if reuse['status']!='verified' or len(reuse['archives_sha256'])!=126:raise ValueError('Unverified matched references')
    if digest(root/'reference_archives.zip')!=reuse['reference_zip_sha256']:raise ValueError('Reference bundle changed')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_no_workers()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
        safe=min(len(os.sched_getaffinity(0)),int((int(gpu.rsplit(',',1)[1])/1024-8)//4),int((psutil.virtual_memory().available/2**30-16)//8))
        if 'A100' not in gpu or not 1<=max_workers<=safe:raise ValueError('Invalid GPU/worker capacity')
        drive=SDKDrive('results_'+NAME)
        quota=json.loads(drive.call('about','hfh_storage:','--json'))
        pending_limit=design['max_pending_archives'];peak=(max_workers+pending_limit)*2+8
        if min(quota['free'],shutil.disk_usage(root).free)<peak*2**30:raise ValueError('Insufficient archive/recovery headroom')
        write(root/'storage_preflight.json',dict(drive=quota,local_free=shutil.disk_usage(root).free,peak_generated_estimate_gib=peak,max_workers=max_workers,archive_workers=2,gpu=gpu,checked_at=time.time()))
        for rel,sha in read(root/'execution_source_manifest.json').items():
            if digest(root/'bundle/source'/rel)!=sha:raise ValueError('Execution source changed: '+rel)
        previous=read(root/'status.json') if (root/'status.json').exists() else None
        check_prior_attempt(previous,drive.previous_state(root/'previous_remote_status.json'))
        if previous is not None:raise RuntimeError('Existing charged attempt: audit before resuming')
        state=dict(run_id=uuid.uuid4().hex,status='starting',results=[],workers=[],archiving=[],planned_total_runs=351,reused_native_controls=9,reused_1x_runs=117,pairs=design['pairs'],max_gpu_workers=max_workers,cap=CAP)
        write(root/'status.json',state)
        for name in ('execution_source.zip','launch_provenance.json','storage_preflight.json','reference_archives.zip','reuse_verification.json'):drive.upload(root/name)
        drive.upload(package/'weight_design.json')
        env=dict(os.environ,PYTHONPATH=':'.join([str(root)]+[str(root/'bundle'/p) for p in ('compatibility','source','source/scripts','source/baselines/nnUNet')]),
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',
            HIPPO_HFH_GPU_CACHE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',HFH_SEARCH_GPU_FRACTION=str(min(.28,.80/max_workers)))
        queue=list(design['runs']);active={};pending={};archives={};last=0.
        def publish():
            state['workers']=[]
            for name,(spec,proc,log) in active.items():
                ledger=root/'runs'/name/'run/budget_ledger.json'
                if proc.poll() is None:state['workers'].append(dict(name=name,pid=proc.pid,step=read(ledger)['charged_attempts'] if ledger.exists() else 0))
            state.update(archiving=list(pending),remaining_to_launch=len(queue),updated_at=time.time())
            summary=summarize(state['results'],design)
            state['improved_groups']=sum(r['improvement_over_1x_pp']>0 for r in summary)
            write(root/'status.json',state);drive.upload(root/'status.json')
            write(root/'summary.json',summary);drive.upload(root/'summary.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                while queue or active or pending:
                    for name,(spec,future) in list(pending.items()):
                        if future.done():
                            row,sha=future.result();archives[name]=sha;state['results'].append(row);del pending[name]
                            print('WEIGHT_ARCHIVED',name,'gain_pp',round(row['native_gain_pp'],5),flush=True)
                    for name,(spec,proc,log) in list(active.items()):
                        if proc.poll() is None:continue
                        log.close()
                        if proc.returncode:raise RuntimeError('Worker failed; recovery retained; no automatic retry: '+name)
                        if len(pending)<pending_limit:
                            pending[name]=(spec,pool.submit(archive_one,root,package,spec));del active[name]
                    # Reuse the unchanged, verified native archives and all activation hashes.
                    ready=all((root/'runs'/f'native__f{f}_s{s}'/'archive/native_update30.json').exists() for f,s in design['pairs'])
                    while queue and len(active)<max_workers and len(active)+len(pending)<max_workers+pending_limit:
                        spec=queue[0]
                        if spec['arm']!='dice' and not ready:break
                        queue.pop(0);name=spec['name'];output,durable=root/'runs'/name/'run',root/'runs'/name/'archive'
                        output.mkdir(parents=True);durable.mkdir(parents=True)
                        write(output/'controller_attempt.json',dict(run_id=state['run_id'],started_at=time.time()))
                        write(output/'execution_environment.json',read(root/'launch_provenance.json')['hardware'])
                        log=(output/'worker.log').open('a')
                        proc=subprocess.Popen([sys.executable,'-m',design['worker_module'],'--protocol-dir',str(package/'protocols'/name),
                            '--output',str(output),'--durable',str(durable),'--pkl',str(root/'bundle/data_index.json'),
                            '--fold',str(spec['fold']),'--seed',str(spec['seed'])],cwd=root/'bundle/source',
                            env={**env,'HIPPO_DEADLINE':str(time.time()+3600)},stdout=log,stderr=subprocess.STDOUT)
                        active[name]=(spec,proc,log);print('WEIGHT_START',name,proc.pid,flush=True)
                    state['status']='running_weight_grid' if ready else 'missing_verified_native_references'
                    if time.monotonic()-last>30:publish();last=time.monotonic()
                    time.sleep(1)
                if len(state['results'])!=351:raise ValueError('Incomplete confirmation')
                remaining=[str(p) for p in (root/'runs').rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.tar')]
                if remaining:raise ValueError('Completed checkpoints remain')
                write(root/'verification.json',dict(status='verified_complete',runs=351,combinations=13,groups=39,multipliers=design['multipliers'],archives_sha256=archives,remaining_generated_checkpoints=remaining,local_free_gib=shutil.disk_usage(root).free/2**30))
                state.update(status='complete_weight_grid',storage_after=json.loads(drive.call('about','hfh_storage:','--json')))
                publish();drive.upload(root/'verification.json');drive.upload(root/'controller.log')
                with zipfile.ZipFile(root/'all_results.zip','w',zipfile.ZIP_DEFLATED) as z:
                    for name in archives:z.write(root/(name+'.zip'),name+'.zip')
                    for name in ('status.json','summary.json','verification.json','launch_provenance.json','storage_preflight.json','execution_source.zip','controller.log','reference_archives.zip','reuse_verification.json'):z.write(root/name,name)
                    z.write(package/'weight_design.json','weight_design.json')
                drive.upload(root/'all_results.zip');print('WEIGHT_COMPLETE',351,flush=True)
            except BaseException as error:
                state.update(status='error_no_expansion',error=str(error));print('WEIGHT_ERROR',str(error),flush=True)
                # Drain already charged workers without rerunning or destroying failures.
                for name,(spec,proc,log) in list(active.items()):
                    proc.wait();log.close()
                    if proc.returncode==0 and (spec['arm']=='dice' or (root/'runs'/spec['native_name']/'archive/cleanup.json').exists()):
                        pending[name]=(spec,pool.submit(archive_one,root,package,spec))
                    del active[name]
                for name,(spec,future) in list(pending.items()):
                    try:
                        row,sha=future.result();state['results'].append(row);archives[name]=sha;del pending[name]
                    except Exception as failure:print('ARCHIVE_RETAINED',name,str(failure),flush=True)
                publish();drive.upload(root/'controller.log');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--max-workers',type=int,required=True)
    args=parser.parse_args();run(args.root,args.package,args.max_workers)
