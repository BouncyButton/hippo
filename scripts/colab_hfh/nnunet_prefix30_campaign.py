"""Fixed update30 timing ablation; reuse verified unchanged controls and recipes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
import time
import uuid
import zipfile

from scripts.colab_hfh import runtime
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.nnunet_search13_campaign import activity_audit, adherence_summary
from scripts.colab_hfh.nnunet_composition_campaign import SDKDrive, check_no_workers, check_prior_attempt, verify_package, retire, zip_archive

NAME='nnunet_prefix30_shared_a100_v1'
CAP=31976450667090
read,write,digest=runtime.read,runtime.write,runtime.digest


def stage_archive(output,durable,protocol,fold,seed):
    source=inspect.getsource(runtime.archive)
    old="schedule = p['models'][p['trainer_alias']]['objectives']['dice']"
    if source.count(old)!=1 or source.count('    deleted = []')!=1:raise ValueError('Archive integration changed')
    source=source.replace(old,"schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    source=source.replace("evaluation_scope='exploratory_validation_selection'","evaluation_scope='frozen_recipe_confirmation'")
    source=source[:source.index('    deleted = []')]+'    return result\n'
    ns=dict(runtime.__dict__);exec(compile(source,__file__,'exec'),ns)
    return ns['archive'](output,durable,'baselines','nnunet',fold,seed,protocol)


def validate_result(root,package,spec):
    name,fold,seed=spec['name'],spec['fold'],spec['seed']
    output=root/'runs'/name/'run'; protocol=package/'protocols'/name
    p,c,done=read(protocol/'protocol.json'),read(output/'config.json'),read(output/'completion.json')
    arm=p['active_objective'];schedule=p['models']['nnunet']['objectives'][arm]
    refs=read(protocol/'baseline_references.json')[f'nnunet_f{fold}_s{seed}']
    original=refs['configuration']
    for key in ('fold','seed','dataset_sha256','train_cases','validation_cases','precision','initial_lr',
                'lr_horizon_updates','nnunet_warmup_fraction','nnunet_sgd_momentum','augmentation','native_loss'):
        if c[key]!=original[key]:raise ValueError('Baseline recipe mismatch: '+key)
    if read(output/'initialization.json')!=refs['initialization']:raise ValueError('Initialization mismatch')
    used=training_flops(done['updates'],schedule)
    if (p['budget_flops']!=CAP or done['status']!='complete' or done['discarded_attempts'] or
        done['updates']!=spec['updates'] or done['consumed_training_flops']!=used or
        not used<=CAP<training_flops(done['updates']+1,schedule) or
        done['calibration_attempts']!=(0 if arm=='dice' else 1) or done['calibration_flops']!=schedule['calibration_flops']):
        raise ValueError('Budget or attempt mismatch')
    val,train=read(output/'validation_latest.json'),read(output/'training_final.json')
    adherence=adherence_summary(output/'constraint_adherence.json')
    row=dict(name=name,variant=spec['variant'],arm=arm,fold=fold,seed=seed,updates=done['updates'],
        validation_dice=val['summary']['macro_hard_dice'],training_dice=train['summary']['macro_hard_dice'],
        optimization_flops=used,optimization_cap=CAP,calibration_flops=done['calibration_flops'],
        constraint_adherence=adherence,ap_swaps_applicable=False)
    for split,evaluation in [('validation',val),('training',train)]:
        expected=c['validation_cases' if split=='validation' else 'train_cases']
        if len(evaluation['cases'])!=10 or {r['case_name'] for r in evaluation['cases']}!=set(expected):raise ValueError('Evaluation cases differ')
        for error in ('fp','fn'):row[split+'_'+error]=sum(r['foreground_'+error] for r in evaluation['cases'])
    if arm=='dice':
        for step in (20,30,45,60):
            anchor=read(output/f'native_update{step}.json')
            if anchor['after_update']!=step or len(anchor['model_state_sha256'])!=64:raise ValueError('Native prefix missing')
        row['native_gain_pp']=0.
    else:
        native=root/'runs'/spec['native_name']/'archive'
        nc=read(native/'config.json')
        for key in ('fold','seed','train_cases','validation_cases','initial_lr','lr_horizon_updates','native_loss','precision','augmentation'):
            if c[key]!=nc[key]:raise ValueError('Paired native mismatch: '+key)
        cal=read(output/'calibration.json'); prefix=schedule['prefix_updates']
        if cal['anchor_successful_updates']!=prefix or cal['anchor_model_sha256']!=read(native/f'native_update{prefix}.json')['model_state_sha256']:
            raise ValueError('Exact paired native prefix differs')
        row.update(activity=activity_audit(output,p,schedule),recipe=p['search_recipe'],weights=cal['weights'],matched_native_prefix_verified=True)
        baseline=read(native/'validation_latest.json');by_case={r['case_name']:r for r in baseline['cases']}
        deltas=[100*(r['macro_hard_dice']-by_case[r['case_name']]['macro_hard_dice']) for r in val['cases']]
        row.update(native_gain_pp=100*(row['validation_dice']-baseline['summary']['macro_hard_dice']),
            improved_crops=sum(d>0 for d in deltas),worst_crop_pp=min(deltas))
        old=adherence_summary(native/'constraint_adherence.json')
        row['adherence_delta_vs_native']={split:{n:adherence[split][n]-old[split][n] for n in components(arm)} for split in old}
        row['all_included_validation_penalties_improved']=all(d<0 for d in row['adherence_delta_vs_native']['validation'].values())
    return row


def summarize(results,design):
    results=list(results)+design.get('reused_results',[])
    summary=[]
    for arm,recipe in design['selected_recipes'].items():
        rows=[r for r in results if r['arm']==arm]
        entry=dict(arm=arm,frozen_recipe=recipe,completed_pairs=len(rows),expected_pairs=9)
        for label,group in [('other_folds',[r for r in rows if r['fold']!=0]),('same_fold_new_seeds',[r for r in rows if r['fold']==0 and r['seed']!=0]),('selection_pair',[r for r in rows if r['fold']==0 and r['seed']==0]),('all_confirmation',[r for r in rows if (r['fold'],r['seed'])!=(0,0)]),('all_nine',rows)]:
            gains=[r['native_gain_pp'] for r in group]
            entry[label]=dict(n=len(gains),mean_gain_pp=mean(gains) if gains else None,
                sd_gain_pp=stdev(gains) if len(gains)>1 else None,min_gain_pp=min(gains) if gains else None,
                max_gain_pp=max(gains) if gains else None,positive_pairs=sum(g>0 for g in gains),
                all_penalties_improved_pairs=sum(r['all_included_validation_penalties_improved'] for r in group),
                pairs=[dict(fold=r['fold'],seed=r['seed'],gain_pp=r['native_gain_pp'],dice=r['validation_dice']) for r in group])
        summary.append(entry)
    return summary


def archive_one(root,package,spec):
    output,durable=root/'runs'/spec['name']/'run',root/'runs'/spec['name']/'archive'
    row=validate_result(root,package,spec)
    prior=read(root/'previous_results.json')[spec['name']]
    row.update(prior_prefix=20,activation_prefix=30,prior_validation_dice=prior['validation_dice'],
        gain_vs_prior_recipe_pp=100*(row['validation_dice']-prior['validation_dice']))
    write(output/'comparison.json',row)
    stage_archive(output,durable,package/'protocols'/spec['name'],spec['fold'],spec['seed'])
    drive=SDKDrive('results_'+NAME)  # One separate HTTP client per archive task.
    zipped=root/(spec['name']+'.zip')
    zip_archive(zipped,durable,package/'protocols'/spec['name'],package);drive.upload(zipped)
    retire(output,durable)
    zip_archive(zipped,durable,package/'protocols'/spec['name'],package);drive.upload(zipped)
    return row,digest(zipped)


def run(root,package,max_workers):
    import psutil
    verify_package(package);design=read(package/'confirmation_design.json')
    design['reused_results']=read(root/'reused_results.json')
    receipt=read(root/'reuse_verification.json')
    if receipt['status']!='verified' or len(design['reused_results'])!=63:raise ValueError('Unverified reused references')
    for name,sha in receipt['archive_sha256'].items():
        if digest(root/'reused_archives'/name)!=sha:raise ValueError('Reused archive changed')
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
        state=dict(run_id=uuid.uuid4().hex,status='starting',results=[],workers=[],archiving=[],planned_total_runs=63,reused_runs=63,full_matrix_runs=126,pairs=design['pairs'],max_gpu_workers=max_workers,cap=CAP)
        write(root/'status.json',state)
        for name in ('execution_source.zip','launch_provenance.json','storage_preflight.json','reused_references.zip','reuse_verification.json','reused_results.json','previous_results.json'):drive.upload(root/name)
        drive.upload(package/'confirmation_design.json');drive.upload(package/'selected_recipes.json')
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
            write(root/'status.json',state);drive.upload(root/'status.json')
            write(root/'summary.json',summarize(state['results'],design));drive.upload(root/'summary.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                while queue or active or pending:
                    for name,(spec,future) in list(pending.items()):
                        if future.done():
                            row,sha=future.result();archives[name]=sha;state['results'].append(row);del pending[name]
                            print('CONFIRM_ARCHIVED',name,'gain_pp',round(row['native_gain_pp'],5),flush=True)
                    for name,(spec,proc,log) in list(active.items()):
                        if proc.poll() is None:continue
                        log.close()
                        if proc.returncode:raise RuntimeError('Worker failed; recovery retained; no automatic retry: '+name)
                        if len(pending)<pending_limit:
                            pending[name]=(spec,pool.submit(archive_one,root,package,spec));del active[name]
                    # Reused native controls were independently checked before any launch.
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
                        active[name]=(spec,proc,log);print('CONFIRM_START',name,proc.pid,flush=True)
                    state['status']='running_prefix30_ablation' if ready else 'missing_native_references'
                    if time.monotonic()-last>30:publish();last=time.monotonic()
                    time.sleep(1)
                if len(state['results'])!=63:raise ValueError('Incomplete confirmation')
                remaining=[str(p) for p in (root/'runs').rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.tar')]
                if remaining:raise ValueError('Completed checkpoints remain')
                write(root/'verification.json',dict(status='verified_complete',runs=63,reused_runs=63,combinations=13,archives_sha256=archives,remaining_generated_checkpoints=remaining,local_free_gib=shutil.disk_usage(root).free/2**30))
                state.update(status='complete_prefix30_9_pairs',storage_after=json.loads(drive.call('about','hfh_storage:','--json')))
                publish();drive.upload(root/'verification.json');drive.upload(root/'controller.log')
                with zipfile.ZipFile(root/'all_results.zip','w',zipfile.ZIP_DEFLATED) as z:
                    for name in archives:z.write(root/(name+'.zip'),name+'.zip')
                    for name in ('status.json','summary.json','verification.json','launch_provenance.json','storage_preflight.json','execution_source.zip','controller.log','reused_references.zip','reused_results.json','reuse_verification.json','previous_results.json'):z.write(root/name,name)
                    z.write(package/'confirmation_design.json','confirmation_design.json')
                drive.upload(root/'all_results.zip');print('PREFIX30_COMPLETE',63,flush=True)
            except BaseException as error:
                state.update(status='error_no_expansion',error=str(error));print('CONFIRM_ERROR',str(error),flush=True)
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
