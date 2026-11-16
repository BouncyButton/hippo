"""One bounded f0/s0 search; verified archives precede checkpoint retirement."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
import numpy as np

from scripts.colab_hfh import runtime
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh.nnunet_search13_policy import NAME,CAP,ARMS,candidates,active_coefficients,probe_allowance
from scripts.colab_hfh.nnunet_composition_campaign import SDKDrive,check_no_workers,check_prior_attempt,verify_package,retire,zip_archive
from scripts.colab_hfh.nnunet_composition_a100_campaign import archive_without_retirement

read,write,digest = runtime.read,runtime.write,runtime.digest


def activity_audit(output,p,s):
    names = components(p['active_objective'])
    rows = [json.loads(line) for line in (output/'constraint_activity.jsonl').read_text().splitlines()]
    prefix = s['prefix_updates']
    expected = list(range(prefix+1,s['max_steps']+1))
    if [r['charged_attempt'] for r in rows] != expected:
        raise ValueError('Missing/replayed activity observations')
    anchor = read(output/'calibration.json')['weights']
    for row in rows:
        w,stats = active_coefficients(row['gradient_gram'],anchor,names,p['activity_policy'])
        scale = min(1.,(row['charged_attempt']-prefix)/p['constraint_warmup_updates'])
        if (row['arm'] != p['active_objective'] or row['anchor_weights'] != anchor or
                row['probe_flops'] != probe_allowance(p['active_objective']) or
                not math.isclose(row['scale'],scale,abs_tol=1e-12) or
                any(not math.isclose(w[n],row['effective_weights'][n],rel_tol=1e-9) for n in names) or
                any(not math.isclose(stats['component_ratios'][n],row['component_ratios'][n],rel_tol=1e-9) for n in names) or
                set(row['raw_losses']) != set(names) or not all(math.isfinite(v) for v in row['raw_losses'].values())):
            raise ValueError('Activity trace differs from training rule')
    full = [r for r in rows if r['scale'] == 1.]
    if len(full) < p.get('minimum_full_strength_updates',60):
        raise ValueError('Insufficient full-strength exposure')
    result = dict(status='verified',active_updates=len(rows),full_strength_updates=len(full),
        adjusted_updates=sum(r['adjusted'] for r in rows),
        minimum_full_strength_ratio={n:min(r['component_ratios'][n] for r in full) for n in names},
        minimum_full_strength_share={n:min(r['component_shares'][n] for r in full) for n in names},
        optimization_probe_flops=len(rows)*probe_allowance(p['active_objective']),
        interpretation='All included components active on every post-ramp training update, measured at full-resolution logits; no anatomical-feasibility guarantee')
    write(output/'activity_verification.json',result)
    return result


def adherence_summary(path):
    report = read(path)
    if report['status'] != 'complete' or len(report['records']) != 20:
        raise ValueError('Incomplete final adherence evaluation')
    result = {}
    for split in ('training','validation'):
        rows = [r for r in report['records'] if r['split'] == split]
        if len(rows) != 10 or len({r['case_name'] for r in rows}) != 10:
            raise ValueError('Adherence case identities invalid')
        result[split] = {n:float(np.mean([r['losses'][n] for r in rows]))
                         for n in ('bands','edge','sdf','depth')}
    return result


def archive_result(root,package,name):
    output,durable,protocol = root/'runs'/name/'run',root/'runs'/name/'archive',package/'protocols'/name
    p,c,done = read(protocol/'protocol.json'),read(output/'config.json'),read(output/'completion.json')
    arm = p['active_objective']; s = p['models']['nnunet']['objectives'][arm]
    historical = package/p.get('native_reference_folder','reference_a100_native')/'results'
    original = read(historical/'config.json')
    for key in ('fold','seed','dataset_sha256','train_cases','validation_cases','precision','initial_lr',
                'lr_horizon_updates','nnunet_warmup_fraction','nnunet_sgd_momentum','augmentation','native_loss'):
        if c[key] != original[key]:raise ValueError('Native recipe mismatch: '+key)
    if read(output/'initialization.json') != read(historical/'initialization.json'):
        raise ValueError('Initialization mismatch')
    used = training_flops(done['updates'],s)
    if (done['status'] != 'complete' or done['discarded_attempts'] or done['updates'] != s['max_steps']
            or done['consumed_training_flops'] != used or not used <= CAP < training_flops(done['updates']+1,s)
            or done['calibration_attempts'] != (0 if name == 'native' else 1)
            or done['calibration_flops'] != s['calibration_flops']):
        raise ValueError('Optimization/calibration accounting mismatch')
    native = root/'runs/native/archive'
    val,train = read(output/'validation_latest.json'),read(output/'training_final.json')
    adherence = adherence_summary(output/'constraint_adherence.json')
    row = dict(variant=name,arm=arm,fold=0,seed=0,updates=done['updates'],
        validation_dice=val['summary']['macro_hard_dice'],training_dice=train['summary']['macro_hard_dice'],
        optimization_flops=used,optimization_cap=CAP,calibration_flops=done['calibration_flops'],
        constraint_adherence=adherence,ap_swaps_applicable=False)
    for split,evaluation in [('validation',val),('training',train)]:
        for error in ('fp','fn'):
            row[split+'_'+error] = sum(r['foreground_'+error] for r in evaluation['cases'])
    if name == 'native':
        anchor_reference = (package/'reference_a100_native/results' if p.get('shared_cap_search') else historical)
        if read(output/'native_update60.json') != read(anchor_reference/'native_update60.json'):
            raise ValueError('Fresh native does not reproduce the previous A100 update60')
        if not p.get('shared_cap_search') and abs(row['validation_dice']-read(historical/'validation_latest.json')['summary']['macro_hard_dice']) > 1e-12:
            raise ValueError('Fresh native does not reproduce the previous A100 endpoint')
        row.update(native_gain_pp=0.,reproduced_previous_a100_prefix=True,
                   reference_hardware='Fresh A100 shared-cap native' if p.get('shared_cap_search') else 'Previous matched A100 native')
    else:
        cal = read(output/'calibration.json')
        prefix = s['prefix_updates']
        if cal['anchor_successful_updates'] != prefix or cal['anchor_model_sha256'] != read(native/f'native_update{prefix}.json')['model_state_sha256']:
            raise ValueError('Matched native prefix failed')
        row.update(activity=activity_audit(output,p,s),recipe=p['search_recipe'],weights=cal['weights'],
                   matched_native_prefix_verified=True)
        baseline = read(native/'validation_latest.json')
        by_case = {r['case_name']:r for r in baseline['cases']}
        deltas = [100*(r['macro_hard_dice']-by_case[r['case_name']]['macro_hard_dice']) for r in val['cases']]
        row.update(native_gain_pp=100*(row['validation_dice']-baseline['summary']['macro_hard_dice']),
                   improved_crops=sum(d>0 for d in deltas),worst_crop_pp=min(deltas))
        old = adherence_summary(native/'constraint_adherence.json')
        row['adherence_delta_vs_native'] = {split:{n:adherence[split][n]-old[split][n]
            for n in components(arm)} for split in old}
        row['all_included_validation_penalties_improved'] = all(v < 0 for v in row['adherence_delta_vs_native']['validation'].values())
    write(output/'comparison.json',row)
    archive_without_retirement(output,durable,protocol)
    return row


def leaders(results):
    output = []
    for arm in ARMS:
        rows = [r for r in results if r['arm'] == arm and r.get('activity',{}).get('status') == 'verified']
        if not rows:continue
        winner = max(rows,key=lambda r:(r['validation_dice'],-r['recipe']['target'],r['variant']))
        adherent = [r for r in rows if r['all_included_validation_penalties_improved']]
        output.append(dict(arm=arm,candidates_completed=len(rows),best=winner,
                          best_with_all_penalties_improved=max(adherent,key=lambda r:r['validation_dice']) if adherent else None))
    return output


def run(root,package,max_workers):
    import psutil
    verify_package(package)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_no_workers()
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
        if 'A100' not in gpu:raise ValueError('A100 required')
        # Measured previous runs reserve <0.7 GiB; allow 4 GiB GPU and 8 GiB
        # host RAM per worker, plus headroom. One intra-op CPU thread per worker.
        safe = min(len(os.sched_getaffinity(0)),int((int(gpu.rsplit(',',1)[1])/1024-8)//4),
                   int((psutil.virtual_memory().available/2**30-16)//8))
        if not 1 <= max_workers <= safe:raise ValueError('Worker count exceeds measured resource headroom')
        drive = SDKDrive('results_'+NAME)
        quota = json.loads(drive.call('about','hfh_storage:','--json'))
        if min(quota['free'],shutil.disk_usage(root).free) < (max_workers*2+8)*2**30:
            raise ValueError('Insufficient recovery/archive headroom')
        write(root/'storage_preflight.json',dict(drive=quota,local_free=shutil.disk_usage(root).free,
            peak_generated_estimate_bytes=(max_workers*2+8)*2**30,max_workers=max_workers,
            gpu=gpu,available_cpus=len(os.sched_getaffinity(0)),checked_at=time.time()))
        for rel,sha in read(root/'execution_source_manifest.json').items():
            if digest(root/'bundle/source'/rel) != sha:raise ValueError('Execution source changed: '+rel)
        state_path = root/'status.json'
        previous = read(state_path) if state_path.exists() else None
        check_prior_attempt(previous,drive.previous_state(root/'previous_remote_status.json'))
        if previous is not None:raise RuntimeError('Existing search attempt: audit recovery before any relaunch')
        state = dict(run_id=uuid.uuid4().hex,status='starting',fold=0,seed=0,results=[],workers=[],
            max_gpu_workers=max_workers,planned_candidates=len(candidates()),planned_total_runs=1+len(candidates()),
            planned_combinations=list(ARMS),selection='Exploratory f0/s0; no generalization claim')
        write(state_path,state)
        for file in ('execution_source.zip','launch_provenance.json','storage_preflight.json'):
            drive.upload(root/file)
        drive.upload(package/'search_design.json')
        threads = str(read(package/'search_design.json').get('threads_per_worker',1))
        env = dict(os.environ,PYTHONPATH=':'.join([str(root)]+[str(root/'bundle'/p) for p in
            ('compatibility','source','source/scripts','source/baselines/nnUNet')]),
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,
            WANDB_MODE='disabled',HIPPO_HFH_GPU_CACHE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            HFH_SEARCH_GPU_FRACTION=str(min(.28,.80/max_workers)))
        queue = ['native']+[r['id'] for r in candidates()]
        active = {}; completed_archives = {}
        launched = 0
        def publish():
            workers = []
            for name,(proc,log) in active.items():
                if proc.poll() is None:
                    ledger = root/'runs'/name/'run/budget_ledger.json'
                    workers.append(dict(name=name,pid=proc.pid,step=read(ledger)['charged_attempts'] if ledger.exists() else 0))
            state.update(workers=workers,remaining_to_launch=len(queue),updated_at=time.time())
            write(state_path,state);drive.upload(state_path)
            write(root/'leaders.json',leaders(state['results']));drive.upload(root/'leaders.json')
        def finish(name,proc,log):
            log.close()
            if proc.returncode:raise RuntimeError('Worker failed; recovery retained and no silent retry: '+name)
            row = archive_result(root,package,name)
            output,durable = root/'runs'/name/'run',root/'runs'/name/'archive'
            zipped = root/(name+'.zip')
            zip_archive(zipped,durable,package/'protocols'/name,package);drive.upload(zipped)
            retire(output,durable)
            zip_archive(zipped,durable,package/'protocols'/name,package);drive.upload(zipped)
            completed_archives[name] = digest(zipped)
            state['results'].append(row)
            del active[name]
            print('SEARCH_ARCHIVED',name,'gain_pp',round(row['native_gain_pp'],5),flush=True)
        last = 0.
        try:
            while queue or active:
                # Verify the fresh baseline + two instrumented pilots before
                # expanding to the full resource-bound worker count.
                pilot_ready = len(state['results']) >= 3
                limit = max_workers if pilot_ready else min(3,max_workers)
                while queue and len(active) < limit and (pilot_ready or launched < 3):
                    name = queue.pop(0)
                    output,durable = root/'runs'/name/'run',root/'runs'/name/'archive'
                    output.mkdir(parents=True);durable.mkdir(parents=True)
                    write(output/'controller_attempt.json',dict(run_id=state['run_id'],started_at=time.time()))
                    write(output/'execution_environment.json',read(root/'launch_provenance.json')['hardware'])
                    log = (output/'worker.log').open('a')
                    module = read(package/'search_design.json').get('worker_module','scripts.colab_hfh.nnunet_search13_worker')
                    proc = subprocess.Popen([sys.executable,'-m',module,
                        '--protocol-dir',str(package/'protocols'/name),'--output',str(output),'--durable',str(durable),
                        '--pkl',str(root/'bundle/data_index.json'),'--fold','0','--seed','0'],
                        cwd=root/'bundle/source',env={**env,'HIPPO_DEADLINE':str(time.time()+3600)},
                        stdout=log,stderr=subprocess.STDOUT)
                    active[name] = (proc,log);launched += 1
                    print('SEARCH_START',name,proc.pid,flush=True)
                state['status'] = 'running_full_search' if pilot_ready else 'verifying_initial_pilots'
                for name,(proc,log) in list(active.items()):
                    if proc.poll() is None:continue
                    if proc.returncode:raise RuntimeError('Worker failed; recovery retained: '+name)
                    if name != 'native' and not (root/'runs/native/archive/cleanup.json').exists():continue
                    finish(name,proc,log)
                    last = 0.
                if time.monotonic()-last > 20:
                    publish();last = time.monotonic()
                if active:time.sleep(1)
            if len(state['results']) != state['planned_total_runs'] or len(leaders(state['results'])) != 13:
                raise ValueError('Incomplete 13-combination search')
            remaining = [str(p) for p in (root/'runs').rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.tar')]
            if remaining:raise ValueError('Completed generated checkpoints remain')
            write(root/'verification.json',dict(status='verified_complete',runs=len(state['results']),
                combinations=13,archives_sha256=completed_archives,remaining_generated_checkpoints=remaining,
                local_free_gib=shutil.disk_usage(root).free/2**30))
            state.update(status='complete_search13_f0s0',workers=[],
                         storage_after=json.loads(drive.call('about','hfh_storage:','--json')))
            publish();drive.upload(root/'verification.json');drive.upload(root/'controller.log')
            # One downloadable durable bundle for the local project reports.
            with zipfile.ZipFile(root/'all_results.zip','w',zipfile.ZIP_DEFLATED) as z:
                for name in completed_archives:z.write(root/(name+'.zip'),name+'.zip')
                for name in ('status.json','leaders.json','verification.json','launch_provenance.json','storage_preflight.json','execution_source.zip','controller.log'):
                    z.write(root/name,name)
            drive.upload(root/'all_results.zip')
            print('SEARCH_COMPLETE',len(state['results']),flush=True)
        except BaseException as error:
            state.update(status='error_no_expansion',error=str(error))
            # Let already charged workers finish; archive successful ones when
            # possible. Never destroy a failed worker's recovery.
            for name,(proc,log) in list(active.items()):
                proc.wait()
                if proc.returncode == 0 and (name == 'native' or (root/'runs/native/archive/cleanup.json').exists()):
                    try:finish(name,proc,log)
                    except Exception:log.close()
                else:log.close()
            publish();drive.upload(root/'controller.log')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--max-workers',type=int,required=True)
    args = parser.parse_args()
    run(args.root,args.package,args.max_workers)
