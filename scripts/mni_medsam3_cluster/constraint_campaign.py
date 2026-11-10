"""Cluster-owned staged, paired MNI MedSAM3 constraint replication."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign import controller, adni_foundation_full_pool as pool
from scripts.cross_dataset_campaign.common import ARMS, read, write, digest, stopping
from scripts.cross_dataset_campaign.storage import prepare, verify_archive, verify_results, retire, quota
from scripts.mni_medsam3_cluster.offline_ops import offline_ops as transport

PILOT = ('dice_bands', 'dice_bands_edge')
MODULE = 'scripts.mni_medsam3_cluster.constraint_campaign'
VALIDATED = {}
BASELINE_VALIDATED = {}


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen matrix source changed: '+name)
    plan = read(stage/'plan.json')
    if plan['seeds'] != [0, 1] or plan['folds'] != [0, 1, 2] or plan['arms'] != list(ARMS):
        raise ValueError('Matrix differs from authorized folds/seeds/MSD-ADNI arms')
    parent = Path(plan['baseline_stage'])
    if digest(parent/'manifest.json') != plan['baseline_manifest_sha256']:
        raise ValueError('Baseline source identity changed')
    return plan


def baseline_module(plan):
    path = Path(plan['baseline_stage'])/'operations/offline_v1/offline_ops.py'
    if digest(path) != plan['baseline_controller_sha256']:
        raise ValueError('Baseline controller changed')
    spec = importlib.util.spec_from_file_location('mni_baseline_offline', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cohort_rows(rows, threshold=.60):
    result = []
    for fold in (0, 1, 2):
        for seed in (0, 1):
            eligible = [r for r in rows if (r['fold'], r['seed']) == (fold, seed)
                        and r['validation_dice'] >= threshold]
            if not eligible:
                return None
            result.append(min(eligible, key=lambda r:r['updates']))
    return result


def baseline_records(parent, plan):
    """Read-only observer: never race the active controller's receipt writes."""
    rows=[]
    for task in plan['tasks']:
        out=parent/'runs'/task['id'];manifest=out/'archive_manifest.json'
        if not manifest.exists():continue
        key=(str(out),digest(manifest))
        if key not in BASELINE_VALIDATED:
            proof=read(manifest);verify_archive(out,proof);done=read(out/'completion.json')
            if (done['status']!='complete' or done['updates']!=task['updates']
                    or digest(out/'completion.json')!=proof['completion_sha256']
                    or digest(out/'evaluation.json')!=done['evaluation_sha256']
                    or digest(out/'protocol.json')!=done['protocol_sha256']):
                raise ValueError('Baseline completion integrity failed')
            BASELINE_VALIDATED[key]=dict(**task,validation_dice=done['validation_dice'],
                training_dice=done['training_dice'],observed_at=read(out/'milestone_observed.json')['time'])
        rows.append(BASELINE_VALIDATED[key])
    return rows


def paired_protocol(parent, row, arms=ARMS):
    """Bind training-only schedules to the first qualified baseline checkpoint."""
    output = parent/'runs'/row['id']; p = read(output/'protocol.json'); p.pop('arm', None)
    history = read(output/'training_history.json')
    if len(history) != row['updates'] or [h['update'] for h in history] != list(range(1, row['updates']+1)):
        raise ValueError('Baseline learning-rate history incomplete')
    order = []
    for count in range(400, row['updates']+1, 100):
        run = parent/'runs'/f"MNI_MedSAM3_n5_f{row['fold']}s{row['seed']}_native{count}"
        current = read(run/'slab_schedule.json')
        if len(current) != count*p['accumulation']:
            raise ValueError('Baseline sampling history incomplete')
        order.extend(current[len(order):count*p['accumulation']])
    p.update(name=f"MNI_constraints_f{row['fold']}s{row['seed']}", requested_arms=list(arms),
        frozen_learning_rates=[h['lr'] for h in history], frozen_sample_schedule=order,
        decision_rule='Fixed matched baseline budget; no arm-specific early stopping or validation tuning.',
        baseline_reference=dict(run=row['id'], validation_dice=row['validation_dice'],
            completion_sha256=digest(output/'completion.json'), protocol_sha256=digest(output/'protocol.json'),
            initial_adapter_sha256=read(output/'training_summary.json')['adapter_before'],
            frozen_base_sha256=read(output/'training_summary.json')['frozen_before']),
        geometry_adaptation='Native array voxel units as baseline; class-independent depth weights use foreground union. All four labels remain in CE; SDF uses each of the three foreground classes.')
    return p


def inventory(rows):
    return [dict(id=f"MNI_MedSAM3_n5_{arm}_f{row['fold']}s{row['seed']}",
        protocol=f"f{row['fold']}s{row['seed']}",kind='foundation',dataset='MNI',subjects=5,
        fold=row['fold'],seed=row['seed'],updates=row['updates'],arm=arm,
        phase='pilot' if arm in PILOT else 'matrix')
        for arm in (*PILOT, *(a for a in ARMS if a != 'dice' and a not in PILOT)) for row in rows]


def resolve(stage, plan):
    target = stage/'cohort.json'
    if target.exists():
        cohort = read(target)
        if cohort['plan_sha256'] != digest(stage/'plan.json'):
            raise ValueError('Resolved cohort belongs to another plan')
        if cohort['tasks']!=inventory(cohort['baselines']) or len(cohort['tasks'])!=78:
            raise ValueError('Resolved task inventory changed')
        for name, sha in cohort['files'].items():
            if digest(stage/name) != sha: raise ValueError('Resolved protocol changed')
            p=read(stage/name);reference=p['baseline_reference']
            out=Path(plan['baseline_stage'])/'runs'/reference['run']
            if digest(out/'completion.json')!=reference['completion_sha256'] or digest(out/'protocol.json')!=reference['protocol_sha256']:
                raise ValueError('Paired baseline evidence changed')
        return cohort
    parent = Path(plan['baseline_stage'])
    original = read(parent/'plan.json')
    selected_path = parent/'selected_seeds.json'
    if not selected_path.exists() or read(selected_path)['seeds'] != plan['seeds']:
        return None
    if any(r[0] in {p.stem for p in (parent/'submissions').glob('*.json')} for r in controller.queue()):
        return None
    rows = cohort_rows(baseline_records(parent, original), plan['threshold'])
    if rows is None: return None
    tasks = []; files = {}
    for row in rows:
        tag = f"f{row['fold']}s{row['seed']}"
        path = stage/'protocols'/tag/'protocol.json'
        write(path, paired_protocol(parent, row))
        files[str(path.relative_to(stage))] = digest(path)
    tasks=inventory(rows)
    cohort = dict(status='frozen',created_at=time.time(),plan_sha256=digest(stage/'plan.json'),
        files=files,baselines=rows,tasks=tasks,threshold=plan['threshold'],
        policy='First qualifying baseline per pair; paired sample/LR trajectories; all constraint arms start from the same medical LoRA.')
    write(target,cohort)
    return cohort


def completed(stage, tasks):
    result=[]
    for t in tasks:
        out=stage/'runs'/t['id']; path=out/'archive_manifest.json'
        if not path.exists():continue
        key=(str(out),digest(path))
        if key not in VALIDATED:
            verify_archive(out,read(path)); done=read(out/'completion.json')
            if done['updates']!=t['updates'] or done['arm']!=t['arm'] or done['status']!='complete':
                raise ValueError('Completed matrix cell differs')
            VALIDATED[key]=dict(**t,validation_dice=done['validation_dice'],training_dice=done['training_dice'])
        result.append(VALIDATED[key])
    return result


def next_tasks(tasks, rows, active):
    done={r['id'] for r in rows}
    phase='pilot' if any(t['phase']=='pilot' and t['id'] not in done for t in tasks) else 'matrix'
    return [t for t in tasks if t['phase']==phase and t['id'] not in done|active]


def report(stage, cohort, rows):
    reference={(r['fold'],r['seed']):r for r in cohort['baselines']}
    paired=[]
    for r in rows:
        out=stage/'runs'/r['id']; ev=read(out/'evaluation.json')
        validation=[v for v in ev['records'] if v['split']=='validation']
        paired.append(dict(**r,baseline_dice=reference[r['fold'],r['seed']]['validation_dice'],
            delta_dice=r['validation_dice']-reference[r['fold'],r['seed']]['validation_dice'],
            per_class_dice={c:sum(v['per_class'][c]['dice'] for v in validation)/len(validation)
                for c in ('CA1_3','subiculum','CA4_DG')},
            foreground_fp=sum(v['foreground_fp'] for v in validation),
            foreground_fn=sum(v['foreground_fn'] for v in validation)))
    grouped={}
    for arm in (a for a in ARMS if a!='dice'):
        cells=[r for r in paired if r['arm']==arm]
        if cells:grouped[arm]=dict(completed=len(cells),expected=6,
            mean_dice=sum(r['validation_dice'] for r in cells)/len(cells),
            mean_paired_delta=sum(r['delta_dice'] for r in cells)/len(cells),
            improved=sum(r['delta_dice']>0 for r in cells),
            by_fold={str(f):sum(r['delta_dice'] for r in cells if r['fold']==f)/sum(r['fold']==f for r in cells)
                     for f in (0,1,2) if any(r['fold']==f for r in cells)})
    summary=dict(updated_at=time.time(),paired=paired,by_arm=grouped,
        caveat='Selected validation folds; paired descriptive evidence, not an untouched external test set. No selective omission of worse arms.')
    write(stage/'comparison.json',summary)
    if all(grouped.get(a,{}).get('completed')==6 for a in PILOT):
        write(stage/'pilot_comparison.json',dict(updated_at=time.time(),by_arm={a:grouped[a] for a in PILOT},
            paired=[r for r in paired if r['arm'] in PILOT],
            next_phase='Full predefined matrix proceeds irrespective of pilot direction; no outcome-driven retuning.'))


def finish(stage, task):
    out=stage/'runs'/task['id']
    if not (out/'archive_manifest.json').exists():
        verify_results(out,task)
        p=read(out/'protocol.json'); history=read(out/'training_history.json'); train=read(out/'training_summary.json')
        if [h['lr'] for h in history]!=p['frozen_learning_rates'] or read(out/'slab_schedule.json')!=p['frozen_sample_schedule']:
            raise ValueError('Constraint run differs from paired baseline trajectory')
        ref=p['baseline_reference']
        if train['adapter_before']!=ref['initial_adapter_sha256'] or train['frozen_before']!=ref['frozen_base_sha256']:
            raise ValueError('Constraint initialization differs from baseline')
        gate=read(stage/'gpu_preflight'/f"f{task['fold']}"/'complete.json')
        if gate['status']!='passed' or gate['optimizer_updates']!=0:raise ValueError('Missing full-loss GPU gate')
        write(out/'matrix_audit.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),
            cohort_sha256=digest(stage/'cohort.json'),gpu_gate=gate,matched_schedule=True,matched_initialization=True,
            baseline_reference=ref,phase=task['phase']))
        prepare(out,task)


def cleanup(stage, rows):
    for r in rows:
        out=stage/'runs'/r['id']
        if (out/'archive_acknowledgment.json').exists() and not (out/'cleanup.json').exists():retire(out)


def summarize(stage, plan, status, active=None):
    cohort=read(stage/'cohort.json') if (stage/'cohort.json').exists() else None
    rows=completed(stage,cohort['tasks']) if cohort else []
    pending=[r['id'] for r in rows if not (stage/'runs'/r['id']/'cleanup.json').exists()]
    if status=='complete' and pending:status='complete_pending_archive'
    write(stage/'status.json',dict(status=status,active=active or [],completed=len(rows),total=78,
        folds=[0,1,2],seeds=[0,1],pilot_complete=sum(r['phase']=='pilot' for r in rows),
        pending_archive_or_cleanup=pending,job_id=os.environ.get('SLURM_JOB_ID'),updated_at=time.time(),
        laptop_required=False,quota=quota()))


def step_succeeded(output):
    """A launcher error may be reconciled only with successful Slurm accounting."""
    binding=read(output/'gpu_worker.json'); identity=f"{binding['slurm_job_id']}.{binding['slurm_step_id']}"
    text=subprocess.check_output(['sacct','-j',identity,'--format=JobID,State,ExitCode','-n','-P'],text=True)
    return any(line.split('|')[:3]==[identity,'COMPLETED','0:0'] for line in text.splitlines())


def saved_stop(output):
    marker=output/'requested_stop.json'
    if not marker.exists():return False
    files=read(marker).get('files',{})
    return bool(files) and all((output/n).exists() and digest(output/n)==sha for n,sha in files.items())


def gate_passed(stage, task):
    path=task['output']/'complete.json'
    if not path.exists():return False
    proof=read(path)
    if (proof['status']!='passed' or proof['optimizer_updates']!=0
            or proof['adapter_before']!=proof['adapter_after']
            or proof['protocol_sha256']!=digest(stage/'protocols'/task['protocol']/'protocol.json')):
        raise ValueError('Cached full-loss GPU preflight differs')
    return True


def command(stage,plan,task,base,lane,preflight=False):
    args=[*pool.step_prefix(),'env',f'HIPPO_GPU_WORKER={lane}',f'MNI_MANIFEST_SHA={digest(stage/"manifest.json")}',
        plan['foundation_python'],'-m','scripts.mni_medsam3_cluster.constraint_worker',
        '--protocol',str(stage/'protocols'/task['protocol']/'protocol.json'),
        '--output',str(task['output'] if preflight else stage/'runs'/task['id']),
        '--record',str((task['output'] if preflight else stage/'runs'/task['id'])/'gpu_worker.json'),
        '--base-weights',str(base),'--arm','dice_bands_edge_sdf_depth' if preflight else task['arm']]
    if preflight:args+=['--preflight']
    return args


def gpu_gate(stage,plan,base,publish):
    tasks=[dict(id=f'preflight_f{f}',protocol=f'f{f}s0',output=stage/'gpu_preflight'/f'f{f}') for f in (0,1,2)]
    workers={}
    try:
        while True:
            if stopping():pool.signal_workers(workers)
            for lane,w in list(workers.items()):
                code=w['process'].poll()
                if code is None:continue
                w['stream'].close();out=w['output']
                if stopping() and saved_stop(out):del workers[lane];continue
                if code and not(code==73 and (out/'complete.json').exists() and step_succeeded(out)):
                    raise RuntimeError('Full-loss preflight failed: '+w['task']['id'])
                if not gate_passed(stage,w['task']):raise ValueError('Missing GPU gate')
                del workers[lane]
            if stopping():return False
            for lane in range(2):
                if lane in workers:continue
                active={w['task']['id'] for w in workers.values()}
                ready=[t for t in tasks if not gate_passed(stage,t) and t['id'] not in active]
                if not ready:continue
                t=ready[0];out=t['output'];out.mkdir(parents=True,exist_ok=True)
                (out/'requested_stop.json').unlink(missing_ok=True);stream=(out/'preflight.log').open('a')
                process=subprocess.Popen(command(stage,plan,t,base,lane,True),cwd=stage/'source',stdout=stream,stderr=subprocess.STDOUT)
                workers[lane]=dict(task=t,output=out,phase='preflight',process=process,stream=stream)
            publish('gpu_preflight',workers)
            if not workers:return True
            time.sleep(5)
    finally:
        pool.signal_workers(workers)
        for w in workers.values():
            w['process'].wait(timeout=300);w['stream'].close()


def dispatch(stage,plan,base,publish):
    cohort=read(stage/'cohort.json');tasks=cohort['tasks'];workers={};reason=None
    try:
        while True:
            if stopping() and reason is None:reason='paused' if read(stage/'control.json').get('pause') else 'continuing_at_time_limit'
            if reason:pool.signal_workers(workers)
            for lane,w in list(workers.items()):
                code=w['process'].poll()
                if code is None:continue
                w['stream'].close();out=w['output']
                if reason and saved_stop(out):del workers[lane];continue
                if code and not(code==73 and (out/'completion.json').exists() and step_succeeded(out)):
                    raise RuntimeError(f"{w['task']['id']}: worker exited {code}")
                finish(stage,w['task']);del workers[lane]
            rows=completed(stage,tasks);cleanup(stage,rows)
            if reason:
                pool.signal_workers(workers)
                if not workers:publish(reason,{});return reason
            else:
                for lane in range(2):
                    if lane in workers:continue
                    ready=next_tasks(tasks,rows,{w['task']['id'] for w in workers.values()})
                    if not ready:continue
                    task=ready[0];out=stage/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
                    if (out/'completion.json').exists():finish(stage,task);rows=completed(stage,tasks);continue
                    if quota()['free_gib']<plan['storage']['minimum_free_gib']:reason='waiting_for_quota';break
                    if task['phase']=='matrix' and not (stage/'pilot_comparison.json').exists():report(stage,cohort,rows)
                    (out/'requested_stop.json').unlink(missing_ok=True);stream=(out/'train.log').open('a')
                    process=subprocess.Popen(command(stage,plan,task,base,lane),cwd=stage/'source',stdout=stream,stderr=subprocess.STDOUT)
                    workers[lane]=dict(task=task,output=out,phase='running',process=process,stream=stream)
                if not workers and not next_tasks(tasks,rows,set()):
                    report(stage,cohort,rows);publish('complete',{});return 'complete'
            publish(reason or 'running',workers)
            # Comparisons are regenerated after each completed arm, not per batch.
            if getattr(dispatch,'last_count',-1)!=len(rows):report(stage,cohort,rows);dispatch.last_count=len(rows)
            time.sleep(5)
    finally:
        pool.signal_workers(workers)
        for w in workers.values():
            if w['process'].poll() is None:w['process'].wait(timeout=300)
            w['stream'].close()


def run(stage):
    plan=verify(stage);cohort=resolve(stage,plan)
    if not cohort:raise RuntimeError('Three-fold baseline gate has not passed')
    controller.summarize=summarize;pool.gpu_preflight=gpu_gate;pool.dispatch=dispatch
    pool.run(stage,plan,lambda:None,lambda:None)


def recover_baseline(parent, module, plan):
    """Recover only the observed launcher-73/completed-step discrepancy."""
    state=read(parent/'status.json')
    if state.get('status')!='failed':return False
    failure=read(parent/'failure.json')
    match=re.search(r'(MNI_MedSAM3_n5_f\d+s\d+_native\d+): worker exited73',failure['traceback'])
    if not match:return False
    out=parent/'runs'/match.group(1)
    if not (out/'completion.json').exists() or not step_succeeded(out):return False
    with (parent/'offline_service.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        task=next(t for t in read(parent/'plan.json')['tasks'] if t['id']==out.name)
        verify_results(out,task);module.prepare_completed(parent,task)
        incident=parent/'offline_state'/f"reconciled_launcher_{failure['job_id']}.json"
        if incident.exists():return False
        write(incident,dict(failure=failure,verified_completion_sha256=digest(out/'completion.json'),time=time.time()))
        (parent/'failure.json').rename(parent/'offline_state'/f"failure_reconciled_{failure['job_id']}.json")
        module.summarize(parent,read(parent/'plan.json'),'ready_for_offline_resume',[])
    env=dict(os.environ,PYTHONPATH=':'.join(str(parent/x) for x in ('source','source/scripts','source/baselines/nnUNet')))
    with (parent/'offline_service.log').open('ab') as log:
        subprocess.Popen([plan['conventional_python'],'-u',str(parent/'operations/offline_v1/offline_ops.py'),'serve','--stage',str(parent)],
            cwd=parent/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return True


def serve(stage):
    plan=verify(stage);parent=Path(plan['baseline_stage']);baseline=baseline_module(plan)
    transport.OPS=stage;source=Path(plan['base_source'])
    if digest(source)!=plan['base_sha256']:raise ValueError('Persistent base model changed')
    with (stage/'service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if read(stage/'control.json').get('pause') or read(parent/'control.json').get('pause'):return
                current=controller.queue();own={p.stem for p in (stage/'submissions').glob('*.json')}
                active=[r for r in current if r[0] in own]
                cohort=resolve(stage,plan)
                if cohort is None:
                    original_active=any(r[0] in {p.stem for p in (parent/'submissions').glob('*.json')} for r in current)
                    if not original_active:recover_baseline(parent,baseline,plan)
                    state=read(parent/'status.json')
                    summarize(stage,plan,'waiting_for_consistent_baselines')
                    write(stage/'baseline_gate.json',dict(status=state['status'],required='Seeds 0 and 1 each >=60% on folds 0/1/2',
                        cap_updates=1000,observed=state.get('runs',[]),time=time.time()))
                else:
                    transport.deliver_base(stage,plan,current,source)
                    if not active:
                        rows=completed(stage,cohort['tasks']);cleanup(stage,rows);report(stage,cohort,rows)
                        state=read(stage/'status.json') if (stage/'status.json').exists() else {}
                        if len(rows)==len(cohort['tasks']):
                            summarize(stage,plan,'complete')
                            if read(stage/'status.json')['status']=='complete':return
                        elif state.get('status') not in ('failed','paused'):
                            transport.submit(stage,plan)
                write(stage/'service.json',dict(status='running',pid=os.getpid(),active_jobs=active,
                    baseline_gate_passed=cohort is not None,last_check=time.time(),laptop_required=False))
            except Exception:
                write(stage/'service_error.json',dict(time=time.time(),traceback=traceback.format_exc()))
            time.sleep(20)


def preflight(stage):
    plan=verify(stage)
    if digest(Path(plan['base_source']))!=plan['base_sha256']:raise ValueError('Base checksum differs')
    import numpy as np
    from scripts.depth_supervision.geometry import build_geometry, GeometryConfig, signed_distance
    from scripts.mni_medsam3_cluster.constraint_support import depth_geometry
    parent=Path(plan['baseline_stage']);inventory=read(parent/'manifest.json');geometry=[];seen=set()
    for fold in (0,1,2):
        p=read(parent/'protocols'/f'MNI_MedSAM3_n5_f{fold}s0_native400'/'protocol.json')
        if digest(Path(p['lora_weights']))!=p['lora_sha256']:raise ValueError('Medical initialization changed')
        if {c.rsplit('_',1)[0] for c in p['train_cases']} & {c.rsplit('_',1)[0] for c in p['evaluation_cases']}:
            raise ValueError('Subject leakage')
        for case in p['train_cases']+p['evaluation_cases']:
            path=Path(p['data'])/(case+'.npz')
            if case not in seen:
                if digest(path)!=inventory['data/'+path.name]:raise ValueError('MNI data differs: '+case)
                seen.add(case)
            if case not in p['train_cases']:continue
            with np.load(path,allow_pickle=False) as item:labels=item['label'].transpose(2,1,0)
            affine=np.eye(4);affine[:3,:3]=np.eye(3)[::-1]
            weights=depth_geometry(build_geometry,labels,affine,GeometryConfig(),fields=False)['weights']
            if not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1.,atol=1e-6):
                raise ValueError('Invalid MNI depth weights')
            for c in (1,2,3):
                sdf=signed_distance(labels==c,(1.,1.,1.),8.)/8.
                if not np.isfinite(sdf).all() or (sdf[labels==c]>=0).any() or (sdf[labels!=c]<=0).any():
                    raise ValueError('Invalid classwise MNI SDF signs')
            geometry.append(dict(fold=fold,case=case,labels=sorted(np.unique(labels).tolist()),weight_sum=float(weights.sum())))
    for name,sha in read(parent/'upstream_manifest.json').items():
        if digest(Path(p['upstream'])/name)!=sha:raise ValueError('Upstream source differs')
    write(stage/'geometry_preflight.json',dict(status='passed',training_crops=geometry,verified_data_files=len(seen),
        labels_preserved=True,units='Native array voxels; same identity geometry convention as the baseline and ADNI engine'))
    usage=quota()
    if usage['free_gib']<plan['offline_peak_gib']+plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient headroom for all baseline and matrix artifacts while laptop is offline')
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    test=subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)],capture_output=True,text=True,check=True)
    write(stage/'preflight.json',dict(status='passed',operational_manifest_sha256=digest(stage/'manifest.json'),
        quota=usage,offline_peak_gib=plan['offline_peak_gib'],scheduler=test.stdout+test.stderr,time=time.time(),
        gpu_gate='Full-loss finite backward and calibration on each fold, zero optimizer updates, before constraint training'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['verify','preflight','serve','run'])
    parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    globals()[args.command](args.stage.resolve())


if __name__=='__main__':main()
