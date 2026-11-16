"""Cluster-owned third-seed search after the immutable two-seed matrix."""
import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from scripts.cross_dataset_campaign import controller, adni_foundation_full_pool as pool
from scripts.cross_dataset_campaign.common import ARMS, read, write, digest
from scripts.cross_dataset_campaign.storage import quota, retire
from scripts.mni_medsam3_cluster import constraint_campaign as matrix
from scripts.mni_medsam3_cluster.offline_ops import offline_ops as ops

MODULE='scripts.mni_medsam3_cluster.seed_search'


def verify_files(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen search source changed: '+name)
    return read(stage/'plan.json')


def baseline_verify(stage):
    plan=verify_files(stage);seed=plan['candidate_seed']
    expected={(f,seed,u) for f in range(3) for u in range(400,1001,100)}
    if (plan['threshold'],plan['maximum_updates'],plan['parallelism']['workers'])!=(.6,1000,2):
        raise ValueError('Candidate baseline policy changed')
    if len(plan['tasks'])!=21 or {(t['fold'],t['seed'],t['updates']) for t in plan['tasks']}!=expected:
        raise ValueError('Candidate milestone inventory differs')
    for task in plan['tasks']:
        p=read(stage/'protocols'/task['protocol']/'protocol.json')
        if (p['fold'],p['seed'],p['updates'],p['train_subjects'],p['requested_arms'],p['augmentation'])!=(task['fold'],seed,task['updates'],5,['dice'],'none'):
            raise ValueError('Candidate protocol differs')
        if {c.rsplit('_',1)[0] for c in p['train_cases']} & {c.rsplit('_',1)[0] for c in p['evaluation_cases']}:
            raise ValueError('Subject leakage')
    return plan


def next_baselines(plan,rows,selected,active):
    ready=[]
    for fold in range(3):
        history=[r for r in rows if r['fold']==fold]
        if any(r['validation_dice']>=plan['threshold'] for r in history):continue
        updates=max((r['updates'] for r in history),default=300)+100
        if updates>plan['maximum_updates']:continue
        task=next(t for t in plan['tasks'] if t['fold']==fold and t['updates']==updates)
        if task['id'] not in active:ready.append(task)
    return sorted(ready,key=lambda t:(t['updates'],t['fold']))


def baseline_cleanup(stage,plan,rows,selected,active):
    # Empty discovery selection keeps failed fold0 checkpoints until the next
    # milestone's verified optimizer/RNG transfer, just as on other folds.
    archived=[r for r in rows if (stage/'runs'/r['id']/'archive_acknowledgment.json').exists()]
    ops.study.base.retire=retire
    ops.study.retire_ready(stage,plan,archived,[],active)


def baseline_summary(stage,plan,status,active=None):
    rows=ops.records(stage,plan)
    write(stage/'status.json',dict(status=status,active=active or [],runs=rows,
        candidate_seed=plan['candidate_seed'],completed_milestones=len(rows),updated_at=time.time(),
        pending_archive_or_cleanup=[r['id'] for r in rows if not (stage/'runs'/r['id']/'cleanup.json').exists()],
        job_id=os.environ.get('SLURM_JOB_ID'),laptop_required=False,quota=quota()))


def run_baseline(stage):
    plan=baseline_verify(stage);ops.OPS=stage
    ops.next_trials=next_baselines;ops.retire_archived=baseline_cleanup
    controller.summarize=baseline_summary;pool.dispatch=ops.dispatch
    pool.gpu_preflight=lambda *args:True
    pool.run(stage,plan,lambda:None,lambda:None)


def freeze(stage):
    manifest={str(p.relative_to(stage)):digest(p)
        for p in sorted(stage.rglob('*')) if p.is_file() and not p.is_symlink()}
    # source is a shared immutable directory, explicitly bound by every file.
    manifest.update({'source/'+str(p.relative_to(stage/'source')):digest(p)
        for p in sorted((stage/'source').rglob('*')) if p.is_file()})
    write(stage/'manifest.json',manifest)
    write(stage/'control.json',dict(pause=False,controller_owner='offline_v1',automatic_continuation_owner='seed_search'))


def make_baseline(root,seed):
    config=read(root/'plan.json');reference=Path(config['baseline_stage'])
    stage=root/'candidates'/f'seed{seed}'/'baseline'
    if (stage/'manifest.json').exists():baseline_verify(stage);return stage
    stage.mkdir(parents=True,exist_ok=True)
    if not (stage/'source').exists():(stage/'source').symlink_to(root/'source',target_is_directory=True)
    tasks=[]
    for fold in range(3):
        for updates in range(400,1001,100):
            template=f'MNI_MedSAM3_n5_f{fold}s2_native{updates}'
            p=read(reference/'protocols'/template/'protocol.json')
            name=f'MNI_MedSAM3_n5_f{fold}s{seed}_native{updates}'
            p.update(name=name,seed=seed)
            write(stage/'protocols'/name/'protocol.json',p)
            tasks.append(dict(id=name,protocol=name,kind='foundation',dataset='MNI',subjects=5,
                fold=fold,seed=seed,updates=updates,arm='dice'))
    plan=dict(read(reference/'plan.json'),name=f'MNI_third_seed_{seed}',candidate_seed=seed,seeds=[seed],tasks=tasks)
    plan.pop('initial_recovery',None);plan.pop('archive_resume_module',None)
    write(stage/'plan.json',plan)
    script=(root/'run.sbatch').read_text().replace('constraint_campaign run','seed_search run-baseline')
    (stage/'run.sbatch').write_text(script)
    write(stage/'provenance.json',dict(search_manifest_sha256=digest(root/'manifest.json'),
        original_baseline_manifest_sha256=config['baseline_manifest_sha256'],seed=seed,
        scientific_change='Random seed only; fixed five-subject fold supports and original native training engine.',
        schedule='400 then100-update continuation up to1000 on each fold; all folds must reach60%.'))
    freeze(stage)
    write(stage/'selected_seeds.json',dict(seeds=[seed],method='Fixed candidate; no first-two discovery shortcut'))
    if seed==2:
        rows=matrix.baseline_records(reference,read(reference/'plan.json'))
        for row in rows:
            if row['seed']!=2:continue
            target=stage/'runs'/row['id'];target.parent.mkdir(exist_ok=True)
            if not target.exists():target.symlink_to(reference/'runs'/row['id'],target_is_directory=True)
        write(stage/'reused_baselines.json',dict(rows=[r for r in rows if r['seed']==2]))
    return stage


def make_matrix(root,seed,baseline):
    config=read(root/'plan.json');prior=Path(config['previous_matrix_stage'])
    stage=root/'candidates'/f'seed{seed}'/'constraints'
    if (stage/'manifest.json').exists():matrix.verify(stage);return stage
    stage.mkdir(parents=True,exist_ok=True)
    if not (stage/'source').exists():(stage/'source').symlink_to(root/'source',target_is_directory=True)
    plan=dict(read(prior/'plan.json'),name=f'MNI_third_seed_{seed}_constraints',seeds=[seed],
        baseline_stage=str(baseline),baseline_manifest_sha256=digest(baseline/'manifest.json'),
        data_reference_stage=config['baseline_stage'],new_training_runs=39,offline_peak_gib=10,
        conditional_matrix_seed=seed,stages=['Three baseline folds>=60%','Six bands/bands+edge comparisons',
        'Remaining33 only if at least one pilot arm improves mean paired Dice across all3 folds'])
    write(stage/'plan.json',plan);(stage/'run.sbatch').write_text((root/'run.sbatch').read_text())
    write(stage/'provenance.json',dict(search_manifest_sha256=digest(root/'manifest.json'),
        original_matrix_manifest_sha256=config['previous_matrix_manifest_sha256'],
        candidate_seed=seed,matching='Fresh medical initialization; exact paired baseline sampling and LR trajectory'))
    freeze(stage)
    return stage


def scheduler_preflight(stage,plan):
    if (stage/'preflight.json').exists():
        proof=read(stage/'preflight.json')
        if proof['operational_manifest_sha256']==digest(stage/'manifest.json'):return
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    result=subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)],capture_output=True,text=True,check=True)
    write(stage/'preflight.json',dict(status='passed',operational_manifest_sha256=digest(stage/'manifest.json'),
        quota=quota(),scheduler=result.stdout+result.stderr,time=time.time()))


def own_jobs(stage,current):
    known={p.stem for p in (stage/'submissions').glob('*.json')}
    return [r for r in current if r[0] in known]


def prior_ready(prior,current):
    cohort=read(prior/'cohort.json');rows=matrix.completed(prior,cohort['tasks'])
    return not own_jobs(prior,current) and len(rows)==len(cohort['tasks']),len(rows)


def register(root,stage,kind):
    path=root/'registry.json';registry=read(path) if path.exists() else dict(stages=[])
    entry=dict(path=str(stage),kind=kind,manifest_sha256=digest(stage/'manifest.json'))
    if entry not in registry['stages']:
        registry['stages'].append(entry);write(path,registry)


def clean_registered(root,current):
    path=root/'registry.json'
    for entry in read(path)['stages'] if path.exists() else []:
        stage=Path(entry['path'])
        if own_jobs(stage,current):continue
        plan=read(stage/'plan.json')
        if entry['kind']=='baseline':
            rows=matrix.baseline_records(stage,plan)
            baseline_cleanup(stage,plan,rows,plan['seeds'],set())
        elif (stage/'cohort.json').exists():
            matrix.cleanup(stage,matrix.completed(stage,read(stage/'cohort.json')['tasks']))


def final_analysis(root,candidate):
    config=read(root/'plan.json');prior=Path(config['previous_matrix_stage'])
    stages=[prior,candidate];cells=[];baselines=[]
    for stage in stages:
        cohort=read(stage/'cohort.json');plan=read(stage/'plan.json')
        rows=matrix.completed(stage,cohort['tasks']);matrix.report(stage,cohort,rows)
        cells.extend(read(stage/'comparison.json')['paired'])
        for row in cohort['baselines']:
            out=Path(plan['baseline_stage'])/'runs'/row['id'];ev=read(out/'evaluation.json')
            validation=[r for r in ev['records'] if r['split']=='validation']
            baselines.append(dict(**row,baseline_dice=row['validation_dice'],delta_dice=0.,
                per_class_dice={c:statistics.mean(r['per_class'][c]['dice'] for r in validation)
                    for c in ('CA1_3','subiculum','CA4_DG')},
                foreground_fp=sum(r['foreground_fp'] for r in validation),
                foreground_fn=sum(r['foreground_fn'] for r in validation)))
    cells=baselines+cells;seeds=sorted({r['seed'] for r in cells})
    if len(seeds)!=3 or len(cells)!=126 or len({(r['arm'],r['fold'],r['seed']) for r in cells})!=126:
        raise ValueError('Final analysis is not a complete3-fold3-seed14-arm matrix')
    grouped={}
    for arm in ARMS:
        rows=[r for r in cells if r['arm']==arm]
        if {(r['fold'],r['seed']) for r in rows}!={(f,s) for f in range(3) for s in seeds}:
            raise ValueError('Missing final fold/seed cell')
        grouped[arm]=dict(n=9,mean_dice=statistics.mean(r['validation_dice'] for r in rows),
            std_dice=statistics.stdev(r['validation_dice'] for r in rows),
            mean_paired_delta=statistics.mean(r['delta_dice'] for r in rows),
            per_class_mean_dice={c:statistics.mean(r['per_class_dice'][c] for r in rows)
                for c in ('CA1_3','subiculum','CA4_DG')},
            mean_foreground_fp=statistics.mean(r['foreground_fp'] for r in rows),
            mean_foreground_fn=statistics.mean(r['foreground_fn'] for r in rows),
            by_fold={str(f):statistics.mean(r['validation_dice'] for r in rows if r['fold']==f) for f in range(3)},
            by_seed={str(s):statistics.mean(r['validation_dice'] for r in rows if r['seed']==s) for s in seeds})
    write(root/'final_analysis.json',dict(status='complete',seeds=seeds,folds=[0,1,2],cells=cells,by_arm=grouped,
        search_history=read(root/'state.json')['history'],created_at=time.time(),
        caveat='Validation-selected seeds, including selection for constraint improvement. Descriptive paired results; folds/seeds are not independent external test observations. All rejected candidate results remain in the search history.'))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=['arm','fold','seed','updates','validation_dice','baseline_dice','delta_dice','foreground_fp','foreground_fn'])
    writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in cells)
    (root/'final_cells.csv').write_text(buffer.getvalue())


def tick(root,config,state,current):
    prior=Path(config['previous_matrix_stage']);ready,count=prior_ready(prior,current)
    if not ready:return dict(phase='waiting_for_two_seed_matrix',completed=count,total=78)
    clean_registered(root,current)
    seed=state['candidate_seed'];baseline=make_baseline(root,seed);register(root,baseline,'baseline')
    plan=baseline_verify(baseline);rows=matrix.baseline_records(baseline,plan)
    ops.deliver_base(baseline,plan,current,Path(config['base_source']))
    if own_jobs(baseline,current):return dict(phase='candidate_baselines',seed=seed)
    if (baseline/'failure.json').exists():return dict(phase='error',stage=str(baseline),failure=read(baseline/'failure.json'))
    selected=matrix.cohort_rows(rows,plan['threshold'],[seed])
    if selected is None:
        if next_baselines(plan,rows,[seed],set()):
            scheduler_preflight(baseline,plan);ops.OPS=baseline;ops.submit(baseline,plan)
            return dict(phase='candidate_baselines',seed=seed)
        state['history'].append(dict(seed=seed,outcome='baseline_below60_at1000',rows=rows,time=time.time()))
        state['candidate_seed']+=1;write(root/'state.json',state)
        return dict(phase='advancing_candidate',rejected_seed=seed)
    stage=make_matrix(root,seed,baseline);register(root,stage,'constraints')
    plan=matrix.verify(stage);cohort=matrix.resolve(stage,plan)
    if cohort is None:raise ValueError('Qualified candidate did not resolve')
    ops.deliver_base(stage,plan,current,Path(config['base_source']))
    if own_jobs(stage,current):return dict(phase='candidate_constraints',seed=seed)
    if (stage/'failure.json').exists():return dict(phase='error',stage=str(stage),failure=read(stage/'failure.json'))
    rows=matrix.completed(stage,cohort['tasks']);matrix.cleanup(stage,rows);matrix.report(stage,cohort,rows)
    decision=matrix.pilot_decision(cohort,rows,seed)
    if decision['status']=='not_improved':
        write(stage/'pilot_decision.json',decision)
        state['history'].append(dict(seed=seed,outcome='pilot_did_not_improve',decision=decision,time=time.time()))
        state['candidate_seed']+=1;write(root/'state.json',state)
        return dict(phase='advancing_candidate',rejected_seed=seed)
    if len(rows)==39:
        if state.get('selected_third_seed')!=seed:
            state['history'].append(dict(seed=seed,outcome='selected_full_matrix_complete',decision=decision,
                baselines=cohort['baselines'],time=time.time()))
        state['selected_third_seed']=seed;write(root/'state.json',state)
        final_analysis(root,stage)
        return dict(phase='complete',selected_seeds=[0,1,seed],folds=[0,1,2],cells=126)
    if not (stage/'preflight.json').exists():matrix.preflight(stage)
    ops.OPS=stage;ops.submit(stage,plan)
    return dict(phase='candidate_constraints',seed=seed,completed=len(rows),pilot=decision['status'])


def serve(root):
    config=verify_files(root)
    for field,sha in [('baseline_stage','baseline_manifest_sha256'),('previous_matrix_stage','previous_matrix_manifest_sha256')]:
        if digest(Path(config[field])/'manifest.json')!=config[sha]:raise ValueError('Parent stage identity changed')
    if digest(Path(config['base_source']))!=config['base_sha256']:raise ValueError('Base model changed')
    with (root/'service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if any(read(p/'control.json').get('pause') for p in
                       (root,Path(config['baseline_stage']),Path(config['previous_matrix_stage']))):return
                state=read(root/'state.json') if (root/'state.json').exists() else dict(candidate_seed=2,history=[])
                current=controller.queue()
                result=tick(root,config,state,current)
                write(root/'status.json',dict(**result,pid=os.getpid(),last_check=time.time(),laptop_required=False,quota=quota()))
                if result['phase']=='complete':
                    clean_registered(root,current)
                    # Stay available to retire results when the archive mirror returns.
                    if all((p.parent/'cleanup.json').exists() for p in (root/'candidates').glob('*/**/runs/*/archive_manifest.json')):return
            except Exception:
                write(root/'service_error.json',dict(time=time.time(),traceback=traceback.format_exc()))
            time.sleep(20)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['serve','run-baseline'])
    parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    (serve if args.command=='serve' else run_baseline)(args.stage.resolve())


if __name__=='__main__':main()
