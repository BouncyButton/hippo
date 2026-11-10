"""Preserve the screening/outer sweep and replace broad A/P refinement."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import ap82_resume as prior
from . import ap82_targeted_matrix as matrix

SCOPE='ap82_targeted_scope.json'
EVIDENCE='ap82_shared_tissue.json'
SOURCES=('ap82_shared_tissue.py','ap82_targeted_matrix.py','ap82_targeted_resume.py',
         'ap82_targeted_tests.py','run_ap82_targeted.sh')
REPLACED_JOBS=('697547','697548')
BASE_LOAD=prior.load
sha,write=prior.sha,prior.write


def extend_protocol(p, entries):
    result=dict(p,runs=p['runs']+entries)
    assert len({e['id'] for e in result['runs']})==len(result['runs'])
    result['main_runs']=len(result['runs'])
    result['interpretation']=('Completed 30-arm A/P screen and unchanged retained outer sweep, followed by '
        'at most 20 targeted, mirrored A/P refinements selected using shared-tissue swaps, interface '
        'geometry and foreground preservation. No automatic broad refinement or interaction sweep. '
        'Same 5%, fold0 seed0, from-scratch ordinary SwinUNETR, max75/StepLR20. '
        'Selection uses development validation data and is not independent replication.')
    return result


def load(root, full=True):
    assert not (root/'ap82_stage2.json').exists(), 'Original broad stage already generated; reconcile explicitly'
    assert not (root/'ap82_stage3.json').exists()
    p,base_scope=BASE_LOAD(root,full=full)
    scope=json.loads((root/SCOPE).read_text())
    assert scope['base_scope_sha256']==sha(root/prior.SCOPE)
    assert scope['evidence_sha256']==sha(root/EVIDENCE)
    for path,expected in scope['sources'].items():assert sha(root/path)==expected,path
    checks=json.loads((root/'ap82_targeted_local_checks.json').read_text())
    assert checks['passed'] and checks['source_hashes']==scope['source_hashes']
    assert scope['local_checks_sha256']==sha(root/'ap82_targeted_local_checks.json')
    evidence=json.loads((root/EVIDENCE).read_text())
    assert evidence['stage1_scope_sha256']==sha(root/prior.SCOPE)
    assert evidence['source_sha256']==sha(root/'source/scripts/band_focus/ap82_shared_tissue.py')
    entries,parent=matrix.targeted_entries(base_scope,evidence)
    assert scope['entries']==entries and scope['parent_id']==parent and scope['selection_rule']==matrix.RULE
    if full:
        for path,expected in evidence['artifacts'].items():assert sha(root/path)==expected,path
    return extend_protocol(p,entries),scope


def prepare(root):
    import torch
    from .train import completed_record,identity
    queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
    assert not set(REPLACED_JOBS)&set(queue), 'Wait for original workers to exit'
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p,base_scope=BASE_LOAD(root)
        assert not (root/'ap82_stage2.json').exists() and not (root/'ap82_stage3.json').exists()
        evidence=json.loads((root/EVIDENCE).read_text())
        entries,parent=matrix.targeted_entries(base_scope,evidence)
        completed={}
        for e in p['runs']:
            folder=root/'runs'/e['id']
            if completed_record(folder,identity(p,e)) is not None:completed[e['id']]=sha(folder/'completion.json')
        assert all(e['id'] in completed for e in base_scope['stage1'])
        resumes={}
        for path in (root/'runs').glob('*/latest.pt'):
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['identity']['protocol_sha256']==p['identity_sha']
            assert saved['scheduler']['step_size']==20
            resumes[path.parent.name]=dict(epoch=len(saved['history']),latest_sha256=sha(path),
                best_sha256=sha(path.parent/'best.pt') if (path.parent/'best.pt').exists() else None)
            del saved
        checks=json.loads((root/'ap82_targeted_local_checks.json').read_text())
        sources={f'source/scripts/band_focus/{n}':sha(root/'source/scripts/band_focus'/n) for n in SOURCES}
        assert checks['passed'] and checks['source_hashes']=={n:sources[f'source/scripts/band_focus/{n}'] for n in SOURCES}
        if not (root/SCOPE).exists():
            scope=dict(created_unix=time.time(),base_scope_sha256=sha(root/prior.SCOPE),
                evidence_sha256=sha(root/EVIDENCE),selection_rule=matrix.RULE,parent_id=parent,entries=entries,
                replaced_jobs=list(REPLACED_JOBS),completed_preserved=completed,resumes=resumes,
                replaced_stage2_count=44,removed_optional_stage3_max=32,
                sources=sources,source_hashes=checks['source_hashes'],
                local_checks_sha256=sha(root/'ap82_targeted_local_checks.json'))
            write(root/'ap82_targeted_previous_status.json',json.loads((root/'status.json').read_text()))
            write(root/SCOPE,scope)
        p,scope=load(root);value=update_report(p,root)
        write(root/'status.json',dict(state='prepared',current_stage=None,completed_runs=value['completed'],
            planned_runs=value['planned'],planned_max_runs=value['planned'],ap_completed=value['ap_completed'],
            active_scope_sha256=sha(root/SCOPE),preserved_resumes=scope['resumes']))
        print(json.dumps(dict(completed=value['completed'],planned=value['planned'],targeted=len(entries),
            parent_id=parent,preserved_resumes=resumes,scope_sha256=sha(root/SCOPE))),flush=True)


def phase_entries(root,p,scope,phase):
    assert phase==2
    return scope['entries'],sha(root/SCOPE)


def gate(root):
    # Reuse the already tested calibration and actual-model update gate.
    prior.load=load
    prior.phase_entries=phase_entries
    prior.gate(root,2)


def validate_gate(root,p,scope):
    prior.phase_entries=phase_entries
    prior.validate_gate(root,p,scope,2)


def update_report(p,root):
    from .report import report
    value=report(p,root)
    value.update(active_scope_sha256=sha(root/SCOPE),planned_known=len(p['runs']),planned_max=len(p['runs']),
        ap_completed=sum(r['objective'].startswith('ap82_') for r in value['rows']),
        ap_targeted_planned=sum(e['family']=='ap82_targeted' for e in p['runs']),
        ap_targeted_completed=sum(r['family']=='ap82_targeted' for r in value['rows']))
    write(root/'results.json',value)
    prior.retain(p,root,value['rows'])
    return value


def submit(root):
    from .retention import quota_free
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=root/'ap82_targeted_submission.json'
        if receipt.exists():print(receipt.read_text());return
        p,scope=load(root)
        queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
        assert not set(REPLACED_JOBS)&set(queue)
        for name,resume in scope['resumes'].items():
            assert sha(root/'runs'/name/'latest.pt')==resume['latest_sha256']
            if resume['best_sha256']:assert sha(root/'runs'/name/'best.pt')==resume['best_sha256']
        jobs=[]
        for part in (1,2):
            path=root/f'ap82_targeted_submission_part{part}.json'
            if path.exists():jobs.append(json.loads(path.read_text())['job_id']);continue
            attempt=root/f'ap82_targeted_submission_part{part}_attempt.json'
            assert not attempt.exists(),'Resolve unknown submission before retry'
            free=quota_free();assert free>=p['minimum_free_gib']
            command=['sbatch','--parsable',f'--job-name=hippo-ap-cut-{part}',
                     f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
            if part==2:command+=['--dependency=afterok:'+jobs[0]]
            command += [str(root/'source/scripts/band_focus/run_ap82_targeted.sh'),str(root)]
            value=dict(command=command,requested_unix=time.time(),scope_sha256=sha(root/SCOPE),free_gib=free)
            write(attempt,value);result=subprocess.run(command,capture_output=True,text=True)
            value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            write(root/f'ap82_targeted_submission_part{part}_response.json',value)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not job.isdigit():raise RuntimeError('Submission unconfirmed; inspect response')
            value['job_id']=job;write(path,value);jobs.append(job)
        value=dict(job_ids=jobs,scope_sha256=sha(root/SCOPE),planned=len(p['runs']),
            targeted=len(scope['entries']),parent_id=scope['parent_id'],max_epochs=75,step_size=20,
            fold=0,seed=0,fraction='5pct',optional_stage3=0)
        write(receipt,value);print(json.dumps(value),flush=True)


def worker(root):
    import torch
    from .worker import verify
    from .warmup import prepare_prefix,seed_run
    from .train import completed_record,identity
    from .retention import quota_free
    torch.set_num_threads(4);started=time.monotonic()
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(job_id=os.environ['SLURM_JOB_ID'],state='verifying',current_stage=None,
                   stages=[],started_unix=time.time(),active_scope_sha256=sha(root/SCOPE))
        write(root/'status.json',state)
        def execute(stage,action,extra):
            state['current_stage']=stage;write(root/'status.json',state)
            begin=time.monotonic();print('Starting '+stage,flush=True)
            command=[sys.executable,'-u','-m','scripts.band_focus.ap82_targeted_resume',action,'--root',str(root),*extra]
            with (root/'logs'/(stage+'.log')).open('a',buffering=1) as log:
                result=subprocess.run(command,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage,returncode=result.returncode,seconds=time.monotonic()-begin))
            write(root/'status.json',state)
            if result.returncode:raise RuntimeError(stage+' failed; inspect log')
        try:
            verify(root);p,scope=load(root);prepare_prefix(p,root)
            base,base_scope=BASE_LOAD(root,full=False)
            prior.validate_gate(root,base,base_scope,1)
            previous,previous_scope=prior.prior.load(root,full=False)
            prior.prior.validate_gate(previous,previous_scope,root)
            value=update_report(p,root)
            state.update(state='running',completed_runs=value['completed'],planned_runs=value['planned'],
                         planned_max_runs=value['planned'],ap_completed=value['ap_completed'])
            write(root/'status.json',state)
            checked=False
            for e in p['runs']:
                if completed_record(root/'runs'/e['id'],identity(p,e)) is not None:continue
                if time.monotonic()-started>22.5*3600:
                    state.update(state='awaiting_continuation',current_stage=None);write(root/'status.json',state);return
                if quota_free()<1.:raise RuntimeError('Less than 1 GiB free; resume checkpoints preserved')
                state['state']='running'
                if e.get('ap82_phase')==2 and not checked:
                    if not (root/'ap82_gate_s2.json').exists():execute('ap82_targeted_preflight','gate',[])
                    validate_gate(root,p,scope);checked=True
                seed_run(p,root,e);execute(e['id'],'train-one',['--run-id',e['id']])
                value=update_report(p,root)
                state.update(completed_runs=value['completed'],planned_runs=value['planned'],
                    planned_max_runs=value['planned'],ap_completed=value['ap_completed'])
                write(root/'status.json',state)
            execute('ap82_targeted_final_comparison','audit-final',[])
            state.update(state='complete',current_stage=None,finished_unix=time.time());write(root/'status.json',state)
            receipt=json.loads((root/'ap82_targeted_submission.json').read_text())
            queue=dict(line.split('|') for line in subprocess.check_output(['squeue','-u','3160552','-h','-o','%i|%T'],text=True).splitlines() if line.strip())
            for job in receipt['job_ids']:
                if job!=os.environ['SLURM_JOB_ID'] and queue.get(job)=='PENDING':subprocess.run(['scancel',job],check=True)
        except Exception as error:
            state.update(state='failed',error=repr(error),finished_unix=time.time());write(root/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','submit','gate','worker','train-one','verify','audit-final'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run-id');args=parser.parse_args()
    if args.action=='train-one':
        import torch
        from . import train
        p,_=load(args.root,full=False);e=next(e for e in p['runs'] if e['id']==args.run_id)
        if e.get('objective_version')==prior.matrix.VERSION:
            from .ap82_objectives import install
        else:
            from .winner_objectives import install
        install();torch.set_num_threads(4);train.run(p,args.root,e)
    elif args.action=='verify':
        p,scope=load(args.root);print(json.dumps(dict(planned=len(p['runs']),targeted=len(scope['entries']),parent=scope['parent_id'])))
    elif args.action=='audit-final':
        from .ap82_shared_tissue import audit
        load(args.root)
        audit(args.root,args.root/'ap82_targeted_comparisons.json',include_targeted=True)
    else:globals()[args.action](args.root)


if __name__=='__main__':main()
