"""Preserve prior evidence and run a frozen, prioritized winner refinement."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from . import asym32_resume as prior
from .winner_matrix import added_matrix, extend_protocol, retained_ids

SCOPE = 'winner_scope.json'
GATE = 'winner_calibration_preflight.json'
SOURCES = ('winner_matrix.py', 'winner_objectives.py', 'winner_resume.py', 'winner_tests.py', 'run_winner.sh')
sha, write = prior.sha, prior.write
REPLACED_JOBS = ('696517', '696518')


def load(root, *, full=True):
    previous, _ = prior.load(root, full=full)
    scope = json.loads((root/SCOPE).read_text())
    assert scope['base_scope_sha256'] == sha(root/prior.SCOPE)
    assert scope['protocol_identity'] == previous['identity_sha']
    assert scope['added_runs'] == added_matrix(previous['runs'])
    p = extend_protocol(previous, scope['completed_preserved'])
    assert scope['execution_order'] == [e['id'] for e in p['runs']]
    assert scope['retained_originals'] == retained_ids(previous['runs'])
    for name, expected in scope['sources'].items():
        assert sha(root/name) == expected, name
    checks = json.loads((root/'winner_local_checks.json').read_text())
    assert checks['passed'] and checks['new_objectives_checked'] == len(scope['added_runs'])
    for name, expected in checks['source_hashes'].items():
        assert sha(root/'source/scripts/band_focus'/name) == expected, name
    return p, scope


def update_report(p, root):
    from .report import report
    from .retention import retain_checkpoints
    value = report(p, root)
    value.update(active_scope_sha256=sha(root/SCOPE), winner_planned=sum(e['id'].startswith('w81_') for e in p['runs']),
        winner_completed=sum(e['objective'].startswith('w81_') for e in value['rows']),
        added_completed=sum(e['objective'].startswith('a32_') for e in value['rows']))
    write(root/'results.json', value)
    retain_checkpoints(p, root, value)
    return value


def prepare(root):
    import torch
    from .train import completed_record, identity
    from .warmup import prepare_prefix
    queue = subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
    assert not set(REPLACED_JOBS)&set(queue), 'Wait for replaced allocations to exit'
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, previous_scope = prior.load(root)
        prior.validate_gate(previous, previous_scope, root)
        prepare_prefix(previous, root)
        if not (root/SCOPE).exists():
            completed, resumes = {}, {}
            for entry in previous['runs']:
                folder = root/'runs'/entry['id']
                if completed_record(folder, identity(previous, entry)) is not None:
                    completed[entry['id']] = sha(folder/'completion.json')
                elif (folder/'latest.pt').exists():
                    state = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
                    assert state['identity'] == identity(previous, entry)
                    assert state['scheduler']['step_size'] == 20
                    resumes[entry['id']] = dict(epoch=len(state['history']),
                        latest_sha256=sha(folder/'latest.pt'), best_sha256=sha(folder/'best.pt'))
            p = extend_protocol(previous, completed)
            active = {e['id'] for e in p['runs']}
            scope = dict(created_unix=time.time(), replaced_jobs=list(REPLACED_JOBS),
                reason='User requested finishing the remaining asymmetric runs, retaining useful penalty/endpoint/shell comparisons, and replacing lower-priority unfinished arms with sensible winner combinations, most promising first.',
                protocol_identity=p['identity_sha'], base_scope_sha256=sha(root/prior.SCOPE),
                added_runs=added_matrix(previous['runs']), execution_order=[e['id'] for e in p['runs']],
                completed_preserved=completed, resumes=resumes,
                retained_originals=retained_ids(previous['runs']),
                skipped_unfinished=[e['id'] for e in previous['runs'] if e['id'] not in active],
                pending_asym_first=[e['id'] for e in previous['runs'] if e['id'].startswith('a32_') and e['id'] not in completed],
                sources={f'source/scripts/band_focus/{n}':sha(root/'source/scripts/band_focus'/n) for n in SOURCES})
            write(root/SCOPE, scope)
            write(root/'winner_previous_status.json', json.loads((root/'status.json').read_text()))
        p, scope = load(root)
        value = update_report(p, root)
        write(root/'status.json', dict(state='reprioritized', completed_runs=value['completed'], planned_runs=p['main_runs'],
            current_stage=None, winner_planned=len(scope['added_runs']), winner_completed=0,
            active_scope_sha256=sha(root/SCOPE), preserved_resumes=scope['resumes']))
        print(json.dumps(dict(completed=value['completed'], planned=p['main_runs'], new=len(scope['added_runs']),
            pending_asym_first=scope['pending_asym_first'], retained=len(scope['retained_originals']),
            skipped=len(scope['skipped_unfinished']), resumes=scope['resumes'], scope_sha256=sha(root/SCOPE))),flush=True)


def gate(root):
    """All-new-arm gradients/geometry and ten-case calibration precede training."""
    import torch
    from . import train
    from .geometry import Targets
    from .winner_objectives import components, install
    install()
    from scripts.depth_supervision.utils import restore_rng, seed_all
    from scripts.depth_supervision.losses import combined_loss
    torch.set_num_threads(4)
    seed_all(0)
    p, scope = load(root, full=False)
    state = torch.load(root/'shared_warmup/latest.pt', map_location='cpu', weights_only=False)
    warmup_sha, scope_sha = sha(root/'shared_warmup/latest.pt'), sha(root/SCOPE)
    device = torch.device('cuda')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = train.model_factory(device)
    assert sum(x.numel() for x in model.parameters()) == p['model_parameters']
    model.load_state_dict(state['model']); model.eval()
    patients = train.Patients(train._load_pkl_dataframe(p['pkl']),p['splits']['5pct']['train'],p['native_geometry'])
    assert len(patients) == 10
    index_path = root/'winner_calibration_progress.json'
    progress = (json.loads(index_path.read_text()) if index_path.exists() else
                dict(scope_sha256=scope_sha, warmup_sha256=warmup_sha, artifacts={}))
    assert progress['scope_sha256'] == scope_sha and progress['warmup_sha256'] == warmup_sha
    for entry in scope['added_runs']:
        folder = root/'calibration'/entry['id']; path = folder/'record.json'
        if entry['id'] in progress['artifacts']:
            assert sha(path) == progress['artifacts'][entry['id']]
            continue
        restore_rng(state['rng'])
        cache = Targets(root,p,entry)
        result = train.calibrate(model,patients,cache,device,p['calibration'],entry)
        assert len(result['cases']) == 10
        assert all(math.isfinite(v) and v >= 0 for v in result['coefficients'].values())
        for case in result['cases']:
            assert all(math.isfinite(v) for v in case['losses'].values())
            assert all(math.isfinite(v) for v in case['gradient_gram'].values())
        record = dict(identity=train.identity(p,entry), warmup_sha256=warmup_sha,
                      calibration=result, geometry=cache.statistics(), scope_sha256=scope_sha)
        if path.exists():
            assert json.loads(path.read_text()) == record, 'Unindexed calibration changed: '+entry['id']
        folder.mkdir(parents=True,exist_ok=True); write(path,record)
        progress['artifacts'][entry['id']] = sha(path); write(index_path,progress)
        print(json.dumps(dict(stage='calibrated_added',id=entry['id'],completed=len(progress['artifacts']),
                             planned=len(scope['added_runs']))),flush=True)
    # Exercise actual encoder/decoder updates for the main interactions, then discard them.
    representative = [scope['added_runs'][i] for i in (0,5,8,-1)]
    item = patients[0]; labels = item['label'][None].to(device)
    for entry in representative:
        model.load_state_dict(state['model']);model.train();restore_rng(state['rng'])
        model.zero_grad(set_to_none=True)
        with train.autocast(device):logits=model(item['image'][None].to(device))
        values=components(logits,labels,entry,Targets(root,p,entry).get(item,device))
        record=json.loads((root/'calibration'/entry['id']/'record.json').read_text())
        loss=combined_loss(values,record['calibration']['coefficients'])
        assert torch.isfinite(loss);loss.backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
        watched={n:(q,q.detach().clone()) for n,q in model.named_parameters()
                 if n in ('swinViT.patch_embed.proj.weight','out.conv.conv.weight')}
        assert len(watched)==2
        optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        optimizer.step();assert all(not torch.equal(q,before) for q,before in watched.values())
        del optimizer,logits,values,loss,watched
    value=dict(passed=True,scope_sha256=scope_sha,warmup_sha256=warmup_sha,
        objectives_checked=len(progress['artifacts']),training_cases=10,parameters_updated=True,
        model_update_variants=[e['id'] for e in representative],ordinary_swinunetr=True,
        artifacts=progress['artifacts'],source_hashes=scope['sources'])
    write(root/GATE,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('artifacts','source_hashes')}),flush=True)



def validate_gate(p, scope, root):
    value=json.loads((root/GATE).read_text())
    assert value['passed'] and value['parameters_updated'] and value['training_cases']==10
    assert value['scope_sha256']==sha(root/SCOPE) and value['warmup_sha256']==sha(root/'shared_warmup/latest.pt')
    assert value['objectives_checked']==len(scope['added_runs'])
    assert value['source_hashes']==scope['sources']
    assert set(value['artifacts'])=={e['id'] for e in scope['added_runs']}
    for name,expected in value['artifacts'].items():
        assert sha(root/'calibration'/name/'record.json')==expected
    return value


def submit(root):
    from .retention import quota_free
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=root/'winner_submission.json'
        if receipt.exists(): print(receipt.read_text()); return
        p,scope=load(root)
        for name,expected in scope['completed_preserved'].items():
            assert sha(root/'runs'/name/'completion.json')==expected
        for name,resume in scope['resumes'].items():
            assert sha(root/'runs'/name/'latest.pt')==resume['latest_sha256']
            assert sha(root/'runs'/name/'best.pt')==resume['best_sha256']
        jobs=[]
        for part in (1,2):
            path=root/f'winner_submission_part{part}.json'
            if path.exists():
                jobs.append(json.loads(path.read_text())['job_id']); continue
            attempt=root/f'winner_submission_part{part}_attempt.json'
            assert not attempt.exists(),'Resolve previous submission attempt before retrying'
            queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
            assert set(queue)<=set(jobs),repr(queue)
            free=quota_free(); assert free>=p['minimum_free_gib']
            command=['sbatch','--parsable',f'--job-name=hippo-w81-{part}',
                f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
            if part==2: command.append('--dependency=afterok:'+jobs[0])
            command += [str(root/'source/scripts/band_focus/run_winner.sh'),str(root)]
            value=dict(command=command,part=part,scope_sha256=sha(root/SCOPE),requested_unix=time.time(),
                free_gib=free,new=len(scope['added_runs']),planned=p['main_runs'],max_epochs=75,step_size=20)
            write(attempt,value)
            result=subprocess.run(command,cwd=root/'source',capture_output=True,text=True)
            value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            write(root/f'winner_submission_part{part}_response.json',value)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not job.isdigit(): raise RuntimeError('Submission unconfirmed; inspect response')
            value['job_id']=job; write(path,value); jobs.append(job)
        value=dict(job_ids=jobs,scope_sha256=sha(root/SCOPE),new=len(scope['added_runs']),planned=p['main_runs'],
            completed_preserved=len(scope['completed_preserved']),remaining_asym_first=len(scope['pending_asym_first']),
            retained=len(scope['retained_originals']),skipped=len(scope['skipped_unfinished']),
            max_epochs=75,step_size=20,fold=0,seed=0,fraction='5pct')
        write(receipt,value); print(json.dumps(value),flush=True)


def worker(root):
    import torch
    from .worker import verify
    from .warmup import prepare_prefix,seed_run
    from .train import completed_record,identity
    from .retention import quota_free
    torch.set_num_threads(4)
    started=time.monotonic()
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(job_id=os.environ['SLURM_JOB_ID'],state='verifying',stages=[],started_unix=time.time(),
                   active_scope_sha256=sha(root/SCOPE))
        write(root/'status.json',state)
        def execute(stage,action,run_id=None):
            state['current_stage']=stage; write(root/'status.json',state)
            print('Starting '+stage,flush=True); begin=time.monotonic()
            command=[sys.executable,'-u','-m','scripts.band_focus.winner_resume',action,'--root',str(root)]
            if run_id: command += ['--run-id',run_id]
            with (root/'logs'/(stage+'.log')).open('a',buffering=1) as log:
                result=subprocess.run(command,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage,returncode=result.returncode,seconds=time.monotonic()-begin))
            write(root/'status.json',state)
            if result.returncode: raise RuntimeError(stage+' failed; inspect log')
        try:
            verify(root)
            p,scope=load(root); prepare_prefix(p,root)
            previous,previous_scope=prior.load(root,full=False)
            prior.validate_gate(previous,previous_scope,root)
            result=update_report(p,root)
            state.update(state='running',planned_runs=p['main_runs'],winner_planned=len(scope['added_runs']),
                completed_runs=result['completed'],winner_completed=result['winner_completed'],added_completed=result['added_completed'])
            write(root/'status.json',state)
            gate_checked=False
            for entry in p['runs']:
                if completed_record(root/'runs'/entry['id'],identity(p,entry)) is not None: continue
                if time.monotonic()-started>22.5*3600:
                    state.update(state='awaiting_continuation',current_stage=None,finished_unix=time.time())
                    write(root/'status.json',state); return
                if quota_free()<1.: raise RuntimeError('Less than 1 GiB free; all resume checkpoints preserved')
                if entry['id'].startswith('w81_') and not gate_checked:
                    if not (root/GATE).exists(): execute('winner_preflight','gate')
                    validate_gate(p,scope,root); gate_checked=True
                seed_run(p,root,entry)
                execute(entry['id'],'train-one',entry['id'])
                result=update_report(p,root)
                state.update(completed_runs=result['completed'],winner_completed=result['winner_completed'],added_completed=result['added_completed'])
                write(root/'status.json',state)
            assert result['completed']==p['main_runs']
            state.update(state='complete',current_stage=None,finished_unix=time.time());write(root/'status.json',state)
            receipt=json.loads((root/'winner_submission.json').read_text())
            queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i|%T'],text=True)
            queued=dict(line.split('|') for line in queue.splitlines() if line.strip())
            for job in receipt['job_ids']:
                if job!=os.environ['SLURM_JOB_ID'] and queued.get(job)=='PENDING': subprocess.run(['scancel',job],check=True)
        except Exception as error:
            state.update(state='failed',error=repr(error),finished_unix=time.time());write(root/'status.json',state)
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','submit','gate','worker','train-one'))
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.action=='train-one':
        import torch
        from .train import run
        from .winner_objectives import install
        install(); torch.set_num_threads(4)
        p,_=load(args.root,full=False)
        entry=next(e for e in p['runs'] if e['id']==args.run_id)
        run(p,args.root,entry)
    else: globals()[args.action](args.root)


if __name__=='__main__': main()
