"""Immutable staged A/P amendment; preserve evidence and resume exact checkpoints."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from . import winner_resume as prior
from . import ap82_matrix as matrix

SCOPE = 'ap82_scope.json'
SOURCES = ('ap82_matrix.py','ap82_geometry.py','ap82_objectives.py','ap82_resume.py','ap82_tests.py','run_ap82.sh')
REPLACED_JOBS = ('697145','697146')
sha, write = prior.sha, prior.write


def load(root, full=True):
    previous, _ = prior.load(root, full=full)
    scope = json.loads((root/SCOPE).read_text())
    assert scope['base_scope_sha256'] == sha(root/prior.SCOPE)
    for path, expected in scope['sources'].items(): assert sha(root/path) == expected, path
    checks = json.loads((root/'ap82_local_checks.json').read_text())
    assert checks['passed'] and checks['stage1_count'] == 30 and checks['stage2_count'] == 44
    for name, expected in checks['source_hashes'].items():
        assert sha(root/'source/scripts/band_focus'/name) == expected
    assert scope['stage1'] == matrix.stage1(scope['bases'], scope['coefficients'])
    assert scope['retained_ids'] == matrix.retained_ids()
    old = {e['id']:e for e in previous['runs']}
    assert scope['bases'] == dict(winner=old[matrix.ANCHOR], dice=old['dice'])
    for name, expected in scope['base_calibrations'].items():
        assert sha(root/'calibration'/name/'record.json') == expected
    original = [old[n] for n in scope['preserved_order']]
    original += [old[n] for n in scope['retained_ids'] if n not in scope['completed_preserved']]
    p = dict(previous, runs=original[:len(scope['preserved_order'])]+scope['stage1']+original[len(scope['preserved_order']):])
    p['interpretation'] = ('Adaptive staged A/P study: frozen 81.112% outer objective or Dice alone, '
        'training-only actual-interface bands and independent 1+3 mm endpoints. Same 5%, fold0 seed0, '
        'max75/StepLR20; no architecture or data change. Stages two/three explicitly use development '
        'validation results; independent replication is required for generalization claims.')
    second = root/'ap82_stage2.json'
    if second.exists():
        value = json.loads(second.read_text())
        assert value['scope_sha256'] == sha(root/SCOPE)
        parent = matrix.select_parent(scope['stage1'], value['decision_rows'])
        assert value['parent_id'] == parent['id']
        assert value['entries'] == matrix.stage2(scope['bases'], scope['coefficients'], parent)
        p['runs'] += value['entries']
        third = root/'ap82_stage3.json'
        if third.exists():
            last = json.loads(third.read_text())
            assert last['stage2_sha256'] == sha(second)
            entries, factors = matrix.stage3(scope['bases'],scope['coefficients'],parent,value['entries'],last['decision_rows'])
            assert last['entries'] == entries and last['positive_factors'] == factors
            p['runs'] += entries
    assert len({e['id'] for e in p['runs']}) == len(p['runs'])
    p['main_runs'] = len(p['runs'])
    return p, scope


def retain(p, root, rows):
    """Keep controls, fixed winner, top five and the best A/P arm per background."""
    from .train import completed_record, identity
    keep = {e['id'] for e in p['runs'] if e['family'] == 'control'} | {matrix.ANCHOR}
    keep |= {r['objective'] for r in sorted(rows,key=matrix.score_key)[:5]}
    lookup = {e['id']:e for e in p['runs']}
    for bg in matrix.BACKGROUNDS:
        subset = [r for r in rows if lookup[r['objective']].get('background') == bg]
        if subset: keep.add(min(subset,key=matrix.score_key)['objective'])
    for row in rows:
        name = row['objective']; folder = root/'runs'/name; path = folder/'best.pt'
        if name in keep or not path.exists(): continue
        if folder.is_symlink() or path.is_symlink(): raise ValueError('Unexpected checkpoint symlink')
        record = completed_record(folder, identity(p,lookup[name]))
        assert record is not None
        if 'best.pt' in record['artifacts']:
            artifact = {'best.pt':record['artifacts'].pop('best.pt')}
            assert sha(path) == artifact['best.pt']
            write(folder/'checkpoint_retired.json',dict(identity=record['identity'],artifacts=artifact,
                reason='Outside fixed controls, frozen winner, top five and best A/P background; metrics/predictions retained',unix=time.time()))
            record['retired_artifacts'] = artifact; write(folder/'completion.json',record)
        path.unlink()
    write(root/'checkpoint_retention.json',dict(retained=sorted(keep),completed=len(rows),
        policy='Fixed controls, frozen outer winner, top five, best AP arm for each background; all active/parked resumes preserved.'))


def update_report(p, root):
    from .report import report
    value = report(p, root)
    value.update(active_scope_sha256=sha(root/SCOPE),
        ap_completed=sum(r['objective'].startswith('ap82_') for r in value['rows']),
        ap_stage2_selected=(root/'ap82_stage2.json').exists(), ap_stage3_selected=(root/'ap82_stage3.json').exists())
    value['planned_known'] = len(p['runs'])
    value['planned'] = len(p['runs'])+(0 if value['ap_stage2_selected'] else 44)
    value['planned_max'] = value['planned']+(0 if value['ap_stage3_selected'] else 32)
    write(root/'results.json',value); retain(p,root,value['rows'])
    return value


def prepare(root):
    import torch
    from .train import completed_record, identity
    queue = subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
    assert not set(REPLACED_JOBS)&set(queue), 'Old jobs must exit before amendment'
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous, old_scope = prior.load(root)
        prior.validate_gate(previous,old_scope,root)
        if not (root/SCOPE).exists():
            old = {e['id']:e for e in previous['runs']}
            assert set(matrix.retained_ids()) <= set(old)
            completed = {}
            for e in previous['runs']:
                folder = root/'runs'/e['id']
                if completed_record(folder,identity(previous,e)) is not None:
                    completed[e['id']] = sha(folder/'completion.json')
            assert matrix.ANCHOR in completed and 'dice' in completed
            calpath = root/'calibration'/matrix.ANCHOR/'record.json'
            calibration = json.loads(calpath.read_text())
            assert calibration['identity'] == identity(previous,old[matrix.ANCHOR])
            coefficients = dict(winner=calibration['calibration']['coefficients'], dice={})
            bases = dict(winner=old[matrix.ANCHOR], dice=old['dice'])
            resumes = {}
            for path in (root/'runs').glob('*/latest.pt'):
                saved = torch.load(path,map_location='cpu',weights_only=False)
                assert saved['identity']['protocol_sha256'] == previous['identity_sha']
                assert saved['scheduler']['step_size'] == 20
                resumes[path.parent.name] = dict(epoch=len(saved['history']),latest_sha256=sha(path),
                    best_sha256=sha(path.parent/'best.pt') if (path.parent/'best.pt').exists() else None)
                del saved
            selected = set(completed)|set(matrix.retained_ids())
            scope = dict(created_unix=time.time(),base_scope_sha256=sha(root/prior.SCOPE),
                replaced_jobs=list(REPLACED_JOBS),completed_preserved=completed,preserved_order=list(completed),
                retained_ids=matrix.retained_ids(),bases=bases,coefficients=coefficients,
                base_calibrations={matrix.ANCHOR:sha(calpath)},stage1=matrix.stage1(bases,coefficients),
                skipped_unfinished=[e['id'] for e in previous['runs'] if e['id'] not in selected],resumes=resumes,
                stage2_rule='Best mixed stage-one winner-background macro Dice; tie HD95 then identifier. Mirror its settings on both backgrounds; 44 block variations.',
                stage3_rule='Best value per block with >=0.0005 improvement over matched AP parent; at least two blocks; at most 16 interactions mirrored on both backgrounds (32 arms).',
                sources={f'source/scripts/band_focus/{n}':sha(root/'source/scripts/band_focus'/n) for n in SOURCES})
            write(root/'ap82_previous_status.json',json.loads((root/'status.json').read_text()))
            write(root/SCOPE,scope)
        p,scope = load(root); value = update_report(p,root)
        status = dict(state='prepared',current_stage=None,completed_runs=value['completed'],planned_runs=value['planned'],
            planned_max_runs=value['planned_max'],ap_completed=value['ap_completed'],active_scope_sha256=sha(root/SCOPE))
        write(root/'status.json',status)
        print(json.dumps(dict(status,retained_pending=sum(n not in scope['completed_preserved'] for n in scope['retained_ids']),
            skipped=len(scope['skipped_unfinished']),preserved_resumes=scope['resumes'])),flush=True)


def phase_entries(root,p,scope,phase):
    entries = [e for e in p['runs'] if e.get('ap82_phase') == phase]
    path = root/SCOPE if phase == 1 else root/f'ap82_stage{phase}.json'
    return entries,sha(path)


def gate(root, phase):
    import torch
    from . import train
    from .ap82_geometry import Targets
    from .ap82_objectives import install, components
    from scripts.depth_supervision.utils import restore_rng, seed_all
    from scripts.depth_supervision.losses import combined_loss
    install(); torch.set_num_threads(4); seed_all(0)
    p,scope = load(root,full=False); entries,phase_sha = phase_entries(root,p,scope,phase)
    assert entries
    state = torch.load(root/'shared_warmup/latest.pt',map_location='cpu',weights_only=False)
    warmup_sha = sha(root/'shared_warmup/latest.pt'); device = torch.device('cuda')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = train.model_factory(device); model.load_state_dict(state['model']);model.eval()
    assert sum(q.numel() for q in model.parameters()) == p['model_parameters']
    patients = train.Patients(train._load_pkl_dataframe(p['pkl']),p['splits']['5pct']['train'],p['native_geometry'])
    assert len(patients) == 10
    index = root/f'ap82_calibration_progress_s{phase}.json'
    progress = json.loads(index.read_text()) if index.exists() else dict(phase_sha256=phase_sha,warmup_sha256=warmup_sha,artifacts={})
    assert progress['phase_sha256'] == phase_sha and progress['warmup_sha256'] == warmup_sha
    for e in entries:
        folder = root/'calibration'/e['id']; path = folder/'record.json'
        if e['id'] in progress['artifacts']:
            assert sha(path) == progress['artifacts'][e['id']]; continue
        restore_rng(state['rng']); cache = Targets(root,p,e)
        result = train.calibrate(model,patients,cache,device,p['calibration'],e)
        assert len(result['cases']) == 10
        for name,coefficient in e['base_coefficients'].items():
            assert result['coefficients'][name] == coefficient
        assert all(math.isfinite(v) and v >= 0 for v in result['coefficients'].values())
        for stats in cache.stats.values():
            assert stats['band_anterior'] > 0 and stats['band_posterior'] > 0
            assert all(n > 0 for n in stats['endpoint_pairs'].values())
        record = dict(identity=train.identity(p,e),warmup_sha256=warmup_sha,phase_sha256=phase_sha,
                      calibration=result,geometry=cache.statistics())
        if path.exists(): assert json.loads(path.read_text()) == record
        folder.mkdir(parents=True,exist_ok=True);write(path,record)
        progress['artifacts'][e['id']] = sha(path);write(index,progress)
        print(json.dumps(dict(stage='ap_calibrated',phase=phase,id=e['id'],completed=len(progress['artifacts']),planned=len(entries))),flush=True)
    representative = entries[:2]
    if phase == 1:
        representative += [next(e for e in entries if e['background']=='winner' and e['ap82']['band_share']==s) for s in (0.,1.)]
    item = patients[0];labels=item['label'][None].to(device)
    for e in representative:
        model.load_state_dict(state['model']);model.train();restore_rng(state['rng']);model.zero_grad(set_to_none=True)
        with train.autocast(device): logits=train.forward_item(model,item,device)
        values=components(logits,labels,e,Targets(root,p,e).get(item,device))
        coefficients=json.loads((root/'calibration'/e['id']/'record.json').read_text())['calibration']['coefficients']
        loss=combined_loss(values,coefficients);assert torch.isfinite(loss);loss.backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
        watched={n:(q,q.detach().clone()) for n,q in model.named_parameters()
                 if n in ('swinViT.patch_embed.proj.weight','out.conv.conv.weight')}
        assert len(watched)==2
        optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        optimizer.step();assert all(not torch.equal(q,before) for q,before in watched.values())
        del optimizer,watched,logits,values,loss
    value=dict(passed=True,phase=phase,phase_sha256=phase_sha,warmup_sha256=warmup_sha,training_cases=10,
        objectives_checked=len(entries),parameters_updated=True,model_update_variants=[e['id'] for e in representative],
        sources=scope['sources'],artifacts=progress['artifacts'])
    write(root/f'ap82_gate_s{phase}.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('sources','artifacts')}),flush=True)


def validate_gate(root,p,scope,phase):
    entries,phase_sha=phase_entries(root,p,scope,phase)
    value=json.loads((root/f'ap82_gate_s{phase}.json').read_text())
    assert value['passed'] and value['parameters_updated'] and value['training_cases']==10
    assert value['sources']==scope['sources'] and value['phase_sha256']==phase_sha
    assert value['warmup_sha256']==sha(root/'shared_warmup/latest.pt')
    assert set(value['artifacts'])=={e['id'] for e in entries}
    for name,expected in value['artifacts'].items():assert sha(root/'calibration'/name/'record.json')==expected


def decision_rows(rows):
    return [{k:r[k] for k in ('objective','macro_dice','hd95_mm')} for r in rows if r['objective'].startswith('ap82_')]


def extend(root,scope,rows):
    second=root/'ap82_stage2.json'
    if not second.exists():
        evidence=decision_rows(rows);parent=matrix.select_parent(scope['stage1'],evidence)
        write(second,dict(scope_sha256=sha(root/SCOPE),created_unix=time.time(),parent_id=parent['id'],
            decision_rows=evidence,entries=matrix.stage2(scope['bases'],scope['coefficients'],parent)))
        return True
    if not (root/'ap82_stage3.json').exists():
        value=json.loads(second.read_text());parent=matrix.select_parent(scope['stage1'],value['decision_rows'])
        evidence=decision_rows(rows)
        entries,factors=matrix.stage3(scope['bases'],scope['coefficients'],parent,value['entries'],evidence)
        write(root/'ap82_stage3.json',dict(stage2_sha256=sha(second),created_unix=time.time(),
            decision_rows=evidence,positive_factors=factors,entries=entries))
        return True
    return False


def submit(root):
    from .retention import quota_free
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=root/'ap82_submission.json'
        if receipt.exists(): print(receipt.read_text());return
        p,scope=load(root)
        queue=subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
        assert not set(REPLACED_JOBS)&set(queue)
        for name,expected in scope['completed_preserved'].items():
            assert sha(root/'runs'/name/'completion.json')==expected
        for name,resume in scope['resumes'].items():
            assert sha(root/'runs'/name/'latest.pt')==resume['latest_sha256']
            if resume['best_sha256'] is not None:
                assert sha(root/'runs'/name/'best.pt')==resume['best_sha256']
        jobs=[]
        for part in (1,2):
            path=root/f'ap82_submission_part{part}.json'
            if path.exists():jobs.append(json.loads(path.read_text())['job_id']);continue
            attempt=root/f'ap82_submission_part{part}_attempt.json'
            assert not attempt.exists(),'Resolve any unknown submission outcome before retrying'
            free=quota_free();assert free>=p['minimum_free_gib']
            command=['sbatch','--parsable',f'--job-name=hippo-ap82-{part}',f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
            if part==2:command.append('--dependency=afterok:'+jobs[0])
            command += [str(root/'source/scripts/band_focus/run_ap82.sh'),str(root)]
            value=dict(command=command,scope_sha256=sha(root/SCOPE),requested_unix=time.time(),free_gib=free)
            write(attempt,value);result=subprocess.run(command,capture_output=True,text=True)
            value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            write(root/f'ap82_submission_part{part}_response.json',value)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not job.isdigit():raise RuntimeError('Submission unconfirmed; inspect response')
            value['job_id']=job;write(path,value);jobs.append(job)
        receipt_value=dict(job_ids=jobs,scope_sha256=sha(root/SCOPE),stage1=30,stage2=44,stage3_max=32,
            completed_preserved=len(scope['completed_preserved']),retained_pending=sum(n not in scope['completed_preserved'] for n in scope['retained_ids']),
            skipped=len(scope['skipped_unfinished']),max_epochs=75,step_size=20,fold=0,seed=0,fraction='5pct')
        write(receipt,receipt_value);print(json.dumps(receipt_value),flush=True)


def worker(root):
    import torch
    from .worker import verify
    from .warmup import prepare_prefix,seed_run
    from .train import completed_record,identity
    from .retention import quota_free
    torch.set_num_threads(4);started=time.monotonic()
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(job_id=os.environ['SLURM_JOB_ID'],state='verifying',current_stage=None,stages=[],started_unix=time.time())
        write(root/'status.json',state)
        def execute(stage,action,extra):
            state['current_stage']=stage;write(root/'status.json',state)
            begin=time.monotonic();print('Starting '+stage,flush=True)
            command=[sys.executable,'-u','-m','scripts.band_focus.ap82_resume',action,'--root',str(root),*extra]
            with (root/'logs'/(stage+'.log')).open('a',buffering=1) as log:
                result=subprocess.run(command,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage,returncode=result.returncode,seconds=time.monotonic()-begin))
            write(root/'status.json',state)
            if result.returncode:raise RuntimeError(stage+' failed; inspect log')
        try:
            verify(root);p,scope=load(root);prepare_prefix(p,root)
            previous,previous_scope=prior.load(root,full=False);prior.validate_gate(previous,previous_scope,root)
            checked=set()
            while True:
                p,scope=load(root,full=False);value=update_report(p,root)
                state.update(state='running',completed_runs=value['completed'],planned_runs=value['planned'],
                    planned_max_runs=value['planned_max'],ap_completed=value['ap_completed'],active_scope_sha256=sha(root/SCOPE))
                write(root/'status.json',state)
                for e in p['runs']:
                    if completed_record(root/'runs'/e['id'],identity(p,e)) is not None:continue
                    if time.monotonic()-started>22.5*3600:
                        state.update(state='awaiting_continuation',current_stage=None);write(root/'status.json',state);return
                    if quota_free()<1.:raise RuntimeError('Less than 1 GiB free; resume checkpoints preserved')
                    phase=e.get('ap82_phase')
                    if phase and phase not in checked:
                        if not (root/f'ap82_gate_s{phase}.json').exists():execute(f'ap82_preflight_s{phase}','gate',['--phase',str(phase)])
                        validate_gate(root,p,scope,phase);checked.add(phase)
                    seed_run(p,root,e);execute(e['id'],'train-one',['--run-id',e['id']])
                    value=update_report(p,root)
                    state.update(completed_runs=value['completed'],planned_runs=value['planned'],planned_max_runs=value['planned_max'],ap_completed=value['ap_completed'])
                    write(root/'status.json',state)
                if not extend(root,scope,value['rows']):break
            state.update(state='complete',current_stage=None,finished_unix=time.time());write(root/'status.json',state)
            receipt=json.loads((root/'ap82_submission.json').read_text())
            queue=dict(line.split('|') for line in subprocess.check_output(['squeue','-u','3160552','-h','-o','%i|%T'],text=True).splitlines() if line.strip())
            for job in receipt['job_ids']:
                if job!=os.environ['SLURM_JOB_ID'] and queue.get(job)=='PENDING':subprocess.run(['scancel',job],check=True)
        except Exception as error:
            state.update(state='failed',error=repr(error),finished_unix=time.time());write(root/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','submit','gate','worker','train-one'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run-id');parser.add_argument('--phase',type=int)
    args=parser.parse_args()
    if args.action=='train-one':
        import torch
        from . import train
        p,_=load(args.root,full=False);e=next(e for e in p['runs'] if e['id']==args.run_id)
        if e.get('objective_version')==matrix.VERSION:
            from .ap82_objectives import install
        else:
            from .winner_objectives import install
        install();torch.set_num_threads(4);train.run(p,args.root,e)
    elif args.action=='gate':gate(args.root,args.phase)
    else:globals()[args.action](args.root)


if __name__=='__main__':main()
