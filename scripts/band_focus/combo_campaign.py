"""Separate, immutable follow-up; never edit or interrupt the active campaign."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.depth_supervision.utils import atomic_json as write, digest as sha
from . import combo_matrix as matrix


def load(root, full=False):
    manifest = json.loads((root/'payload_manifest.json').read_text())
    for name, expected in manifest.items():
        assert sha(root/name) == expected, 'Changed payload: '+name
    p = json.loads((root/'protocol.json').read_text())
    content = dict(p); identity = content.pop('identity_sha')
    assert hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() == identity
    ref = json.loads((root/'anchor_reference.json').read_text())
    assert p['runs'] == matrix.entries(ref['objective'], ref['coefficients'])
    assert len(p['runs']) == p['main_runs'] == 22
    assert p['fold'] == p['seed'] == 0 and set(p['splits']) == {'5pct'}
    assert len(p['splits']['5pct']['train']) == 10 and len(p['splits']['5pct']['val']) == 52
    assert p['max_epochs'] == 75 and p['step_size'] == 20
    from .combo_replication import validate_inputs
    validate_inputs(json.loads((root/'replication_inputs.json').read_text()),p)
    checks = json.loads((root/'combo_local_checks.json').read_text())
    assert checks['passed']
    assert checks['source_hashes'] == {n: sha(root/'source/scripts/band_focus'/n) for n in matrix.SOURCES}
    if full:
        assert sha(p['pkl']) == p['pkl_sha256']
        for package, version in p['required_dependencies'].items():
            assert importlib.metadata.version(package) == version, package
    return p


def prepare(root, parent):
    from .ap82_targeted_resume import load as parent_load
    from .train import completed_record, identity
    from .warmup import prepare_prefix
    if (root/'protocol.json').exists():
        p = load(root, full=True)
        print(json.dumps(dict(state='already_prepared',runs=len(p['runs']),identity=p['identity_sha'])))
        return
    original, _ = parent_load(parent, full=False)
    base = next(e for e in original['runs'] if e['id'] == matrix.ANCHOR)
    folder = parent/'runs'/matrix.ANCHOR
    record = completed_record(folder, identity(original, base)); assert record is not None
    config = json.loads((folder/'config.json').read_text())
    assert config['objective'] == base and not config['auxiliary_branch'] and config['input_channels'] == 1
    coefficients = config['calibration']['coefficients']
    ref = dict(objective=base, coefficients=coefficients, best_epoch=record['best_epoch'],
               validation=record['validation'], config_sha256=sha(folder/'config.json'),
               metrics_sha256=sha(folder/'metrics.csv'), parent_protocol_identity=original['identity_sha'])
    write(root/'anchor_reference.json', ref)
    (root/'anchor_metrics.csv').write_bytes((folder/'metrics.csv').read_bytes())
    p = deepcopy(original)
    for key in ('identity_sha', 'recovery', 'reference_controls', 'source_sha256'):
        p.pop(key, None)
    p.update(name=root.name, predecessor_root=str(parent), predecessor_job=None,
             runs=matrix.entries(base, coefficients), main_runs=22,
             required_dependencies={k: importlib.metadata.version(k) for k in ('torch', 'monai', 'numpy', 'scipy')},
             interpretation='Development validation exploration: frozen 81.32397% outer objective plus SDF/edge/depth; one fold and seed, no independent replication.',
             loss_definitions=dict(
                 base='Exact frozen 3/2mm bands and 1/3mm independent endpoints of '+matrix.ANCHOR,
                 sdf='Original full-grid classwise anterior/posterior signed-distance loss, clipped at 8 native 1mm voxels.',
                 edge='Original signed foreground face-contrast squared error in two-step annotation boundary bands.',
                 depth='Original depth-balanced multiclass CE: 2mm bins, 8mm ROI, sparse-bin mass taper at 64 voxels.',
                 additions='All seven nonempty subsets; equal shares after gradient normalization, jointly calibrated to 2.5%, 5%, or 10% of frozen complete-base logit gradient.',
                 schedule='Exact common Dice-only epochs1-5; constraints ramp epochs6-10; max75, StepLR20 gamma0.5, original early stopping.'),
             calibration_policy='Freeze both outer coefficients; calibrate additions jointly on the same ten training cases at the common epoch-five checkpoint.',
             estimated_additional_peak_gib=3.,
             storage_budget='Retain matched control and top five checkpoints; keep all aggregate metrics, configurations, logs and validation predictions.')
    p['replication'] = dict(folds=[0,1,2],seeds=[0,1,2],fresh_runs=9,
        selection='Best validation Dice after incumbent and combination sweeps complete.',
        policy='Freeze loss recipe; repeat training-only calibration within each fold and seed.',
        queue_policy='One incumbent running job and one queued follow-up maximum; combinations and replication share one allocation.')
    from .combo_replication import validate_inputs
    validate_inputs(json.loads((root/'replication_inputs.json').read_text()),p)
    p['source_sha256'] = {str(f.relative_to(root)): sha(f) for f in sorted((root/'source').rglob('*'))
                          if f.is_file() and '__pycache__' not in f.parts}
    p['identity_sha'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    write(root/'protocol.json', p)
    frozen = dict(p['source_sha256'])
    for n in ('protocol.json', 'anchor_reference.json', 'anchor_metrics.csv', 'combo_local_checks.json','replication_inputs.json'):
        frozen[n] = sha(root/n)
    write(root/'payload_manifest.json', frozen)
    load(root, full=True)
    prepare_prefix(p, root)
    (root/'logs').mkdir(exist_ok=True)
    write(root/'status.json', dict(state='prepared', completed_runs=0, planned_runs=22, current_stage=None))
    summary = dict(root=str(root), anchor=matrix.ANCHOR, anchor_dice=ref['validation']['macro_dice'],
                   coefficients=coefficients, runs=[dict(id=e['id'], **e['combo']) for e in p['runs']],
                   fold=0, seed=0, fraction='5pct', max_epochs=75, step_size=20,
                   replication=p['replication'],total_new_training_runs=31,
                   parent_root=str(parent), protocol_identity=p['identity_sha'])
    write(root/'plan_summary.json', summary)
    print(json.dumps(summary), flush=True)


def validate_gate(root, p):
    gate = json.loads((root/'combo_gate.json').read_text())
    assert gate['passed'] and gate['parameters_updated'] and gate['training_cases'] == 10
    assert gate['protocol_identity'] == p['identity_sha']
    assert gate['warmup_sha256'] == sha(root/'shared_warmup/latest.pt')
    assert set(gate['artifacts']) == {e['id'] for e in p['runs']}
    for name, expected in gate['artifacts'].items():
        assert sha(root/'calibration'/name/'record.json') == expected


def gate(root):
    import torch
    from . import combo_train as train
    from .combo_objectives import Targets, components, select_weights
    from scripts.native_branch_constraints.objectives import gradient_record
    from scripts.depth_supervision.utils import seed_all, restore_rng
    from scripts.depth_supervision.losses import combined_loss
    p = load(root); torch.set_num_threads(4); seed_all(0)
    if (root/'combo_gate.json').exists():
        validate_gate(root, p); return
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    state = torch.load(root/'shared_warmup/latest.pt', map_location='cpu', weights_only=False)
    assert len(state['history']) == 5 and state['scheduler']['step_size'] == 20 and state['calibration'] is None
    device = torch.device('cuda'); model = train.model_factory(device)
    model.load_state_dict(state['model']); model.eval()
    assert sum(q.numel() for q in model.parameters()) == p['model_parameters']
    patients = train.Patients(train._load_pkl_dataframe(p['pkl']), p['splits']['5pct']['train'], p['native_geometry'])
    assert len(patients) == 10
    full = p['runs'][-1]; assert set(full['combo']['terms']) == set(matrix.TERMS)
    cache = Targets(root, p, full); rows = []
    for item in patients.items:
        with torch.no_grad(), train.autocast(device): output = train.forward_item(model, item, device)
        logits = output.detach().float().requires_grad_(True)
        values = components(logits, item['label'][None].to(device), full, cache.get(item, device))
        assert set(values) == {'dice', 'bands', 'pair', *matrix.TERMS}
        rows.append(dict(case_name=item['case_name'], **gradient_record(values, logits)))
    artifacts = {}; warmup_sha = sha(root/'shared_warmup/latest.pt')
    for e in p['runs']:
        result = select_weights(rows, e, p['calibration'])
        for k, value in e['base_coefficients'].items(): assert result['coefficients'][k] == value
        assert all(math.isfinite(v) and v > 0 for v in result['coefficients'].values())
        if e['combo']['terms']:
            stat = result['statistics']['added_joint']
            assert stat['weighted_median_ratio'] <= e['combo']['gradient_ratio']*(1+1e-6)
            assert stat['weighted_p95_ratio'] <= p['calibration']['cap_ratio']*(1+1e-6)
        folder = root/'calibration'/e['id']; folder.mkdir(parents=True, exist_ok=True)
        value = dict(identity=train.identity(p,e), warmup_sha256=warmup_sha,
                     calibration=result, geometry=cache.statistics())
        write(folder/'record.json',value); artifacts[e['id']] = sha(folder/'record.json')
        print(json.dumps(dict(stage='calibrated', run=e['id'], coefficients=result['coefficients'])), flush=True)
    item = patients[0]; labels = item['label'][None].to(device)
    for e in (p['runs'][0], full):
        model.load_state_dict(state['model']); model.train(); restore_rng(state['rng']); model.zero_grad(set_to_none=True)
        with train.autocast(device): logits = train.forward_item(model, item, device)
        values = components(logits, labels, e, cache.get(item, device))
        weights = json.loads((root/'calibration'/e['id']/'record.json').read_text())['calibration']['coefficients']
        loss = combined_loss(values, weights); assert torch.isfinite(loss); loss.backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
        watched = {n: (q, q.detach().clone()) for n,q in model.named_parameters()
                   if n in ('swinViT.patch_embed.proj.weight', 'out.conv.conv.weight')}
        assert len(watched) == 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
        optimizer.step(); assert all(not torch.equal(q, before) for q,before in watched.values())
        del optimizer, watched, logits, values, loss
    write(root/'combo_gate.json', dict(passed=True, parameters_updated=True, training_cases=10,
          objectives_checked=22, warmup_sha256=warmup_sha, protocol_identity=p['identity_sha'], artifacts=artifacts))
    validate_gate(root,p)
    print(json.dumps(dict(passed=True, objectives_checked=22, parameters_updated=True)), flush=True)


def check_control(root, p):
    import csv
    from .combo_train import completed_record, identity
    e = p['runs'][0]; assert e['id'] == matrix.CONTROL
    record = completed_record(root/'runs'/matrix.CONTROL, identity(p,e)); assert record
    ref = json.loads((root/'anchor_reference.json').read_text())
    previous = list(csv.DictReader((root/'anchor_metrics.csv').open()))
    current = list(csv.DictReader((root/'runs'/matrix.CONTROL/'metrics.csv').open()))
    assert len(previous) == len(current), 'Matched control stopping epoch changed'
    max_error = 0.
    for a,b in zip(previous,current):
        for key in ('epoch','updates','learning_rate','train_dice','train_total','validation_dice'):
            error = abs(float(a[key])-float(b[key])); max_error = max(max_error,error)
            assert error <= 1e-6, 'Matched control differs: '+key
    assert record['best_epoch'] == ref['best_epoch']
    assert abs(record['validation']['macro_dice']-ref['validation']['macro_dice']) <= 1e-6
    write(root/'control_replay.json',dict(passed=True,maximum_absolute_difference=max_error,
          expected_dice=ref['validation']['macro_dice'],actual_dice=record['validation']['macro_dice'],
          protocol_identity=p['identity_sha']))


def update_report(root, p):
    from .report import report
    from .retention import retain_checkpoints
    value = report(p,root); retain_checkpoints(p,root,value)
    baseline = next((r['macro_dice'] for r in value['rows'] if r['objective']==matrix.CONTROL), None)
    lines = ['# Outer-constraint combinations: 5%, fold 0, seed 0','',f"Completed: {value['completed']}/22",'',
             '| Variant | Validation Dice % | Change vs control, pp |','|---|---:|---:|']
    for r in sorted(value['rows'],key=lambda r:-r['macro_dice']):
        delta = '' if baseline is None else f"{100*(r['macro_dice']-baseline):+.4f}"
        lines.append(f"| {r['objective']} | {100*r['macro_dice']:.4f} | {delta} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return value


def submit(root):
    from .retention import quota_free
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt = root/'submission.json'
        if receipt.exists(): print(receipt.read_text()); return
        p = load(root,full=True); assert quota_free() >= p['minimum_free_gib']
        parent = Path(p['predecessor_root'])
        old_jobs = json.loads((parent/'ap82_targeted_submission.json').read_text())['job_ids']
        queue = subprocess.check_output(['squeue','-u','3160552','-h','-o','%i'],text=True).split()
        # There must be at most one existing allocation; replace the old pending
        # spare before calling submit. Never create a third queued/running job.
        assert len(queue) <= 1, 'Two-job limit: retire the unused pending continuation first'
        dependencies = [j for j in old_jobs if j in queue]
        assert set(queue) <= set(dependencies), 'Unexpected unrelated job; do not oversubscribe'
        assert dependencies or json.loads((parent/'status.json').read_text())['state']=='complete'
        attempt = root/'submission_attempt.json'
        assert not attempt.exists(), 'Resolve unknown submission before retry'
        command = ['sbatch','--parsable','--job-name=hippo-outer-mix',f'--output={root}/slurm-%j.out',f'--error={root}/slurm-%j.err']
        # afterany handles the incumbent controller cancelling its unused continuation.
        # The worker additionally requires the predecessor's complete receipt.
        if dependencies: command += ['--dependency=afterany:'+':'.join(dependencies)]
        command += [str(root/'source/scripts/band_focus/run_combo.sh'),str(root)]
        value = dict(command=command,dependencies=dependencies,requested_unix=time.time(),protocol_identity=p['identity_sha'])
        write(attempt,value); result = subprocess.run(command,capture_output=True,text=True)
        value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        write(root/'submission_response.json',value)
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit(): raise RuntimeError('Submission unconfirmed; inspect response')
        value.update(job_id=job, combination_runs=22,replication_runs=9,total_runs=31); write(receipt,value)
        write(root/'status.json',dict(state='queued',job_id=job,current_stage=None,completed_runs=0,planned_runs=22,dependencies=dependencies))
        print(json.dumps(value),flush=True)


def worker(root):
    from .warmup import prepare_prefix, seed_run
    from .combo_train import completed_record, identity
    from .retention import quota_free
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = dict(state='verifying',job_id=os.environ['SLURM_JOB_ID'],current_stage=None,completed_runs=0,planned_runs=22)
        write(root/'status.json',state)
        def execute(stage,action,extra=()):
            state.update(state='running',current_stage=stage);write(root/'status.json',state)
            with (root/'logs'/(stage+'.log')).open('a',buffering=1) as log:
                subprocess.run([sys.executable,'-u','-m','scripts.band_focus.combo_campaign',action,
                     '--root',str(root),*extra],cwd=root/'source',stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            p = load(root,full=True)
            assert json.loads((Path(p['predecessor_root'])/'status.json').read_text())['state']=='complete', 'Predecessor incomplete; preserve both campaigns'
            prepare_prefix(p,root)
            execute('preflight','gate');validate_gate(root,p)
            for e in p['runs']:
                if completed_record(root/'runs'/e['id'],identity(p,e)) is None:
                    assert quota_free() >= 1., 'Insufficient free quota; checkpoints preserved'
                    seed_run(p,root,e);execute(e['id'],'train-one',('--run-id',e['id']))
                if e['id'] == matrix.CONTROL: check_control(root,p)
                value = update_report(root,p)
                state['completed_runs'] = value['completed'];write(root/'status.json',state)
            from .combo_replication import choose,report as replication_report
            selection=choose(root,p)
            state.update(replication_winner=selection['winner'],replication_completed=0,replication_planned=9)
            for spec in selection['cells']:
                execute('replication_'+spec['id'],'replicate-one',('--run-id',spec['id']))
                results=replication_report(root,p)
                state.update(replication_completed=results['completed']);write(root/'status.json',state)
            state.update(state='complete',current_stage=None,finished_unix=time.time());write(root/'status.json',state)
        except Exception as error:
            state.update(state='failed',error=repr(error),finished_unix=time.time());write(root/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','verify','submit','worker','gate','train-one','replicate-one'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path);parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root,args.parent)
    elif args.action=='verify':
        p=load(args.root,full=True);print(json.dumps(dict(passed=True,runs=len(p['runs']),identity=p['identity_sha'])))
    elif args.action=='train-one':
        import torch
        from .combo_train import run
        torch.set_num_threads(4);p=load(args.root);e=next(e for e in p['runs'] if e['id']==args.run_id)
        validate_gate(args.root,p);run(p,args.root,e)
    elif args.action=='replicate-one':
        from .combo_replication import run_cell
        run_cell(args.root,load(args.root),args.run_id)
    else:globals()[args.action](args.root)


if __name__=='__main__':main()
