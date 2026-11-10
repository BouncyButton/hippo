"""Freeze, check, submit and run the nnU-Net compute-relative phase pilot."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from campaign90 import ARMS, BUDGET, MODELS, ROOT, digest, write_json
from constraint_budget_phases import budget_relative_schedule
from optimization90 import CAMPAIGN as PARENT_CAMPAIGN, SCOPE, validation_updates


CAMPAIGN = 'nnunet_early_calibration_31976_20261006'
OBJECTIVES = ARMS[1:]
PARENT_ROOT = Path('/mnt/beegfsstudents/home/3160552/four_models_six_optimization_31976_685853')
DIAGNOSTIC_ROOT = Path('/mnt/beegfsstudents/home/3160552/nnunet_constraint_diagnosis_686162')


def prepare():
    import shutil

    parent_dir = ROOT / 'experiments' / PARENT_CAMPAIGN / 'nnunet'
    parent_path = parent_dir / 'protocol.json'
    parent = json.loads(parent_path.read_text())
    for name, sha in parent['training_source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Original training source changed: ' + name)
    profile_path = ROOT / parent['profile_evidence']['file']
    if digest(profile_path) != parent['profile_evidence']['sha256']:
        raise ValueError('Profile changed')
    profile = json.loads(profile_path.read_text())
    costs = profile['costs']
    spec = MODELS['nnunet']
    schedules = {arm: budget_relative_schedule(BUDGET, spec['baseline_flops'],
                 costs[arm]['update_flops'], costs[arm]['calibration_flops']) for arm in OBJECTIVES}
    if any((s['prefix_updates'], s['constraint_warmup_updates'], s['max_steps'],
            s['full_strength_updates']) != (9, 9, 89, 72) for s in schedules.values()):
        raise ValueError('Unexpected nnU-Net phases')
    directory = ROOT / 'experiments' / CAMPAIGN
    model_dir = directory / 'nnunet'
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ('splits_10patients.json', 'nnunet_architecture.json', 'split_provenance.json'):
        shutil.copyfile(parent_dir / name, model_dir / name)
    protocol = {**parent,
        'name': CAMPAIGN + '_nnunet',
        'parent_protocol_sha256': digest(parent_path),
        'requested_arm_order': list(OBJECTIVES),
        'constraint_warmup_updates': 9,
        'models': {'nnunet': {'parameters': spec['parameters'], 'objectives': schedules}},
        'calibration_policy': 'After9 native updates (10% optimization compute), calibrate on all ten unaugmented training volumes in eval mode. Preserve detached FP32 full-resolution logit gradient calibration, median ratio0.1 and p95 cap0.5, sequential bands/edge/SDF. Calibration excluded from optimization cap and reported separately.',
        'constraint_scale': 'Ramp from1/9 to1 over updates10 through18; full weight at updates18 through89. Constraint loss arithmetic counted throughout ramp.',
        'restart_policy': 'Five fresh nnU-Net constrained runs, fold0/seed0, in one allocation. Reuse verified unchanged90-update baseline and exact89-update diagnostic replay as references.',
        'phase_policy': 'compute_relative_10pct_prefix_10pct_ramp',
        'policy_selection': 'Exploratory schedule pilot after inspecting the completed campaign. Common compute fractions correct the architecture-dependent constraint exposure; no validation-based weight search. Prefix and ramp change together, so this pilot does not isolate their individual effects.'}
    protocol['training_source_sha256'] = {**parent['training_source_sha256'],
        **{name: digest(ROOT / name) for name in
           ('scripts/constraint_budget_phases.py', 'scripts/nnunet_early_calibration.py')}}
    write_json(model_dir / 'protocol.json', protocol)
    diagnostic = ROOT / 'reports/nnunet_constraint_diagnosis_20261006'
    plan = dict(name=CAMPAIGN, model='nnunet', fold=0, seed=0, total_runs=5,
                objectives=list(OBJECTIVES), budget_flops=BUDGET, budget_scope=SCOPE,
                all_runs_fresh_initialization=True, protocol_sha256=digest(model_dir / 'protocol.json'),
                parent_protocol_sha256=digest(parent_path), parent_run_root=str(PARENT_ROOT),
                diagnostic_root=str(DIAGNOSTIC_ROOT), schedules=schedules,
                pkl_sha256=parent['pkl_sha256'],
                baseline_references={'final_updates':90, 'matched_updates':89,
                    'diagnosis_sha256':digest(diagnostic / 'diagnosis.json'),
                    'matched89_sha256':digest(diagnostic / 'matched89.json')},
                primary_endpoint='Final validation macro foreground hard Dice at the optimization cap versus original baseline90.',
                secondary_endpoint='Matched-update89 Dice versus exact diagnostic baseline89, and versus original prefix50/ramp50 constrained endpoints.',
                policy_selection=protocol['policy_selection'])
    write_json(directory / 'plan.json', plan)
    write_json(ROOT / 'reports' / CAMPAIGN / 'prepared_plan.json', plan)
    print(json.dumps({'campaign':CAMPAIGN, 'objectives':list(OBJECTIVES),
                      'prefix':9, 'ramp':9, 'updates':89, 'full_strength_updates':72}), flush=True)


def check_frozen(stage, pkl):
    source = stage / 'source'
    manifest_path = stage / 'source_manifest.json'
    for name, sha in json.loads(manifest_path.read_text()).items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    directory = source / 'experiments' / CAMPAIGN
    plan = json.loads((directory / 'plan.json').read_text())
    model_dir = directory / 'nnunet'
    protocol = json.loads((model_dir / 'protocol.json').read_text())
    if (plan['name'] != CAMPAIGN or plan['budget_scope'] != SCOPE or plan['budget_flops'] != BUDGET
            or (plan['model'], plan['fold'], plan['seed'], plan['total_runs']) != ('nnunet',0,0,5)
            or plan['objectives'] != list(OBJECTIVES) or not plan['all_runs_fresh_initialization']
            or digest(model_dir / 'protocol.json') != plan['protocol_sha256']
            or digest(pkl) != plan['pkl_sha256']):
        raise ValueError('Authorized identity, protocol or data differs')
    parent_dir = source / 'experiments' / PARENT_CAMPAIGN / 'nnunet'
    parent = json.loads((parent_dir / 'protocol.json').read_text())
    if digest(parent_dir / 'protocol.json') != plan['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    allowed_changes = {'name','parent_protocol_sha256','requested_arm_order','constraint_warmup_updates',
                       'models','calibration_policy','constraint_scale','restart_policy',
                       'phase_policy','policy_selection','training_source_sha256'}
    for key in set(parent) | set(protocol):
        if key not in allowed_changes and protocol.get(key) != parent.get(key):
            raise ValueError('Unexpected protocol change: ' + key)
    for name, sha in protocol['training_source_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError('Training source binding differs: ' + name)
    for key, filename in [('splits_sha256','splits_10patients.json'),
                          ('nnunet_architecture_sha256','nnunet_architecture.json')]:
        if digest(model_dir / filename) != protocol[key]:
            raise ValueError('Architecture or split changed')
    split = json.loads((model_dir / 'splits_10patients.json').read_text())[0]
    if len(set(split['train'])) != 10 or len(set(split['val'])) != 52 or set(split['train']) & set(split['val']):
        raise ValueError('Invalid split')
    profile_path = source / protocol['profile_evidence']['file']
    if digest(profile_path) != protocol['profile_evidence']['sha256']:
        raise ValueError('Profile changed')
    profile = json.loads(profile_path.read_text())
    for name, sha in profile['source_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError('Profiled loss graph changed')
    for arm in OBJECTIVES:
        measured = profile['costs'][arm]
        expected = budget_relative_schedule(BUDGET, MODELS['nnunet']['baseline_flops'],
                   measured['update_flops'], measured['calibration_flops'])
        schedule = protocol['models']['nnunet']['objectives'][arm]
        if schedule != expected or schedule != plan['schedules'][arm]:
            raise ValueError('Compute-relative schedule changed')
        scales = [min(1., (charged-9+1)/9) for charged in range(9,89)]
        if scales[0] != 1/9 or sum(s == 1 for s in scales) != 72:
            raise ValueError('Wrong active-loss phases')
        consumed = schedule['consumed_training_flops']
        if not consumed <= BUDGET < consumed + schedule['flops_per_update']:
            raise ValueError('Training cap floor differs')
        if validation_updates(protocol, arm)[-1] != 89:
            raise ValueError('Missing final validation')
    for filename, key in [('diagnosis.json','diagnosis_sha256'), ('matched89.json','matched89_sha256')]:
        if digest(Path(plan['diagnostic_root']) / filename) != plan['baseline_references'][key]:
            raise ValueError('Diagnostic reference changed')
    if not json.loads((Path(plan['diagnostic_root']) / 'diagnosis.json').read_text())['reference_replay_exact']:
        raise ValueError('Baseline replay not exact')
    return plan, protocol, model_dir


def preflight(stage, pkl):
    import importlib.metadata
    import monai
    import torch
    import train_optimization90_core as trainer
    from train_optimization90 import configure
    from preflight_optimization90 import semantic_checks

    plan, protocol, model_dir = check_frozen(stage, pkl)
    if (torch.__version__ != '2.12.1+cu130' or monai.__version__ != '1.6.0'
            or not importlib.metadata.version('dynamic-network-architectures').startswith('0.4.')):
        raise ValueError('Expected validated cluster runtime')
    torch.set_num_threads(2)
    original_lr = trainer.base.learning_rate
    original_factory = trainer.base.build_model
    original_validation = trainer.base.validation_steps
    for arm in OBJECTIVES:
        configure(protocol, arm)
        try:
            for step in range(89):
                if trainer.base.learning_rate('nnunet',step,89,.01) != original_lr('nnunet',step,174,.01,0.):
                    raise ValueError('Baseline LR trajectory differs')
        finally:
            trainer.base.learning_rate = original_lr
            trainer.base.build_model = original_factory
            trainer.base.validation_steps = original_validation
    trainer.base.seed_everything(0)
    architecture = json.loads((model_dir / 'nnunet_architecture.json').read_text())
    model = trainer.base.build_model('nnunet',architecture)
    criterion, optimizer = trainer.base.build_loss_and_optimizer('nnunet',model,protocol)
    if sum(p.numel() for p in model.parameters()) != MODELS['nnunet']['parameters'] or optimizer.param_groups[0]['momentum'] != .95:
        raise ValueError('Native architecture or optimizer changed')
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1,1,64,64,64))
    if tuple(outputs[0].shape) != (1,3,64,64,64) or not all(torch.isfinite(x).all() for x in outputs):
        raise ValueError('Invalid architecture output')
    checks = semantic_checks(criterion)
    checks.update(all_schedule_floors_verified=True, ramp_full_strength_count_verified=True,
                  all_lr_values_equal_original_baseline=True, architecture_and_split_unchanged=True)
    write_json(stage / 'preflight.json', dict(status='passed', checks=checks,
               torch=torch.__version__, monai=monai.__version__, total_runs=5,
               plan_sha256=digest(model_dir.parent / 'plan.json'),
               source_manifest_sha256=digest(stage / 'source_manifest.json'), budget_scope=SCOPE))
    print(json.dumps({'preflight':'passed','checks':checks}), flush=True)


def run(root, pkl):
    from run_optimization90 import verify_run

    plan, protocol, model_dir = check_frozen(root, pkl)
    pre = json.loads((root / 'preflight.json').read_text())
    if (pre['status'] != 'passed' or pre['plan_sha256'] != digest(model_dir.parent / 'plan.json')
            or pre['source_manifest_sha256'] != digest(root / 'source_manifest.json')):
        raise ValueError('Preflight certificate changed')
    parent_root = Path(plan['parent_run_root'])
    parent_dir = parent_root / 'source/experiments' / PARENT_CAMPAIGN / 'nnunet'
    baseline = verify_run(parent_root,'nnunet','dice',parent_dir)
    matched = json.loads((Path(plan['diagnostic_root']) / 'matched89.json').read_text())
    reference = {r['objective']:r for r in matched['comparisons']}
    write_json(root / 'references.json', dict(baseline90=baseline, matched89=matched,
        matched89_sha256=plan['baseline_references']['matched89_sha256'],
        baseline_completion_sha256=digest(parent_root / 'nnunet/dice/fold0/seed0/run/completion.json')))
    rows, failures = [], []
    for arm in OBJECTIVES:
        check_frozen(root,pkl)
        output = root / 'nnunet' / arm / 'fold0/seed0/run'
        write_json(root / 'progress.json', dict(status='running',objective=arm,completed=len(rows),failed=len(failures),total=5))
        print(json.dumps({'run_start':arm,'completed':len(rows),'total':5}), flush=True)
        command = [sys.executable,str(root / 'source/scripts/train_optimization90.py'),
                   '--model','nnunet','--objective',arm,'--fold','0','--seed','0',
                   '--protocol-dir',str(model_dir),'--output',str(output),
                   '--pkl',str(pkl),'--threads','4','--resume']
        try:
            subprocess.run(command,check=True)
            check_frozen(root,pkl)
            row = verify_run(root,'nnunet',arm,model_dir)
            row.update(prefix_updates=9,ramp_updates=9,full_strength_updates=72,
                       gain_vs_baseline_at_cap_pp=100*(row['final_dice']-baseline['final_dice']),
                       gain_vs_baseline89_pp=100*(row['final_dice']-reference['dice']['dice']),
                       gain_vs_old_same_objective89_pp=100*(row['final_dice']-reference[arm]['dice']))
            rows.append(row)
            print(json.dumps({'run_complete':row}), flush=True)
        except Exception as error:
            failures.append({'objective':arm,'error':repr(error)})
            print(json.dumps({'run_failure':failures[-1]}), flush=True)
        write_json(root / 'summary.json', dict(status='in_progress',runs=rows,failures=failures,total=5,
            budget_scope=SCOPE,budget_gflops=BUDGET/1e9,fold=0,seed=0,baseline90=baseline,
            baseline89=reference['dice'],phase_policy=protocol['phase_policy']))
    status = 'complete' if len(rows)==5 and not failures else 'failed'
    write_json(root / 'summary.json', dict(status=status,runs=rows,failures=failures,total=5,
        budget_scope=SCOPE,budget_gflops=BUDGET/1e9,fold=0,seed=0,baseline90=baseline,
        baseline89=reference['dice'],phase_policy=protocol['phase_policy']))
    write_json(root / 'progress.json', dict(status=status,completed=len(rows),failed=len(failures),total=5))
    if failures:
        raise RuntimeError('Unverified or failed pilot runs: ' + repr(failures))


def submit(stage, pkl):
    import fcntl
    import re
    import time

    if os.getuid() != 1395:
        raise ValueError('Unexpected cluster account')
    with (stage / 'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt = stage / 'submission.json'
        if receipt.exists():
            print(receipt.read_text())
            return
        plan, _, model_dir = check_frozen(stage,pkl)
        pre = json.loads((stage / 'preflight.json').read_text())
        if (pre['status'] != 'passed' or pre['plan_sha256'] != digest(model_dir.parent / 'plan.json')
                or pre['source_manifest_sha256'] != digest(stage / 'source_manifest.json') or pre['total_runs'] != 5):
            raise ValueError('Missing bound preflight')
        queue = subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines()
        active = [r.split('|') for r in queue if r.strip()]
        if len(active)>1 or any(r[1]!='RUNNING' for r in active):
            raise RuntimeError('One-running/one-queued limit: '+repr(queue))
        quota = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not limits or float(limits[2])-float(limits[1])<8:
            raise RuntimeError('Need8GiB free for pilot outputs: '+quota)
        command = ['sbatch','--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:'+dependency)
        command.append(str(stage / 'source/scripts/run_nnunet_early_calibration.sbatch'))
        response = subprocess.run(command,cwd=stage,capture_output=True,text=True,check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = dict(job_id=job,submitted_at_unix=time.time(),stage=str(stage),
                      run_root=f'/mnt/beegfsstudents/home/3160552/nnunet_early_calibration_31976_{job}',
                      objectives=list(OBJECTIVES),total_runs=5,model='nnunet',fold=0,seed=0,
                      budget_gflops_per_run=BUDGET/1e9,budget_scope=SCOPE,single_allocation=True,
                      prefix_updates=9,ramp_updates=9,updates_per_run=89,
                      baseline_reused_from=plan['parent_run_root'],dependency_afterany=dependency,
                      plan_sha256=digest(model_dir.parent / 'plan.json'),
                      preflight_sha256=digest(stage / 'preflight.json'),
                      source_manifest_sha256=digest(stage / 'source_manifest.json'),
                      queue_before_submission=queue,quota_before_submission=quota)
        write_json(receipt,record)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','preflight','run','submit'))
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--pkl',type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        target = args.root if args.mode == 'run' else args.stage
        if target is None or args.pkl is None:
            parser.error('Stage/root and pkl are required')
        {'preflight':preflight,'run':run,'submit':submit}[args.mode](target,args.pkl)


if __name__ == '__main__':
    main()
