"""One-factor nnU-Net delayed SDF-only pilot with frozen weights."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from campaign90 import BUDGET, MODELS, ROOT, digest, write_json
from delayed_sdf_only_schedule import schedule_costs, training_flops, charge_for_step, validation_updates
from optimization90 import SCOPE


CAMPAIGN = 'nnunet_delayed_sdf_only_31976_20261006'
PARENT_CAMPAIGN = 'nnunet_early_calibration_31976_20261006'
ARM = 'dice_sdf'
PARENT_ROOT = Path('/mnt/beegfsstudents/home/3160552/nnunet_early_calibration_31976_686327')
BASELINE_ROOT = Path('/mnt/beegfsstudents/home/3160552/four_models_six_optimization_31976_685853')
NEW_SOURCES = ('scripts/delayed_sdf_only_schedule.py','scripts/train_delayed_sdf_only_core.py',
               'scripts/train_delayed_sdf_only.py','scripts/verify_delayed_sdf_only.py',
               'scripts/delayed_sdf_only_pilot.py')


def prepare():
    import shutil

    parent_dir = ROOT / 'experiments' / PARENT_CAMPAIGN / 'nnunet'
    parent_path = parent_dir / 'protocol.json'
    parent = json.loads(parent_path.read_text())
    for name, sha in parent['training_source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Parent training source changed: '+name)
    profile_path = ROOT / parent['profile_evidence']['file']
    if digest(profile_path) != parent['profile_evidence']['sha256']:
        raise ValueError('Original profile changed')
    costs = json.loads(profile_path.read_text())['costs']
    schedule = schedule_costs(BUDGET,costs['dice']['update_flops'],
                              costs[ARM]['update_flops'],costs[ARM]['calibration_flops'])
    directory = ROOT / 'experiments' / CAMPAIGN / 'nnunet'
    directory.mkdir(parents=True,exist_ok=True)
    for name in ('splits_10patients.json','nnunet_architecture.json','split_provenance.json'):
        shutil.copyfile(parent_dir / name,directory / name)
    previous = ROOT / 'reports' / PARENT_CAMPAIGN / 'nnunet'
    shutil.copyfile(previous / ARM / 'fold0/seed0/run/calibration.json',directory / 'calibration_reference.json')
    shutil.copyfile(ROOT / 'reports/four_models_six_optimization_31976_20261006/runs_metadata/nnunet/dice/fold0/seed0/run/metrics.csv',directory / 'baseline_control_metrics.csv')
    old_recipe = json.loads((previous / ARM / 'fold0/seed0/run/runtime_recipe.json').read_text())
    protocol = {**parent,'name':CAMPAIGN+'_nnunet',
        'parent_protocol_sha256':digest(parent_path),'requested_arm_order':[ARM],
        'models':{'nnunet':{'parameters':MODELS['nnunet']['parameters'],'objectives':{ARM:schedule}}},
        'phase_policy':'sdf_only_activation_after50_updates',
        'constraint_scale':'Native loss only through50. SDF activates at51, ramps over51..59, and stays full through89. Calibrated SDF-only base weight unchanged.',
        'restart_policy':'One fresh fold0/seed0 timing pilot. Preserve native optimizer/LR/data and all previous comparison results.',
        'calibration_policy':'Reproduce original SDF-only calibration after9 native updates on the same ten training volumes; require exact original weights and anchor model hash. Calibration excluded from cap and separately reported. Only SDF activation is delayed.',
        'calibration_reference_sha256':digest(directory / 'calibration_reference.json'),
        'baseline_control_metrics_sha256':digest(directory / 'baseline_control_metrics.csv'),
        'control_validation_updates':old_recipe['validation_updates'],
        'accounting':'Optimization-only cap; native-only graph through50, SDF-only auxiliary graph from51. Existing measured dense arithmetic reused for unchanged tensor loss graphs; add two host scalar operations per active SDF update for its ramp division and independent coefficient multiplication. Calibration remains outside cap. Validation, augmentation, target preprocessing, comparisons and I/O excluded consistently.',
        'policy_selection':'Exploratory follow-up to completed686327. First SDF update51 uses the historical50-update prefix as a fixed timing anchor, selected before this run; no validation-triggered activation or weight search.'}
    protocol['training_source_sha256'] = {**parent['training_source_sha256'],
                                         **{name:digest(ROOT/name) for name in NEW_SOURCES}}
    if schedule['max_steps'] != 89 or validation_updates(protocol,ARM) != old_recipe['validation_updates']:
        raise ValueError('Pilot changes update allowance or recorded checkpoint grid')
    write_json(directory / 'protocol.json',protocol)
    plan = dict(name=CAMPAIGN,model='nnunet',objective=ARM,fold=0,seed=0,total_runs=1,
                budget_flops=BUDGET,budget_scope=SCOPE,schedule=schedule,
                protocol_sha256=digest(directory / 'protocol.json'),parent_protocol_sha256=digest(parent_path),
                all_runs_fresh_initialization=True,pkl_sha256=parent['pkl_sha256'],
                parent_run_root=str(PARENT_ROOT),baseline_run_root=str(BASELINE_ROOT),
                primary_endpoint='Final macro foreground validation Dice at the unchanged optimization cap.',
                controls=['native baseline90','early_sdf89'],
                only_intervention='SDF activation: first update10 becomes51; calibrated weights and9-step ramp unchanged.',
                expected_control_trajectory='All shared recorded Dice, class Dice and LR checkpoints through50 must equal the native baseline685853 exactly.',
                policy_selection=protocol['policy_selection'])
    write_json(directory.parent / 'plan.json',plan)
    write_json(ROOT / 'reports' / CAMPAIGN / 'prepared_plan.json',plan)
    print(json.dumps({'campaign':CAMPAIGN,'schedule':schedule}),flush=True)


def check_frozen(stage,pkl):
    source = stage / 'source'
    for name, sha in json.loads((stage / 'source_manifest.json').read_text()).items():
        if digest(source/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    directory = source / 'experiments' / CAMPAIGN / 'nnunet'
    plan = json.loads((directory.parent / 'plan.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    if (plan['name'] != CAMPAIGN or (plan['model'],plan['objective'],plan['fold'],plan['seed'],plan['total_runs']) != ('nnunet',ARM,0,0,1)
            or plan['budget_flops'] != BUDGET or plan['budget_scope'] != SCOPE
            or not plan['all_runs_fresh_initialization']
            or digest(directory/'protocol.json') != plan['protocol_sha256'] or digest(pkl) != plan['pkl_sha256']):
        raise ValueError('Authorized pilot, protocol or data differs')
    parent_path = source / 'experiments' / PARENT_CAMPAIGN / 'nnunet/protocol.json'
    parent = json.loads(parent_path.read_text())
    if digest(parent_path) != plan['parent_protocol_sha256']:
        raise ValueError('Parent protocol differs')
    allowed = {'name','parent_protocol_sha256','requested_arm_order','models','phase_policy','constraint_scale',
               'restart_policy','calibration_policy','calibration_reference_sha256','baseline_control_metrics_sha256',
               'control_validation_updates','accounting','policy_selection','training_source_sha256'}
    for key in set(parent)|set(protocol):
        if key not in allowed and parent.get(key) != protocol.get(key):
            raise ValueError('Unexpected change: '+key)
    for name, sha in protocol['training_source_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('Training source binding differs')
    for key,name in [('splits_sha256','splits_10patients.json'),('nnunet_architecture_sha256','nnunet_architecture.json'),
                     ('calibration_reference_sha256','calibration_reference.json'),('baseline_control_metrics_sha256','baseline_control_metrics.csv')]:
        if digest(directory/name) != protocol[key]:
            raise ValueError('Frozen input changed: '+name)
    expected_cal = Path(plan['parent_run_root'])/'nnunet'/ARM/'fold0/seed0/run/calibration.json'
    if digest(expected_cal) != protocol['calibration_reference_sha256']:
        raise ValueError('Parent calibration changed')
    profile_path = source / protocol['profile_evidence']['file']
    if digest(profile_path) != protocol['profile_evidence']['sha256']:
        raise ValueError('Profile changed')
    profile = json.loads(profile_path.read_text())
    for name, sha in profile['source_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('Measured native loss graph changed')
    costs = profile['costs']
    expected = schedule_costs(BUDGET,costs['dice']['update_flops'],
                             costs[ARM]['update_flops'],costs[ARM]['calibration_flops'])
    if (expected != plan['schedule'] or expected != protocol['models']['nnunet']['objectives'][ARM]
            or protocol['constraint_warmup_updates'] != 9 or protocol['lr_horizon_updates'] != 174
            or validation_updates(protocol,ARM) != protocol['control_validation_updates']):
        raise ValueError('Schedule or LR differs')
    for n in range(91):
        if training_flops(n,expected) != sum(charge_for_step(i,expected) for i in range(n)):
            raise ValueError('Piecewise ledger mismatch')
    return plan,protocol,directory


def preflight(stage,pkl):
    import importlib.metadata
    import torch
    import monai
    import train_delayed_sdf_only_core as trainer
    import nnunet_multi_objective as native
    from train_delayed_sdf_only import configure
    from delayed_sdf_only_schedule import native_training_loss,sdf_scale_for_step
    from thesis.new_constraints.signed_distance import signed_distance_maps

    plan,protocol,directory = check_frozen(stage,pkl)
    if (torch.__version__ != '2.12.1+cu130' or monai.__version__ != '1.6.0'
            or not importlib.metadata.version('dynamic-network-architectures').startswith('0.4.')):
        raise ValueError('Validated cluster runtime required')
    torch.set_num_threads(2)
    original_lr = trainer.base.learning_rate
    original_validation = trainer.base.validation_steps
    configure(protocol,ARM)
    try:
        for step in range(89):
            if trainer.base.learning_rate('nnunet',step,89,.01) != original_lr('nnunet',step,174,.01,0.):
                raise ValueError('Native LR trajectory changed')
    finally:
        trainer.base.learning_rate = original_lr
        trainer.base.validation_steps = original_validation
    architecture = json.loads((directory/'nnunet_architecture.json').read_text())
    model = trainer.base.build_model('nnunet',architecture)
    criterion,optimizer = trainer.base.build_loss_and_optimizer('nnunet',model,protocol)
    if sum(p.numel() for p in model.parameters()) != 16548076 or optimizer.param_groups[0]['momentum'] != .95:
        raise ValueError('Native architecture or optimizer changed')
    coords = torch.arange(64)
    x,y,z = torch.meshgrid(coords,coords,coords,indexing='ij')
    inside = (x>=20)&(x<44)&(y>=22)&(y<42)&(z>=25)&(z<39)
    labels = (inside.long()*(1+(x>=32).long()))[None,None]
    sdf = signed_distance_maps(labels)
    outputs = [torch.randn(1,3,n,n,n,requires_grad=True) for n in (64,32,16,8)]
    weights = json.loads((directory/'calibration_reference.json').read_text())['weights']
    schedule = plan['schedule']
    for charged in (9,17,49,50,57,58,88):
        scale = min(1.,(charged-9+1)/9)
        active = 'dice' if charged<50 else ARM
        late_scale = sdf_scale_for_step(charged,schedule)
        actual = native_training_loss(outputs,labels,active,weights,scale,criterion,
                                      sdf if charged>=50 else None,late_scale)
        expected_weights = weights if charged<50 else {**weights,'sdf':weights['sdf']*late_scale}
        expected = native.training_loss(outputs,labels,active,expected_weights,scale,criterion,sdf)
        ga = torch.autograd.grad(actual,outputs[:3],retain_graph=True)
        ge = torch.autograd.grad(expected,outputs[:3],retain_graph=True)
        if not torch.equal(actual,expected) or any(not torch.equal(a,b) or not torch.isfinite(a).all() for a,b in zip(ga,ge)):
            raise ValueError('Phase loss or native deep-supervision gradients differ')
    checks = dict(piecewise_accounting_all_0_to90_attempts_verified=True,
                  native_tensor_loss_graphs_unchanged=True,
                  phase_boundary_losses_and_gradients_exact=True,
                  calibrated_base_weights_unchanged=True,
                  native_only_through50_verified=True,sdf_first_update=51,sdf_full_strength_update=59,
                  lr_trajectory_unchanged=True,architecture_and_split_unchanged=True,
                  host_scalar_weight_selection_flops_per_sdf_update=2)
    write_json(stage/'preflight.json',dict(status='passed',checks=checks,
               torch=torch.__version__,monai=monai.__version__,
               plan_sha256=digest(directory.parent/'plan.json'),
               source_manifest_sha256=digest(stage/'source_manifest.json'),budget_scope=SCOPE,total_runs=1))
    print(json.dumps({'preflight':'passed','checks':checks}),flush=True)


def run(root,pkl):
    from verify_delayed_sdf_only import verify_run
    from run_optimization90 import verify_run as verify_original

    plan,protocol,directory = check_frozen(root,pkl)
    pre = json.loads((root/'preflight.json').read_text())
    if (pre['status']!='passed' or pre['plan_sha256']!=digest(directory.parent/'plan.json')
            or pre['source_manifest_sha256']!=digest(root/'source_manifest.json')):
        raise ValueError('Bound preflight missing')
    previous = Path(plan['parent_run_root'])
    previous_dir = previous/'source/experiments'/PARENT_CAMPAIGN/'nnunet'
    baseline_root = Path(plan['baseline_run_root'])
    baseline_dir = baseline_root/'source/experiments/four_models_six_optimization_31976_20261006/nnunet'
    references = {'early_sdf89':verify_original(previous,'nnunet',ARM,previous_dir),
                  'baseline90':verify_original(baseline_root,'nnunet','dice',baseline_dir)}
    write_json(root/'references.json',references)
    write_json(root/'progress.json',dict(status='running',model='nnunet',objective=ARM,total=1,completed=0))
    output = root/'nnunet'/ARM/'fold0/seed0/run'
    subprocess.run([sys.executable,str(root/'source/scripts/train_delayed_sdf_only.py'),
                   '--model','nnunet','--objective',ARM,'--fold','0','--seed','0',
                   '--protocol-dir',str(directory),'--output',str(output),
                   '--pkl',str(pkl),'--threads','4','--resume'],check=True)
    check_frozen(root,pkl)
    row = verify_run(root,'nnunet',ARM,directory)
    expected_cal = json.loads((directory/'calibration_reference.json').read_text())
    cal = json.loads((output/'calibration.json').read_text())
    if any(cal[key]!=expected_cal[key] for key in ('weights','statistics','cases','anchor_model_sha256')):
        raise ValueError('Calibration differs from original update9 reference')
    config = json.loads((output/'config.json').read_text())
    if config['sdf_timing_sha256'] != digest(root/'source/scripts/delayed_sdf_only_schedule.py'):
        raise ValueError('Timing source differs')
    old = {int(r['step']):r for r in csv.DictReader((directory/'baseline_control_metrics.csv').open())}
    new = {int(r['step']):r for r in csv.DictReader((output/'metrics.csv').open())}
    prefix_checks = []
    for step in sorted(s for s in new if s<=50 and s in old):
        # Mean loss spans the interval since the preceding validation, which
        # differs between the native and early-SDF checkpoint grids.
        for key in ('macro_hard_dice','anterior_hard_dice','posterior_hard_dice','learning_rate'):
            if float(new[step][key]) != float(old[step][key]):
                raise ValueError('Pre-SDF trajectory differs from native baseline at'+str(step)+':'+key)
        prefix_checks.append(step)
    if 9 not in prefix_checks or 50 not in prefix_checks or len(prefix_checks) < 10:
        raise ValueError('Insufficient native prefix checkpoints verified')
    row.update(gain_vs_early_sdf89_pp=100*(row['final_dice']-references['early_sdf89']['final_dice']),
               gain_vs_baseline_at_cap_pp=100*(row['final_dice']-references['baseline90']['final_dice']),
               calibration_weights_and_anchor_exact=True,baseline_control_prefix_exact=True,
               control_prefix_checkpoints=prefix_checks,sdf_first_update=51,sdf_full_strength_update=59)
    write_json(root/'summary.json',dict(status='complete',runs=[row],failures=[],total=1,
               references=references,budget_gflops=BUDGET/1e9,budget_scope=SCOPE,fold=0,seed=0,
               only_intervention=plan['only_intervention']))
    write_json(root/'progress.json',dict(status='complete',completed=1,failed=0,total=1))
    print(json.dumps({'run_complete':row}),flush=True)


def submit(stage,pkl):
    import fcntl
    import re
    import time

    if os.getuid()!=1395:
        raise ValueError('Unexpected cluster account')
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt = stage/'submission.json'
        if receipt.exists():
            print(receipt.read_text())
            return
        plan,_,directory = check_frozen(stage,pkl)
        pre = json.loads((stage/'preflight.json').read_text())
        if (pre['status']!='passed' or pre['plan_sha256']!=digest(directory.parent/'plan.json')
                or pre['source_manifest_sha256']!=digest(stage/'source_manifest.json')):
            raise ValueError('Preflight binding differs')
        queue = subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines()
        active = [r.split('|') for r in queue if r.strip()]
        if len(active)>1 or any(r[1]!='RUNNING' for r in active):
            raise RuntimeError('One-running/one-queued limit: '+repr(queue))
        quota = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not limits or float(limits[2])-float(limits[1])<2:
            raise RuntimeError('Need2GiB free for this single pilot: '+quota)
        command = ['sbatch','--parsable']
        dependency = active[0][0] if active else None
        if dependency:
            command.append('--dependency=afterany:'+dependency)
        command.append(str(stage/'source/scripts/run_delayed_sdf_only.sbatch'))
        response = subprocess.run(command,cwd=stage,capture_output=True,text=True,check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = dict(job_id=job,submitted_at_unix=time.time(),stage=str(stage),
                      run_root=f'/mnt/beegfsstudents/home/3160552/nnunet_delayed_sdf_only_31976_{job}',
                      model='nnunet',objective=ARM,fold=0,seed=0,total_runs=1,single_allocation=True,
                      budget_gflops=BUDGET/1e9,budget_scope=SCOPE,updates=89,
                      calibration_update=9,
                      sdf_first_update=51,sdf_full_strength_update=59,
                      dependency_afterany=dependency,queue_before_submission=queue,
                      quota_before_submission=quota,plan_sha256=digest(directory.parent/'plan.json'),
                      preflight_sha256=digest(stage/'preflight.json'),
                      source_manifest_sha256=digest(stage/'source_manifest.json'))
        write_json(receipt,record)
        print(json.dumps(record),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','preflight','run','submit'))
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--pkl',type=Path)
    args = parser.parse_args()
    if args.mode=='prepare':
        prepare()
        return
    target = args.root if args.mode=='run' else args.stage
    if target is None or args.pkl is None:
        parser.error('Stage/root and pkl required')
    {'preflight':preflight,'run':run,'submit':submit}[args.mode](target,args.pkl)


if __name__ == '__main__':
    main()
