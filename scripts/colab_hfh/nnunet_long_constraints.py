"""Four predeclared non-BAND arms, one seed/fold, reused strong native control."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from scripts.colab_hfh.bands_campaign import read, write, sha
from scripts.colab_hfh.constraint_schedule import make_payload, training_flops

NAME = 'nnunet_long_constraints_v1'
ARMS = ('dice_depth', 'dice_edge_depth', 'dice_sdf_depth', 'dice_edge_sdf_depth')
FILES = ('nnunet_long_constraints.py', 'nnunet_long_constraints_worker.py')
CAP = 63952901334180
ANCHOR = {'after_update': 60, 'model_state_sha256': '4809a3cda62d96b53fb043cf131f648d91e577d6522cb70f4835799cc0070136'}


def pack_for(base, plan, arm):
    from scripts.colab_hfh.nnunet_seed0_long_worker import SCHEDULE
    if arm not in ARMS:
        raise ValueError('Not a predeclared arm')
    base = deepcopy(base)
    base['protocols']['nnunet']['protocol.json']['budget_flops'] = CAP
    pack = make_payload(base, 'nnunet', arm, plan['cost_references']['nnunet']['canonical_costs'],
                        prefix=60, ramp=9, target=.1, cap=.5, screening=False)
    p = pack['protocols']['nnunet']['protocol.json']
    p.update(name=NAME+'_'+arm, augmentation='none', seeds=[0], folds=[0],
             lr_horizon_updates=180, nnunet_warmup_fraction=0, nnunet_warmup_updates=0,
             long_schedule=SCHEDULE, matched_readiness_anchor=ANCHOR,
             endpoint='Final cap endpoint; seed0 fold0 only; no validation pruning',
             constraint_definition='Canonical physical non-BAND constraints, frozen native readiness anchor60, shared nine-step ramp and joint10%/p95 50% training-only calibration')
    p['initial_lrs']['nnunet'] = .1
    p['sdf_policy'][arm] = 'original_shared_ramp' if 'sdf' in arm else 'not_applicable'
    s = p['models']['nnunet']['objectives'][arm]
    s.update(sdf_prefix_updates=60, sdf_delay_rule='All components start at verified native readiness anchor60')
    # Reserve the same small monitoring allowance as the saved long control.
    for key in ('baseline_flops_per_update', 'early_flops_per_update', 'flops_per_update'):
        s[key] += 64*64**3
    s['matched_monitoring_allowance_per_update'] = 64*64**3
    steps = 0
    while training_flops(steps+1, s) <= CAP:
        steps += 1
    used = training_flops(steps, s)
    s.update(max_steps=steps, consumed_training_flops=used, consumed_flops=used,
             consumed_total_flops=used+s['calibration_flops'], unused_flops=CAP-used,
             patient_pass_epochs=steps/10)
    if steps not in (179, 180):
        raise ValueError('No saved matched endpoint for this schedule')
    pack.update(folds=[0], seeds=[0], rates={'nnunet': .1})
    return pack


def archive(output, durable, protocol, native):
    from scripts.colab_hfh import runtime
    p = read(protocol/'protocol.json'); arm = p['active_objective']
    s = p['models']['nnunet']['objectives'][arm]
    c = read(output/'config.json'); n = read(native/'config.json'); done = read(output/'completion.json')
    for key in ('fold', 'seed', 'dataset_sha256', 'train_cases', 'validation_cases', 'precision',
                'initial_lr', 'lr_horizon_updates', 'nnunet_warmup_fraction', 'nnunet_sgd_momentum', 'augmentation', 'native_loss'):
        if c[key] != n[key]:
            raise ValueError('Matched configuration differs: '+key)
    if read(output/'initialization.json') != read(native/'initialization.json'):
        raise ValueError('Matched initialization differs')
    used = training_flops(done['updates'], s)
    if done['discarded_attempts'] or done['updates'] != s['max_steps'] or done['consumed_training_flops'] != used or not used <= CAP < training_flops(done['updates']+1, s):
        raise ValueError('Cap or update accounting differs')
    cal = read(output/'calibration.json')
    if cal['anchor_successful_updates'] != 60 or cal['anchor_model_sha256'] != ANCHOR['model_state_sha256']:
        raise ValueError('Readiness anchor differs')
    if [r['case_name'] for r in cal['cases']] != c['train_cases'] or done['calibration_attempts'] != 1:
        raise ValueError('Calibration scope differs')
    if done['calibration_flops'] != s['calibration_flops']:
        raise ValueError('Calibration charge differs')
    stats = cal['statistics']['joint']
    if stats['worst_ramp_median_ratio'] > .1+1e-9 or stats['worst_ramp_p95_ratio'] > .5+1e-9:
        raise ValueError('Joint strength cap exceeded')
    val = read(output/'validation_latest.json'); train = read(output/'training_final.json')
    reference = read(native/'validation_latest.json')
    equal = read(native/('validation_update179.json' if done['updates'] == 179 else 'validation_latest.json'))
    dice = val['summary']['macro_hard_dice']
    row = dict(arm=arm, fold=0, seed=0, updates=done['updates'], dice=dice,
        training_dice=train['summary']['macro_hard_dice'],
        delta_vs_long_native_pp=100*(dice-reference['summary']['macro_hard_dice']),
        equal_update_delta_pp=100*(dice-equal['summary']['macro_hard_dice']),
        optimization_flops=used, optimization_cap=CAP, calibration_flops=done['calibration_flops'],
        active_constraint_updates=done['updates']-60, matched_prefix_verified=True)
    for split, evaluation in [('validation', val), ('training', train)]:
        for error in ('fp', 'fn'):
            row[split+'_'+error] = sum(r['foreground_'+error] for r in evaluation['cases'])
    write(output/'comparison.json', row)
    code = Path(runtime.__file__).read_text().replace("schedule = p['models'][p['trainer_alias']]['objectives']['dice']",
        "schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    ns = {'__name__': NAME+'_archive', '__file__': runtime.__file__}
    exec(compile(code, runtime.__file__, 'exec'), ns)
    ns['archive'](output, durable, 'baselines', 'nnunet', 0, 0, protocol)
    return row


def run(root):
    from scripts.colab_hfh.constraint_drive import Drive
    from scripts.colab_hfh.runtime import verify_archive
    work = root/NAME; work.mkdir(exist_ok=True)
    native = root/'nnunet_seed0_long_v1/native/archive'; verify_archive(native)
    if read(native/'readiness_model.json') != ANCHOR:
        raise ValueError('Saved native anchor differs')
    bundle = root/'bundle'; source = bundle/'source/scripts/colab_hfh'
    base = read(root/'bands_v1_payload.json'); plan = read(root/'constraint_matrix_plan.json')
    state = dict(status='preflight_complete', arms=ARMS, results=[], seed=0, fold=0,
                 native_dice=read(native/'result.json')['validation_dice'], cap=CAP)
    write(work/'status.json', state)
    env = dict(os.environ, PYTHONPATH=':'.join(str(bundle/p) for p in ('compatibility', 'source', 'source/scripts', 'source/baselines/nnUNet')),
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled', HIPPO_HFH_GPU_CACHE='1')
    with (root/'bands_campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        drive = Drive('results_'+NAME)
        try:
            for arm in ARMS:
                pack = pack_for(base, plan, arm)
                protocol = work/arm/'protocol'; protocol.mkdir(parents=True, exist_ok=True)
                for filename, value in pack['protocols']['nnunet'].items():
                    content = value if isinstance(value, str) else json.dumps(value, indent=2)+'\n'
                    target = protocol/filename
                    if target.exists() and target.read_text() != content:
                        raise ValueError('Frozen protocol differs')
                    target.write_text(content)
                write(protocol/'baseline_references.json', pack['references'])
                output = work/arm/'run'; durable = work/arm/'archive'
                output.mkdir(exist_ok=True); durable.mkdir(exist_ok=True)
                if (durable/'cleanup.json').exists():
                    verify_archive(durable); row = read(durable/'comparison.json')
                else:
                    state.update(status='running', current=arm); write(work/'status.json', state)
                    print('CONSTRAINT_PILOT_START', arm, flush=True)
                    if not (output/'completion.json').exists():
                        write(output/'execution_environment.json', dict(gpu=subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip()))
                        with (output/'worker.log').open('a') as log:
                            subprocess.run([sys.executable, '-m', 'scripts.colab_hfh.nnunet_long_constraints_worker',
                                '--protocol-dir', str(protocol), '--output', str(output), '--durable', str(durable),
                                '--pkl', str(bundle/'data_index.json'), '--fold', '0', '--seed', '0'],
                                cwd=bundle/'source', env={**env, 'HIPPO_DEADLINE': str(time.time()+1800)},
                                stdout=log, stderr=subprocess.STDOUT, check=True)
                    row = archive(output, durable, protocol, native)
                zipped = work/(arm+'.zip')
                with zipfile.ZipFile(zipped, 'w', zipfile.ZIP_DEFLATED) as z:
                    for folder, prefix in ((durable, 'results'), (protocol, 'protocol')):
                        for path in sorted(folder.iterdir()):
                            if path.is_file(): z.write(path, prefix+'/'+path.name)
                    for filename in FILES+('constraint_worker.py','constraint_calibration.py','constraint_geometry.py','nnunet_seed0_long_worker.py'):
                        z.write(source/filename, 'source/'+filename)
                state['results'].append(row); write(work/'status.json', state)
                drive.upload(zipped); drive.upload(work/'status.json')
                print('CONSTRAINT_PILOT_VERIFIED', json.dumps(row), flush=True)
            state.update(status='complete_four_arms_no_expansion', current=None)
            write(work/'status.json', state); drive.upload(work/'status.json')
        except BaseException as error:
            state.update(status='error_no_expansion', error=str(error)); write(work/'status.json', state)
            raise


if __name__ == '__main__':
    run(Path('/content/hippo_hfh_baselines_v2'))
