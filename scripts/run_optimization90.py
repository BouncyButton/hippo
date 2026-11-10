"""Run and verify 24 fresh comparisons sequentially in one Slurm allocation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from campaign90 import ARMS, BUDGET, MODELS, digest, write_json
from optimization90 import CAMPAIGN, SCOPE, validation_updates


def verify_run(root, identity, arm, directory):
    from train_optimization90_core import ARTIFACTS
    output = root / identity / arm / 'fold0/seed0/run'
    source = root / 'source'
    protocol = json.loads((directory / 'protocol.json').read_text())
    alias = MODELS[identity]['alias']
    config = json.loads((output / 'config.json').read_text())
    done = json.loads((output / 'completion.json').read_text())
    schedule = protocol['models'][alias]['objectives'][arm]
    if (done['status'] != 'complete' or config['smoke_test']
            or done['budget_scope'] != SCOPE or config['budget_scope'] != SCOPE
            or (done['model'], done['objective'], done['fold'], done['seed']) != (alias, arm, 0, 0)
            or config['model_identity'] != identity or config['objective'] != arm
            or config['protocol_sha256'] != digest(directory / 'protocol.json')
            or config['dataset_sha256'] != protocol['pkl_sha256'] or config['budget_flops'] != BUDGET
            or config['lr_horizon_updates'] != MODELS[identity]['lr_horizon']
            or config['schedule'] != schedule or config['max_steps'] != schedule['max_steps']):
        raise ValueError('Completed identity or frozen recipe differs')
    for key, name in ARTIFACTS.items():
        if digest(output / name) != done[key]:
            raise ValueError('Completed artifact changed: ' + name)
    for key, name in [('script_sha256','train_optimization90_core.py'),
                      ('shared_source_sha256','train_compute_matched_baselines.py'),
                      ('objective_source_sha256','compute_constraint_objectives.py')]:
        if config[key] != digest(source / 'scripts' / name):
            raise ValueError('Training source changed')
    if alias == 'nnunet' and config['native_objective_sha256'] != digest(source / 'scripts/nnunet_multi_objective.py'):
        raise ValueError('Native objective differs')
    updates, attempts, calibration_attempts = done['updates'], done['training_attempts_charged'], done['calibration_attempts']
    prefix, f0, f1, cal = [schedule[k] for k in
                          ['prefix_updates','baseline_flops_per_update','flops_per_update','calibration_flops']]
    training = min(prefix, attempts) * f0 + max(0, attempts - prefix) * f1
    calibration = calibration_attempts * cal
    total = training + calibration
    if (updates != attempts or done['discarded_attempts'] != 0
            or attempts != schedule['max_steps'] or calibration_attempts != (0 if arm == 'dice' else 1)
            or done['consumed_training_flops'] != training or done['calibration_flops'] != calibration
            or done['consumed_total_flops'] != total or training != schedule['consumed_training_flops']
            or done['unused_flops'] != BUDGET - training or not 0 <= BUDGET - training < f1):
        raise ValueError('Training/calibration compute differs')
    ledger = json.loads((output / 'budget_ledger.json').read_text())
    if (ledger['training_flops'] != training or ledger['calibration_flops'] != calibration
            or ledger['charged_attempts'] != attempts or ledger['calibration_attempts'] != calibration_attempts
            or ledger['config_sha256'] != done['config_sha256']
            or ledger['calibration_sha256'] != done['calibration_sha256']):
        raise ValueError('Ledger differs')
    split = json.loads((directory / 'splits_10patients.json').read_text())[0]
    cal_report = json.loads((output / 'calibration.json').read_text())
    expected_weights = {name for name in ['bands','edge','sdf'] if name in arm}
    if cal_report['config_sha256'] != done['config_sha256'] or set(cal_report['weights']) != expected_weights:
        raise ValueError('Calibration identity differs')
    if arm != 'dice':
        if (cal_report['status'] != 'complete' or [r['case_name'] for r in cal_report['cases']] != split['train']
                or cal_report['anchor_training_attempts'] != prefix or cal_report['charged_flops'] != cal):
            raise ValueError('Calibration used unexpected data or checkpoint')
        for name, weight in cal_report['weights'].items():
            stats = cal_report['statistics'][name]
            if (not math.isfinite(weight) or weight <= 0
                    or not 0 < stats['weighted_median_ratio'] <= .1 + 1e-8
                    or not 0 < stats['weighted_p95_ratio'] <= .5 + 1e-8):
                raise ValueError('Invalid calibrated weight or ratio')
    if config['train_cases'] != split['train'] or config['validation_cases'] != split['val']:
        raise ValueError('Data split differs')
    recipe = json.loads((output / 'runtime_recipe.json').read_text())
    milestones = validation_updates(protocol, arm)
    if (recipe['adapter_sha256'] != digest(source / 'scripts/train_optimization90.py')
            or recipe['protocol_sha256'] != config['protocol_sha256']
            or recipe['validation_updates'] != milestones or recipe['model_identity'] != identity
            or recipe['lr_horizon_updates'] != MODELS[identity]['lr_horizon']):
        raise ValueError('Runtime adapter differs')
    best = json.loads((output / 'validation_best.json').read_text())
    latest = json.loads((output / 'validation_latest.json').read_text())
    for validation in [best, latest]:
        if [r['case_name'] for r in validation['cases']] != split['val']:
            raise ValueError('Validation set changed')
    rows = list(csv.DictReader((output / 'metrics.csv').open()))
    if [int(r['charged_attempts']) for r in rows] != milestones:
        raise ValueError('Validation checkpoints differ')
    for row in rows:
        c = int(row['charged_attempts'])
        expected_training = min(c, prefix) * f0 + max(0, c - prefix) * f1
        expected_calibration = cal if c > prefix else 0
        if (int(row['step']) != c or int(row['estimated_training_flops']) != expected_training
                or int(row['calibration_flops']) != expected_calibration
                or int(row['total_flops']) != expected_training + expected_calibration
                or expected_training > BUDGET
                or abs(float(row['compute_fraction']) - expected_training/BUDGET) > 1e-12
                or not math.isfinite(float(row['mean_train_loss']))):
            raise ValueError('Validation compute ledger differs')
    score, final = best['summary']['macro_hard_dice'], latest['summary']['macro_hard_dice']
    if (score != done['best_validation_hard_dice'] or score != max(float(r['macro_hard_dice']) for r in rows)
            or final != float(rows[-1]['macro_hard_dice'])
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in [score,final])):
        raise ValueError('Validation selection differs')
    return {'model': identity, 'objective': arm, 'fold': 0, 'seed': 0, 'updates': updates,
            'epochs': updates / 10, 'calibration_gflops': calibration / 1e9,
            'budget_scope': SCOPE, 'optimization_gflops': training/1e9,
            'total_gflops': total / 1e9, 'unused_optimization_gflops': (BUDGET-training)/1e9,
            'weights': cal_report['weights'], 'best_dice': score, 'final_dice': final,
            'anterior_dice': latest['summary']['anterior_hard_dice'],
            'posterior_dice': latest['summary']['posterior_hard_dice'],
            'completion_sha256': digest(output / 'completion.json'),
            'validation_latest_sha256': digest(output / 'validation_latest.json'),
            'runtime_recipe_sha256': digest(output / 'runtime_recipe.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'source'
    manifest = json.loads((args.root / 'source_manifest.json').read_text())
    def verify_source():
        for name, sha in manifest.items():
            if digest(source / name) != sha:
                raise ValueError('Frozen source changed: ' + name)
    verify_source()
    directory = source / 'experiments' / CAMPAIGN
    plan = json.loads((directory / 'plan.json').read_text())
    if (plan['budget_flops_per_run'] != BUDGET or plan['budget_scope'] != SCOPE
            or plan['model_order'] != list(MODELS) or plan['arm_order'] != list(ARMS)):
        raise ValueError('Authorized campaign differs')
    if digest(args.pkl) != plan['pkl_sha256']:
        raise ValueError('Dataset changed')
    environment = {**os.environ, 'PYTHONPATH': str(source)+':'+str(source/'scripts')+':'+str(source/'baselines/nnUNet')}
    rows, failures = [], []
    for identity, arm in plan['run_order']:
        verify_source()
        protocol_dir = directory / identity
        if digest(protocol_dir/'protocol.json') != plan['protocol_sha256'][identity]:
            raise ValueError('Protocol changed')
        output = args.root / identity / arm / 'fold0/seed0/run'
        command = [sys.executable, str(source/'scripts/train_optimization90.py'), '--model', MODELS[identity]['alias'],
                   '--objective', arm, '--fold','0','--seed','0','--protocol-dir',str(protocol_dir),
                   '--output',str(output),'--pkl',str(args.pkl),'--threads','4','--resume']
        write_json(args.root/'progress.json', {'status':'running','model':identity,'objective':arm,
                                              'completed':len(rows),'failed':len(failures),'total':24})
        print(json.dumps({'run_start':identity,'objective':arm,'completed':len(rows)}),flush=True)
        try:
            subprocess.run(command, env=environment, check=True)
            verify_source()
            row = verify_run(args.root, identity, arm, protocol_dir)
            rows.append(row)
            print(json.dumps({'run_complete':row}),flush=True)
        except Exception as error:
            failures.append({'model':identity,'objective':arm,'error':repr(error)})
            print(json.dumps({'run_failure':failures[-1]}),flush=True)
        write_json(args.root/'summary.json', {'status':'in_progress','optimization_budget_gflops_per_run':BUDGET/1e9, 'budget_scope':SCOPE,
                                             'fold':0,'seed':0,'runs':rows,'failures':failures,'total':24})
    status = 'complete' if len(rows)==24 and not failures else 'failed'
    write_json(args.root/'summary.json', {'status':status,'optimization_budget_gflops_per_run':BUDGET/1e9, 'budget_scope':SCOPE,
                                         'fold':0,'seed':0,'runs':rows,'failures':failures,'total':24})
    write_json(args.root/'progress.json', {'status':status,'completed':len(rows),'failed':len(failures),'total':24})
    if failures:
        raise RuntimeError('Campaign has failed or unverified runs: '+repr(failures))


if __name__ == '__main__':
    main()
