"""Production artifact verification for the piecewise SDF timing pilot."""
import csv
import json
import math
from campaign90 import BUDGET, MODELS, digest
from scripts.capped_depth_delay.schedule import training_flops, validation_updates
from optimization90 import SCOPE


def verify_run(root, identity, arm, directory):
    from scripts.capped_depth_delay.core import ARTIFACTS
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
    for key, name in [('script_sha256','capped_depth_delay/core.py'),
                      ('shared_source_sha256','train_compute_matched_baselines.py'),
                      ('objective_source_sha256','compute_constraint_objectives.py')]:
        if config[key] != digest(source / 'scripts' / name):
            raise ValueError('Training source changed')
    if alias == 'nnunet' and config['native_objective_sha256'] != digest(source / 'scripts/nnunet_multi_objective.py'):
        raise ValueError('Native objective differs')
    for key, name in [('sdf_timing_sha256', 'schedule.py'), ('combined_objective_sha256', 'objectives.py')]:
        if config[key] != digest(source / 'scripts/capped_depth_delay' / name):
            raise ValueError('Combined objective or timing source differs')
    updates, attempts, calibration_attempts = done['updates'], done['training_attempts_charged'], done['calibration_attempts']
    prefix, f0, f1, cal = [schedule[k] for k in
                          ['prefix_updates','baseline_flops_per_update','flops_per_update','calibration_flops']]
    training = training_flops(attempts, schedule)
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
    expected_weights = {name for name in ['bands','edge','sdf','depth'] if name in arm}
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
    if (recipe['adapter_sha256'] != digest(source / 'scripts/capped_depth_delay/train.py')
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
        expected_training = training_flops(c, schedule)
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

