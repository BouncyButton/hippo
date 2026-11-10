"""Run three calibrated bands comparisons sequentially in one allocation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_budget_baselines import digest, write_json
from train_budget_bands import validation_updates


def verify_run(root, model, directory):
    from train_compute_matched_constraints import ARTIFACTS
    output = root / model / 'dice_bands/fold0/seed0/run'
    source = root / 'source'
    protocol = json.loads((directory / 'protocol.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    done = json.loads((output / 'completion.json').read_text())
    for key, name in ARTIFACTS.items():
        if digest(output / name) != done[key]:
            raise ValueError('Changed completed artifact: ' + name)
    if (done['status'] != 'complete' or config['smoke_test']
            or (done['model'], done['objective'], done['fold'], done['seed']) != (model, 'dice_bands', 0, 0)
            or (config['model'], config['objective'], config['fold'], config['seed']) != (model, 'dice_bands', 0, 0)
            or config['protocol_sha256'] != digest(directory / 'protocol.json')
            or config['dataset_sha256'] != protocol['pkl_sha256']
            or config['budget_flops'] != protocol['budget_flops']):
        raise ValueError('Identity, data, or protocol mismatch')
    for key, name in [('script_sha256', 'train_compute_matched_constraints.py'),
                      ('shared_source_sha256', 'train_compute_matched_baselines.py'),
                      ('objective_source_sha256', 'compute_constraint_objectives.py')]:
        if config[key] != digest(source / 'scripts' / name):
            raise ValueError('Training source mismatch')
    if model == 'nnunet' and config['native_objective_sha256'] != digest(source / 'scripts/nnunet_band_objective.py'):
        raise ValueError('Native nnU-Net objective differs')
    schedule = protocol['models'][model]['objectives']['dice_bands']
    attempts, calibration_attempts = done['training_attempts_charged'], done['calibration_attempts']
    prefix, f0, f1, cal = [schedule[k] for k in ['prefix_updates', 'baseline_flops_per_update', 'flops_per_update', 'calibration_flops']]
    training_flops = min(prefix, attempts) * f0 + max(0, attempts - prefix) * f1
    calibration_flops = calibration_attempts * cal
    total = training_flops + calibration_flops
    expected_attempts = prefix + (protocol['budget_flops'] - prefix * f0 - calibration_flops) // f1
    if (config['schedule'] != schedule or config['max_steps'] != schedule['max_steps']
            or calibration_attempts < 1 or attempts != expected_attempts
            or done['consumed_training_flops'] != training_flops
            or done['calibration_flops'] != calibration_flops or done['consumed_total_flops'] != total
            or done['unused_flops'] != protocol['budget_flops'] - total
            or not 0 <= done['unused_flops'] < f1
            or not 0 < done['updates'] <= attempts
            or done['discarded_attempts'] != attempts - done['updates']):
        raise ValueError('Training/calibration budget mismatch')
    ledger = json.loads((output / 'budget_ledger.json').read_text())
    if (ledger['training_flops'] != training_flops or ledger['calibration_flops'] != calibration_flops
            or ledger['charged_attempts'] != attempts or ledger['calibration_attempts'] != calibration_attempts
            or ledger['config_sha256'] != done['config_sha256']
            or ledger['calibration_sha256'] != done['calibration_sha256']):
        raise ValueError('Budget ledger mismatch')
    split = json.loads((directory / 'splits_10patients.json').read_text())[0]
    calibration = json.loads((output / 'calibration.json').read_text())
    if (calibration['status'] != 'complete' or calibration['config_sha256'] != done['config_sha256']
            or [r['case_name'] for r in calibration['cases']] != split['train']
            or calibration['anchor_training_attempts'] != prefix or calibration['charged_flops'] != cal
            or set(calibration['weights']) != {'bands'}
            or not math.isfinite(calibration['weights']['bands']) or calibration['weights']['bands'] <= 0):
        raise ValueError('Training-only calibration differs')
    stats = calibration['statistics']['bands']
    if not 0 < stats['weighted_median_ratio'] <= .1 + 1e-8 or not 0 < stats['weighted_p95_ratio'] <= .5 + 1e-8:
        raise ValueError('Calibration ratio caps exceeded')
    if config['train_cases'] != split['train'] or config['validation_cases'] != split['val']:
        raise ValueError('Data split differs')
    recipe = json.loads((output / 'runtime_recipe.json').read_text())
    milestones = validation_updates(protocol, model)
    if (recipe['adapter_sha256'] != digest(source / 'scripts/train_budget_bands.py')
            or recipe['protocol_sha256'] != config['protocol_sha256']
            or recipe['validation_updates'] != milestones):
        raise ValueError('Runtime recipe differs')
    best = json.loads((output / 'validation_best.json').read_text())
    latest = json.loads((output / 'validation_latest.json').read_text())
    for validation in [best, latest]:
        if [r['case_name'] for r in validation['cases']] != split['val']:
            raise ValueError('Validation cases differ')
    with (output / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    allowed = set(milestones) | {attempts}
    if (not rows or int(rows[-1]['charged_attempts']) != attempts
            or any(int(r['charged_attempts']) not in allowed for r in rows)
            or any(not math.isfinite(float(r['mean_train_loss'])) for r in rows)):
        raise ValueError('Incomplete or invalid validation curve')
    for row in rows:
        c = int(row['charged_attempts'])
        if int(row['estimated_training_flops']) != min(c, prefix) * f0 + max(0, c - prefix) * f1:
            raise ValueError('Metric training-compute ledger differs')
        if int(row['total_flops']) != int(row['estimated_training_flops']) + int(row['calibration_flops']) or int(row['total_flops']) > protocol['budget_flops']:
            raise ValueError('Metric total compute differs')
    score, final = best['summary']['macro_hard_dice'], latest['summary']['macro_hard_dice']
    if (not all(math.isfinite(v) and 0 <= v <= 1 for v in [score, final])
            or score != done['best_validation_hard_dice']
            or score != max(float(r['macro_hard_dice']) for r in rows)
            or final != float(rows[-1]['macro_hard_dice'])
            or not 0 < done['selected_step'] <= done['updates']):
        raise ValueError('Validation selection differs')
    return {'model': model, 'objective': 'dice_bands', 'fold': 0, 'seed': 0,
            'updates': done['updates'], 'charged_attempts': attempts,
            'volume_pass_epochs': done['updates'] / 10, 'calibration_attempts': calibration_attempts,
            'calibration_gflops': calibration_flops / 1e9, 'consumed_gflops': total / 1e9,
            'unused_gflops': done['unused_flops'] / 1e9, 'bands_weight': calibration['weights']['bands'],
            'best_validation_macro_dice': score, 'final_validation_macro_dice': final,
            'final_anterior_dice': latest['summary']['anterior_hard_dice'],
            'final_posterior_dice': latest['summary']['posterior_hard_dice'],
            'completion_sha256': digest(output / 'completion.json'),
            'validation_latest_sha256': digest(output / 'validation_latest.json'),
            'runtime_recipe_sha256': digest(output / 'runtime_recipe.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'source'
    expected = json.loads((args.root / 'source_manifest.json').read_text())

    def verify_source():
        for name, sha in expected.items():
            if digest(source / name) != sha:
                raise ValueError('Frozen source changed: ' + name)

    verify_source()
    directory = source / 'experiments/budget_bands_94603_20261005'
    campaign = json.loads((directory / 'campaign.json').read_text())
    if campaign['model_order'] != ['unet', 'swinunetr', 'nnunet'] or campaign['baseline_job_id'] != 684354:
        raise ValueError('Unexpected comparison campaign')
    if digest(args.pkl) != campaign['pkl_sha256']:
        raise ValueError('Dataset differs')
    history = []
    environment = {**os.environ, 'PYTHONPATH': str(source) + ':' + str(source / 'scripts') + ':' + str(source / 'baselines/nnUNet')}
    for model in campaign['model_order']:
        verify_source()
        protocol_dir = directory / model
        protocol = json.loads((protocol_dir / 'protocol.json').read_text())
        if protocol['budget_flops'] != campaign['budget_flops'] or protocol['folds'] != [0] or protocol['seeds'] != [0]:
            raise ValueError('Model protocol differs')
        output = args.root / model / 'dice_bands/fold0/seed0/run'
        command = [sys.executable, str(source / 'scripts/train_budget_bands.py'),
                   '--model', model, '--objective', 'dice_bands', '--fold', '0', '--seed', '0',
                   '--protocol-dir', str(protocol_dir), '--output', str(output), '--pkl', str(args.pkl),
                   '--threads', '4', '--resume']
        print(json.dumps({'bands_start': model, 'budget_gflops': campaign['budget_flops'] / 1e9}), flush=True)
        subprocess.run(command, env=environment, check=True)
        verify_source()
        history.append(verify_run(args.root, model, protocol_dir))
        write_json(args.root / 'summary.json', {'status': 'complete' if len(history) == 3 else 'in_progress',
                   'baseline_job_id': 684354, 'budget_gflops_per_run': campaign['budget_flops'] / 1e9, 'runs': history})
        print(json.dumps({'bands_complete': history[-1]}), flush=True)


if __name__ == '__main__':
    main()
