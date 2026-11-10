"""Train and verify three baselines sequentially within one Slurm allocation."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_run(root, model, protocol_dir):
    from train_budget_baseline import validation_updates
    source = root / 'source'
    output = root / model / 'fold0/seed0/run'
    protocol = json.loads((protocol_dir / 'protocol.json').read_text())
    done = json.loads((output / 'completion.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    artifacts = {'config_sha256': 'config.json', 'metrics_sha256': 'metrics.csv',
                 'checkpoint_best_sha256': 'checkpoint_best.pt',
                 'checkpoint_latest_sha256': 'checkpoint_latest.pt',
                 'budget_ledger_sha256': 'budget_ledger.json',
                 'validation_best_sha256': 'validation_best.json'}
    if model == 'unet':
        artifacts['calibration_sha256'] = 'calibration.json'
    for key, filename in artifacts.items():
        if digest(output / filename) != done[key]:
            raise ValueError('Changed completed artifact: ' + filename)
    if (done['status'] != 'complete' or config['smoke_test']
            or (done['model'], done['fold'], done['seed']) != (model, 0, 0)
            or (config['model'], config['fold'], config['seed']) != (model, 0, 0)
            or config['protocol_sha256'] != digest(protocol_dir / 'protocol.json')
            or config['dataset_sha256'] != protocol['pkl_sha256']):
        raise ValueError('Completed identity/protocol/data mismatch')
    script = 'train_compute_matched_constraints.py' if model == 'unet' else 'train_compute_matched_baselines.py'
    if config['script_sha256'] != digest(source / 'scripts' / script):
        raise ValueError('Completed training source mismatch')
    schedule = protocol['models'][model]
    cost = schedule['objectives']['dice']['flops_per_update'] if model == 'unet' else schedule['flops_per_update']
    cap = protocol['budget_flops'] // cost
    attempts = done['training_attempts_charged']
    unused_key = 'unused_flops' if model == 'unet' else 'unused_training_flops'
    if (attempts != cap or config['max_steps'] != cap or config['budget_flops'] != protocol['budget_flops']
            or done['consumed_training_flops'] != cap * cost
            or done[unused_key] != protocol['budget_flops'] - cap * cost
            or not 0 < done['updates'] <= attempts
            or done['discarded_attempts'] != attempts - done['updates']
            or not 0 < done['selected_step'] <= done['updates']):
        raise ValueError('Completed training budget mismatch')
    if model == 'unet' and (done['objective'] != 'dice' or done['calibration_flops'] != 0):
        raise ValueError('Expected uncalibrated Dice baseline')
    ledger = json.loads((output / 'budget_ledger.json').read_text())
    charged = ledger['training_flops'] if model == 'unet' else ledger['charged_training_flops']
    if charged != cap * cost or ledger['charged_attempts'] != cap or ledger['config_sha256'] != done['config_sha256']:
        raise ValueError('Completed budget ledger mismatch')
    split = json.loads((protocol_dir / 'splits_10patients.json').read_text())[0]
    best = json.loads((output / 'validation_best.json').read_text())
    latest = json.loads((output / 'validation_latest.json').read_text())
    for validation in [best, latest]:
        if [r['case_name'] for r in validation['cases']] != split['val']:
            raise ValueError('Validation cases differ')
    if config['train_cases'] != split['train'] or config['validation_cases'] != split['val']:
        raise ValueError('Training/validation split differs')
    recipe = json.loads((output / 'runtime_recipe.json').read_text())
    if (recipe['adapter_sha256'] != digest(source / 'scripts/train_budget_baseline.py')
            or recipe['protocol_sha256'] != digest(protocol_dir / 'protocol.json')
            or recipe['validation_updates'] != validation_updates(protocol, model)):
        raise ValueError('Adapter or validation recipe differs')
    with (output / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(rows[-1]['charged_attempts']) != cap:
        raise ValueError('Incomplete validation curve')
    expected = set(validation_updates(protocol, model))
    if any(int(r['charged_attempts']) not in expected for r in rows):
        raise ValueError('Unexpected validation checkpoint')
    score = best['summary']['macro_hard_dice']
    final = latest['summary']['macro_hard_dice']
    if (not all(math.isfinite(s) and 0 <= s <= 1 for s in [score, final])
            or score != done['best_validation_hard_dice']
            or score != max(float(r['macro_hard_dice']) for r in rows)
            or final != float(rows[-1]['macro_hard_dice'])):
        raise ValueError('Validation scores differ')
    return {'model': model, 'fold': 0, 'seed': 0, 'updates': done['updates'],
            'charged_attempts': attempts, 'volume_pass_epochs': done['updates'] / 10,
            'consumed_gflops': cap * cost / 1e9,
            'unused_gflops': (protocol['budget_flops'] - cap * cost) / 1e9,
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
    plan_dir = source / 'experiments/budget_baselines_94603_20261005'
    campaign = json.loads((plan_dir / 'campaign.json').read_text())
    if campaign['model_order'] != ['unet', 'swinunetr', 'nnunet']:
        raise ValueError('Expected exactly three sequential baselines')
    if digest(args.pkl) != campaign['pkl_sha256']:
        raise ValueError('Dataset differs from campaign')
    environment = {**os.environ, 'PYTHONPATH': str(source) + ':' + str(source / 'scripts') + ':' + str(source / 'baselines/nnUNet')}
    history = []
    for model in campaign['model_order']:
        verify_source()
        directory = plan_dir / model
        protocol = json.loads((directory / 'protocol.json').read_text())
        if (protocol['budget_flops'] != campaign['budget_flops']
                or protocol['folds'] != [0] or protocol['seeds'] != [0]):
            raise ValueError('Model protocol differs from common budget or identity')
        output = args.root / model / 'fold0/seed0/run'
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(source / 'scripts/train_budget_baseline.py'),
                   '--model', model, '--fold', '0', '--seed', '0', '--pkl', str(args.pkl),
                   '--output', str(output), '--protocol-dir', str(directory), '--threads', '4', '--resume']
        if model == 'unet':
            command.extend(['--objective', 'dice'])
        print(json.dumps({'baseline_start': model, 'budget_gflops': campaign['budget_flops'] / 1e9}), flush=True)
        subprocess.run(command, env=environment, check=True)
        verify_source()
        history.append(verify_run(args.root, model, directory))
        write_json(args.root / 'summary.json', {'status': 'complete' if len(history) == 3 else 'in_progress',
                                              'budget_gflops_per_run': campaign['budget_flops'] / 1e9, 'runs': history})
        print(json.dumps({'baseline_complete': history[-1]}), flush=True)


if __name__ == '__main__':
    main()
