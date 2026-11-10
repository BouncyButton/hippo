"""Train the single authorized UNETR ViT-B16 Dice baseline and verify its ledger."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_budget_baselines import digest, write_json
from run_simple_baseline_screen import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'source'
    manifest = json.loads((args.root / 'source_manifest.json').read_text())
    for name, sha in manifest.items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    directory = source / 'experiments/unetr_dice_20261005/dice'
    protocol = json.loads((directory / 'protocol.json').read_text())
    cost = protocol['models']['unet']['flops_per_update']
    if (protocol['candidate_name'] != 'unetr_vit_b16' or protocol['folds'] != [0]
            or protocol['seeds'] != [0] or protocol['budget_flops'] != 62166546655000
            or protocol['initial_lrs']['unet'] != .0001):
        raise ValueError('Run differs from the single authorized baseline')
    output = args.root / 'unetr_vit_b16/dice/fold0/seed0/run'
    environment = {**os.environ, 'PYTHONPATH': str(source) + ':' + str(source / 'scripts')}
    command = [sys.executable, str(source / 'scripts/train_simple_baseline_pilot.py'),
               '--model', 'unet', '--fold', '0', '--seed', '0', '--pkl', str(args.pkl),
               '--protocol-dir', str(directory), '--output', str(output), '--threads', '4', '--resume']
    print(json.dumps({'baseline_start': 'unetr_vit_b16', 'updates': 477,
                      'budget_gflops': protocol['budget_flops'] / 1e9}), flush=True)
    subprocess.run(command, env=environment, check=True)
    for name, sha in manifest.items():
        if digest(source / name) != sha:
            raise ValueError('Source changed while training: ' + name)
    result = verify_run(output, directory, source)
    config = json.loads((output / 'config.json').read_text())
    recipe = json.loads((output / 'runtime_recipe.json').read_text())
    if (config['fold'] != 0 or config['seed'] != 0
            or config['max_steps'] != 477 or result['discarded_attempts'] != 0
            or recipe['candidate'] != 'unetr_vit_b16'
            or recipe['adapter_sha256'] != digest(source / 'scripts/train_simple_baseline_pilot.py')
            or recipe['validation_updates'] != sorted({max(1, min(477, target // cost)) for target in protocol['validation_target_flops']} | {477})):
        raise ValueError('Completed run identity, adapter or update schedule differs')
    write_json(args.root / 'summary.json', {
        'status': 'complete', 'model': 'unetr_vit_b16', 'objective': 'dice',
        'fold': 0, 'seed': 0, 'budget_gflops': protocol['budget_flops'] / 1e9,
        'development_only': True, 'run': result,
        'target_dice_range': [.70, .80],
        'target_met_best_and_final': all(.70 <= result[k] <= .80 for k in ['best_dice', 'final_dice'])})
    print(json.dumps({'baseline_complete': result}), flush=True)


if __name__ == '__main__':
    main()
