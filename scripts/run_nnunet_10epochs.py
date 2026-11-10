"""Train the authorized single ten-epoch nnU-Net baseline pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_budget_baselines import digest, write_json, verify_run as verify_baseline
from run_budget_bands import verify_run as verify_bands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'source'
    manifest = json.loads((args.root / 'source_manifest.json').read_text())
    directory = source / 'experiments/nnunet_10epochs_20261005'
    plan = json.loads((directory / 'plan.json').read_text())
    if (plan['model_order'] != ['nnunet'] or plan['arm_order'] != ['baseline']
            or plan['budget_flops_per_run'] != 35529389630100
            or (plan['fold'], plan['seed']) != (0, 0)):
        raise ValueError('Unexpected comparison')
    environment = {**os.environ, 'PYTHONPATH': str(source) + ':' + str(source / 'scripts')
                   + ':' + str(source / 'baselines/nnUNet')}
    results = {}
    for arm in plan['arm_order']:
        for name, sha in manifest.items():
            if digest(source / name) != sha:
                raise ValueError('Frozen source changed: ' + name)
        protocol_dir = directory / arm
        if digest(protocol_dir / 'protocol.json') != plan['protocol_sha256'][arm]:
            raise ValueError('Protocol changed')
        bands = arm == 'bands'
        output = args.root / ('nnunet/dice_bands/fold0/seed0/run' if bands else 'nnunet/fold0/seed0/run')
        script = 'train_budget_bands.py' if bands else 'train_budget_baseline.py'
        command = [sys.executable, str(source / 'scripts' / script), '--model', 'nnunet',
                   '--fold', '0', '--seed', '0', '--pkl', str(args.pkl), '--protocol-dir', str(protocol_dir),
                   '--output', str(output), '--threads', '4', '--resume']
        if bands:
            command.extend(['--objective', 'dice_bands'])
        print(json.dumps({'run_start': arm, 'updates': plan['updates'][arm]}), flush=True)
        subprocess.run(command, env=environment, check=True)
        for name, sha in manifest.items():
            if digest(source / name) != sha:
                raise ValueError('Source changed during training: ' + name)
        results[arm] = (verify_bands if bands else verify_baseline)(args.root, 'nnunet', protocol_dir)
        done = json.loads((output / 'completion.json').read_text())
        if done['updates'] != plan['updates'][arm] or done['discarded_attempts'] != 0:
            raise ValueError('Run did not complete its full allowance')
        write_json(args.root / 'summary.json', {'status': 'complete',
                   'model': 'nnunet', 'fold': 0, 'seed': 0, 'development_only': True,
                   'budget_gflops_per_run': plan['budget_gflops_per_run'], 'runs': results,
                   'final_delta_points': 100 * (results['bands']['final_validation_macro_dice']
                                                - results['baseline']['final_validation_macro_dice']) if bands else None})
        print(json.dumps({'run_complete': arm, 'result': results[arm]}), flush=True)


if __name__ == '__main__':
    main()
