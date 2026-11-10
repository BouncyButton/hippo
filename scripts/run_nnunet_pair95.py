"""Train the authorized nnU-Net baseline and bands sequentially in one allocation."""
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
    directory = source / 'experiments/nnunet_pair95_20261005'
    plan = json.loads((directory / 'plan.json').read_text())
    if (plan['model_order'] != ['nnunet'] or plan['arm_order'] != ['baseline', 'bands']
            or plan['budget_flops_per_run'] != 33752920148595
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
        script = 'train_nnunet_pair95.py'
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
        import csv, math
        control = json.loads((output / 'schedule_control.json').read_text())
        if (control['lr_horizon_updates'] != 174 or control['training_cap_updates'] != plan['updates'][arm]
                or control['schedule_adapter_sha256'] != digest(source / 'scripts/train_nnunet_pair95.py')):
            raise ValueError('95-update schedule evidence differs')
        rows = list(csv.DictReader((output / 'metrics.csv').open()))
        if [int(row['step']) for row in rows] != plan['validation_updates'][arm]:
            raise ValueError('Equal-update validation schedule differs')
        for row in rows:
            if not math.isclose(float(row['learning_rate']), .01 * (1 - (int(row['step']) - 1) / 174) ** .9, rel_tol=1e-12):
                raise ValueError('LR trajectory differs')
        done = json.loads((output / 'completion.json').read_text())
        if done['updates'] != plan['updates'][arm] or done['discarded_attempts'] != 0:
            raise ValueError('Run did not complete its full allowance')
        write_json(args.root / 'summary.json', {'status': 'complete' if bands else 'in_progress',
                   'model': 'nnunet', 'fold': 0, 'seed': 0, 'development_only': True,
                   'maximum_common_budget_gflops': plan['budget_gflops_per_run'], 'comparison_axis': 'training_flops', 'arm_budget_flops': plan['arm_budget_flops'], 'runs': results,
                   'final_delta_points': 100 * (results['bands']['final_validation_macro_dice']
                                                - results['baseline']['final_validation_macro_dice']) if bands else None})
        print(json.dumps({'run_complete': arm, 'result': results[arm]}), flush=True)


if __name__ == '__main__':
    main()
