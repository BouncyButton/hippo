"""Run the authorized U-Net bands fallback after an unsuitable UNETR pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_budget_baselines import digest, write_json
from run_budget_bands import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--unetr-root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'source'
    manifest = json.loads((args.root / 'source_manifest.json').read_text())
    for name, sha in manifest.items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    unetr = json.loads((args.unetr_root / 'summary.json').read_text())
    if (unetr['status'] != 'complete' or unetr['model'] != 'unetr_vit_b16'
            or unetr['run']['updates'] != 477 or unetr['run']['discarded_attempts'] != 0
            or all(.70 <= unetr['run'][k] <= .80 for k in ['best_dice', 'final_dice'])):
        raise ValueError('The authorized fallback condition is not met')
    directory = source / 'experiments/plain_unet16_bands_20261005/dice_bands'
    protocol = json.loads((directory / 'protocol.json').read_text())
    schedule = protocol['models']['unet']['objectives']['dice_bands']
    if (protocol['candidate_name'] != 'basic_unet16' or protocol['folds'] != [0]
            or protocol['seeds'] != [0] or schedule['max_steps'] != 1091
            or protocol['initial_lrs']['unet'] != .0003
            or protocol['budget_flops'] != 62166546655000):
        raise ValueError('Unexpected fallback recipe')
    output = args.root / 'unet/dice_bands/fold0/seed0/run'
    environment = {**os.environ, 'PYTHONPATH': str(source) + ':' + str(source / 'scripts')}
    subprocess.run([sys.executable, str(source / 'scripts/train_budget_bands.py'),
                    '--model', 'unet', '--objective', 'dice_bands', '--fold', '0', '--seed', '0',
                    '--protocol-dir', str(directory), '--output', str(output), '--pkl', str(args.pkl),
                    '--threads', '4', '--resume'], env=environment, check=True)
    for name, sha in manifest.items():
        if digest(source / name) != sha:
            raise ValueError('Source changed while training: ' + name)
    result = verify_run(args.root, 'unet', directory)
    baseline = json.loads((source / 'reports/plain_unet16_lr3e4_20261005/summary_final.json').read_text())
    if result['updates'] != 1091 or result['calibration_attempts'] != 1:
        raise ValueError('Fallback schedule was not completed exactly')
    write_json(args.root / 'summary.json', {
        'status': 'complete', 'candidate': 'basic_unet16', 'objective': 'dice_bands',
        'baseline_job_id': 684625, 'development_only': True, 'run': result,
        'baseline_final_dice': baseline['run']['final_dice'],
        'final_delta_points': 100 * (result['final_validation_macro_dice'] - baseline['run']['final_dice']),
        'unetr_result_sha256': digest(args.unetr_root / 'summary.json')})
    print(json.dumps({'bands_complete': result}), flush=True)


if __name__ == '__main__':
    main()
