"""Run a bounded, baseline-only architecture screen in one allocation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_budget_baselines import digest, write_json


def verify_run(output, directory, source):
    protocol = json.loads((directory / 'protocol.json').read_text())
    done = json.loads((output / 'completion.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    for key, filename in {
        'config_sha256': 'config.json', 'metrics_sha256': 'metrics.csv',
        'checkpoint_best_sha256': 'checkpoint_best.pt',
        'checkpoint_latest_sha256': 'checkpoint_latest.pt',
        'budget_ledger_sha256': 'budget_ledger.json',
        'validation_best_sha256': 'validation_best.json',
    }.items():
        if done[key] != digest(output / filename):
            raise ValueError('Changed completed artifact: ' + filename)
    cost, budget = protocol['models']['unet']['flops_per_update'], protocol['budget_flops']
    cap = budget // cost
    if (done['status'] != 'complete' or config['smoke_test']
            or done['training_attempts_charged'] != cap
            or done['consumed_training_flops'] != cap * cost
            or done['unused_training_flops'] != budget - cap * cost
            or config['protocol_sha256'] != digest(directory / 'protocol.json')
            or config['dataset_sha256'] != protocol['pkl_sha256']
            or config['script_sha256'] != digest(source / 'scripts/train_compute_matched_baselines.py')):
        raise ValueError('Completion budget, identity or source differs')
    with (output / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    final, best = float(rows[-1]['macro_hard_dice']), done['best_validation_hard_dice']
    if (not all(math.isfinite(v) and 0 <= v <= 1 for v in [final, best])
            or best != max(float(row['macro_hard_dice']) for row in rows)
            or int(rows[-1]['charged_attempts']) != cap):
        raise ValueError('Validation curve differs from completion')
    split = json.loads((directory / 'splits_10patients.json').read_text())[0]
    latest = json.loads((output / 'validation_latest.json').read_text())
    if ([case['case_name'] for case in latest['cases']] != split['val']
            or config['train_cases'] != split['train'] or config['validation_cases'] != split['val']
            or latest['summary']['macro_hard_dice'] != final):
        raise ValueError('Validation data or final score differs')
    return {'candidate': protocol['candidate_name'], 'initial_lr': config['initial_lr'],
            'updates': done['updates'], 'epochs': done['updates'] / 10,
            'parameters': protocol['models']['unet']['parameters'],
            'consumed_gflops': done['consumed_training_flops'] / 1e9,
            'discarded_attempts': done['discarded_attempts'],
            'best_dice': best, 'final_dice': final,
            'best_step': done['selected_step'],
            'completion_sha256': digest(output / 'completion.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'source'
    expected = json.loads((args.root / 'source_manifest.json').read_text())
    plan_dir = source / 'experiments/simple_baseline_screen_20261005'
    plan = json.loads((plan_dir / 'plan.json').read_text())
    if digest(args.pkl) != plan['pkl_sha256']:
        raise ValueError('Dataset differs from screen')
    history = []
    environment = {**os.environ, 'PYTHONPATH': str(source) + ':' + str(source / 'scripts')}
    selected = None
    for candidate in plan['candidate_order']:
        candidate_results = []
        for run in [r for r in plan['runs'] if r['candidate'] == candidate]:
            for name, sha in expected.items():
                if digest(source / name) != sha:
                    raise ValueError('Frozen source changed: ' + name)
            directory = plan_dir / run['name']
            output = args.root / run['name'] / 'fold0/seed0/run'
            print(json.dumps({'pilot_start': run['name']}), flush=True)
            subprocess.run([sys.executable, str(source / 'scripts/train_simple_baseline_pilot.py'),
                            '--model', 'unet', '--fold', '0', '--seed', '0',
                            '--protocol-dir', str(directory), '--output', str(output),
                            '--pkl', str(args.pkl), '--threads', '4', '--resume'],
                           env=environment, check=True)
            result = verify_run(output, directory, source)
            result['run_name'] = run['name']
            history.append(result)
            candidate_results.append(result)
            write_json(args.root / 'summary.json', {'status': 'in_progress', 'runs': history})
            print(json.dumps({'pilot_complete': result}), flush=True)
        strongest = max(candidate_results, key=lambda r: r['best_dice'])
        # A transient low checkpoint or weak LR cannot qualify a converged model.
        if (plan['target_min'] <= strongest['best_dice'] <= plan['target_max']
                and plan['target_min'] <= strongest['final_dice'] <= plan['target_max']
                and all(r['discarded_attempts'] == 0 for r in candidate_results)):
            selected = strongest
            break
    write_json(args.root / 'summary.json', {
        'status': 'complete', 'budget_gflops_per_run': plan['budget_flops'] / 1e9,
        'development_only': True, 'runs': history, 'selected': selected,
        'selection_rule': plan['selection_rule'],
        'total_search_training_gflops': sum(r['consumed_gflops'] for r in history),
        'next_step': 'Confirm architecture, LR and budget on held-out seeds/folds before constraint claims.'
        if selected else 'No tested candidate met the predeclared target; do not force a weak baseline.'})


if __name__ == '__main__':
    main()
