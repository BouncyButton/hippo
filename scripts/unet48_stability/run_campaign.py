"""Run a frozen list of isolated stability experiments in one allocation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    source = args.stage/'source'
    manifest = json.loads((args.stage/'source_manifest.json').read_text())
    campaign = json.loads((source/'experiments/unet48_stability/campaign.json').read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'PYTHONPATH': str(source)+':'+str(source/'scripts')}
    history = []
    for entry in campaign['runs']:
        for name, sha in manifest.items():
            assert digest(source/name) == sha, name
        protocol_dir = source/'experiments/unet48_stability'/entry['recipe']
        protocol = json.loads((protocol_dir/'protocol.json').read_text())
        assert digest(args.pkl) == protocol['pkl_sha256']
        run = args.root/entry['recipe']/f"seed{entry['seed']}"
        print(json.dumps({'start': entry, 'output': str(run)}), flush=True)
        adapter = entry.get('adapter', 'train_stability.py')
        assert adapter in ['train_stability.py', 'train_resize_decoder.py', 'train_linear_head.py']
        command = [sys.executable, str(source/'scripts/unet48_stability'/adapter),
                   '--model', 'unet', '--objective', 'dice', '--fold', '0', '--seed', str(entry['seed']),
                   '--pkl', str(args.pkl), '--output', str(run), '--protocol-dir', str(protocol_dir),
                   '--threads', '4', '--resume']
        subprocess.run(command, env=environment, check=True)
        done = json.loads((run/'completion.json').read_text())
        config = json.loads((run/'config.json').read_text())
        assert (done['status'], done['seed'], done['fold'], done['model'], done['objective']) == (
            'complete', entry['seed'], 0, 'unet', 'dice')
        assert config['protocol_sha256'] == digest(protocol_dir/'protocol.json')
        assert config['schedule'] == protocol['models']['unet']['objectives']['dice']
        assert done['discarded_attempts'] == 0
        assert done['updates'] == config['max_steps'] == campaign['max_updates']
        sys.path.insert(0, str(source/'scripts'))
        from run_budget_band_pilot import ARTIFACTS
        for key, name in ARTIFACTS.items():
            assert done[key] == digest(run/name), name
        for key, name in [('script_sha256','train_compute_matched_constraints.py'),
                          ('shared_source_sha256','train_compute_matched_baselines.py'),
                          ('objective_source_sha256','compute_constraint_objectives.py')]:
            assert config[key] == digest(source/'scripts'/name), name
        assert done['consumed_total_flops'] == protocol['budget_flops']
        with (run/'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        history.append({'recipe': entry['recipe'], 'seed': entry['seed'], 'output': str(run),
                        'completion_sha256': digest(run/'completion.json'),
                        'runtime_recipe_sha256': digest(run/'runtime_recipe.json'),
                        'final_macro_dice': float(rows[-1]['macro_hard_dice']),
                        'best_macro_dice': done['best_validation_hard_dice'],
                        'seconds': done['seconds_this_invocation'], 'curve': rows})
        (args.root/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps({'completed': history[-1]['recipe'], 'seed': entry['seed'],
                          'final_macro_dice': history[-1]['final_macro_dice']}), flush=True)
    (args.root/'campaign_completion.json').write_text(json.dumps(
        {'status': 'complete', 'runs': len(history), 'campaign_sha256': digest(source/'experiments/unet48_stability/campaign.json'),
         'history_sha256': digest(args.root/'history.json')}, indent=2)+'\n')


if __name__ == '__main__':
    main()
