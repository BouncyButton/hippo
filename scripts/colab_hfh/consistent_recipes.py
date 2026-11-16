"""Find model recipes that pass the same fixed seeds and patient folds."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
try:
    from . import recipe_transport as transport
except ImportError:
    import recipe_transport as transport


RECIPES = {
    'unetr_vit_b16': [0.0003, 0.001, 0.0001],
    'nnunet': [0.1, 0.05, 0.2],
    'basic_unet16': [0.0003, 0.001, 0.0001],
    'swinunetr': [0.0001, 0.0003, 0.001],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=6)
    args = parser.parse_args()
    with (args.root/'consistent_recipes.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args):
    precision = json.loads((args.root/'bundle/gpu_overlay.json').read_text())['precision']
    suffix = '' if precision == 'bf16' else '_'+precision
    transport.restore(args.root, suffix)
    deadline = time.time()+args.hours*3600
    state_path = args.root/('consistent_recipes_status'+suffix+'.json')
    state = dict(status='running', seeds=[0, 1, 2], folds=[0, 1, 2],
        threshold=0.5, selected_recipes={}, attempts=[],
        selection_warning='Exploratory recipe selection on validation; no unbiased test claim')
    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)
    try:
        for model, rates in RECIPES.items():
            for lr in rates:
                scores = []
                failed = False
                for fold in (0, 1, 2):
                    for seed in (0, 1, 2):
                        if time.time() >= deadline or (args.root/'STOP_RECIPE_SEARCH').exists():
                            state['status'] = 'paused'; save(); return
                        state['current'] = dict(model=model, lr=lr, fold=fold, seed=seed); save()
                        label = f'{model}_lr{lr:g}_poly_cap'
                        completed = args.root/('recipe_diagnostics'+suffix)/label/'archives'/f'{model}_f{fold}_s{seed}'/'cleanup.json'
                        # Cached results are validated by recipe_probe; they charge no new work.
                        if not completed.exists():
                            transport.publish_status(state_path, suffix)
                        subprocess.run([sys.executable, str(args.root/'recipe_probe.py'),
                            '--root', str(args.root), '--model', model, '--lr', str(lr),
                            '--folds', str(fold), '--seeds', str(seed)], check=True)
                        result = json.loads((args.root/('recipe_diagnostics'+suffix)/label/'archives'/
                            f'{model}_f{fold}_s{seed}'/'result.json').read_text())
                        if not math.isfinite(result['validation_dice']) or not 0 <= result['validation_dice'] <= 1:
                            raise ValueError('Invalid final validation Dice; inspect numerical diagnostics')
                        scores.append(result['validation_dice'])
                        state['attempts'].append(dict(lr=lr, **result)); save()
                        if result['validation_dice'] < 0.5:
                            print('REJECT RECIPE', model, lr, fold, seed, result['validation_dice'], flush=True)
                            failed = True; break
                    if failed:
                        break
                if not failed:
                    state['selected_recipes'][model] = dict(initial_lr=lr,
                        minimum_dice=min(scores), mean_dice=sum(scores)/len(scores), cells=9)
                    print('MODEL PASSED ALL NINE', model, state['selected_recipes'][model], flush=True)
                    save(); break
            if model not in state['selected_recipes']:
                state['status'] = 'needs_recipe_review'; state['struggling_model'] = model
                save(); return
        state['status'] = 'all_four_models_passed_same_nine_cells'; save()
        transport.publish_status(state_path, suffix)
    except Exception as error:
        state.update(status='runtime_error', error=str(error)); save(); raise


if __name__ == '__main__':
    main()
