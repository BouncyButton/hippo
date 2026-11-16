"""Complete the three approved models with fixed recipes; never launch nnU-Net."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
from pathlib import Path
import time

from scripts.colab_hfh.bands_campaign import read, write
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.constraint_schedule import make_payload
from scripts.colab_hfh.constraint_search import recipes, run

MODELS = ('basic_unet16', 'unetr_vit_b16', 'swinunetr')


def prepare(state, plan, base):
    """Keep existing selections; transfer established component targets only to unselected arms."""
    state = deepcopy(state)
    queue = []
    objectives = list(dict.fromkeys(x['arm'] for x in plan['matrix'] if x['arm'] != 'dice'))
    for model in MODELS:
        anchor = state['selected'][model+'__dice_bands_depth']['specification']
        for arm in objectives:
            if arm == 'dice_bands' and model in plan['verified_band_recipes']:
                continue
            key = model+'__'+arm
            specs = recipes(model, plan, arm)
            if key not in state['selected']:
                names = components(arm)
                # Project the successful BAND+depth recipe onto the requested
                # components. These are already declared candidates, not a new
                # validation-driven search or a change to completed protocols.
                source_arm = 'dice_bands_depth' if 'bands' in names else 'dice_depth'
                source = state['selected'][model+'__'+source_arm]['specification']
                spec = {k: v for k, v in source.items() if k != 'component_targets'}
                if 'bands' in names and len(names) > 1:
                    targets = anchor['component_targets']
                    spec['component_targets'] = {
                        name: targets['bands'] if name == 'bands' else targets['depth']
                        for name in names}
                if spec not in specs:
                    raise ValueError('No declared transfer candidate for '+key)
                index = specs.index(spec)
                state['selected'][key] = dict(candidate=index, specification=spec,
                    qualification='fixed_transfer_for_complete_reporting',
                    transfer_source=model+'__'+source_arm)
            selected = state['selected'][key]
            index = selected['candidate']
            if specs[index] != selected['specification']:
                raise ValueError('Selected scientific recipe changed: '+key)
            name = f'{key}__r{index:02d}'
            record = state['attempts'].setdefault(name, dict(model=model, objective=arm,
                candidate=index, specification=specs[index], status='pending'))
            if record['specification'] != specs[index]:
                raise ValueError('Existing attempt changed: '+name)
            payload = make_payload(base, model, arm, plan['cost_references'][model]['canonical_costs'],
                                   **specs[index], screening=False)
            if payload.get('min_pair_delta_pp') is not None:
                raise ValueError('Full matrix must retain negative pairs')
            queue.append(dict(model=model, objective=arm, recipe=name,
                              complete=bool(record.get('full_matrix_complete'))))
    return state, queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    root = args.root
    with (root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, base = read(root/'constraint_matrix_plan.json'), read(root/'bands_v1_payload.json')
        for name, expected in plan['calibration_implementation']['source_sha256'].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Installed source differs from plan: '+name)
        path = root/'constraint_search/search_status.json'
        original = read(path)
        state, queue = prepare(original, plan, base)
        backup = path.with_name('before_three_model_completion_'+str(time.time_ns())+'.json')
        write(backup, original)
        state['fixed_completion_scope'] = dict(models=list(MODELS), excluded_models=['nnunet'],
            policy='Keep all selected recipes; declared transfer for unselected arms; complete all nine, including regressions',
            previous_current=original.get('current'), queue=queue)
        write(path, state)
        # Only selected runs for the explicitly resumed models may be resumed.
        # The paused nnU-Net trial and all lower-budget STOP files stay intact.
        for row in queue:
            if not row['complete']:
                (root/'constraint_search'/row['recipe']/'STOP').unlink(missing_ok=True)
        print('FIXED MATRIX MODELS', list(MODELS), 'NNUNET PAUSED', flush=True)
        print('PENDING OBJECTIVES', sum(not row['complete'] for row in queue), flush=True)
        run(argparse.Namespace(root=root, hours=args.hours, phase='matrix',
            models=list(MODELS), optimize_mean=False, pilot=False, objectives=None))


if __name__ == '__main__':
    main()
