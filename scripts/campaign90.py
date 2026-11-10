"""Frozen identities and scheduling helpers for the four-model, six-loss campaign."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'four_models_six_31976_20261006'
BUDGET = 90 * 355293896301
ARMS = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf',
        'dice_bands_edge', 'dice_bands_edge_sdf')
MODELS = {
    'nnunet': {'alias': 'nnunet', 'candidate': None, 'parameters': 16548076,
               'baseline_flops': 355293896301, 'initial_lr': .01, 'lr_horizon': 174},
    'swinunetr': {'alias': 'swinunetr', 'candidate': None, 'parameters': 15703029,
                  'baseline_flops': 155012136022, 'initial_lr': .0001, 'lr_horizon': 610},
    'unetr_vit_b16': {'alias': 'unet', 'candidate': 'unetr_vit_b16', 'parameters': 92685555,
                     'baseline_flops': 130061273028, 'initial_lr': .0001, 'lr_horizon': 477},
    'basic_unet16': {'alias': 'unet', 'candidate': 'basic_unet16', 'parameters': 1438819,
                     'baseline_flops': 56727085196, 'initial_lr': .0003, 'lr_horizon': 1095},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_model(identity, architecture):
    spec = MODELS[identity]
    if spec['candidate']:
        from baselines.simple_segmentation import build_candidate
        return build_candidate(spec['candidate'])
    from train_compute_matched_baselines import build_model as original
    return original(spec['alias'], architecture)


def validation_updates(protocol, arm, cap=None):
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
    cap = schedule['max_steps'] if cap is None else min(cap, schedule['max_steps'])
    prefix, base, cost, calibration = (schedule[k] for k in
        ['prefix_updates', 'baseline_flops_per_update', 'flops_per_update', 'calibration_flops'])
    result = {cap}
    for target in protocol['validation_target_flops']:
        if target <= prefix * base + calibration:
            step = min(prefix, target // base) if calibration else target // cost
        else:
            step = prefix + (target - prefix * base - calibration) // cost
        if step > 0:
            result.add(min(cap, step))
    # Also record matched-update checkpoints for interpreting calibration overhead.
    stride = 5 if protocol['model_identity'] == 'nnunet' else max(
        5, (BUDGET // MODELS[protocol['model_identity']]['baseline_flops']) // 20)
    result.update(range(stride, cap + 1, stride))
    return sorted(result)
