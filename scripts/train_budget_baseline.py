"""Run a frozen baseline with common absolute-compute validation targets."""
import argparse
import json
from pathlib import Path

import train_compute_matched_baselines as base


def validation_updates(protocol, model, max_steps=None):
    schedule = protocol['models'][model]
    cost = (schedule['objectives']['dice']['flops_per_update'] if model == 'unet'
            else schedule['flops_per_update'])
    cap = protocol['budget_flops'] // cost
    if max_steps is not None:
        cap = min(cap, max_steps)
    targets = protocol['validation_target_flops']
    if cap < 1 or any(not isinstance(t, int) or not 0 < t <= protocol['budget_flops'] for t in targets):
        raise ValueError('Invalid validation compute target or update cap')
    return sorted({max(1, min(cap, t // cost)) for t in targets} | {cap})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', required=True, choices=['unet', 'swinunetr', 'nnunet'])
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text())
    if protocol['campaign_adapter_sha256'] != base.digest(Path(__file__)):
        raise ValueError('Baseline adapter differs from frozen protocol')
    base.validation_steps = lambda max_steps, count=20: validation_updates(protocol, args.model, max_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {'adapter_sha256': base.digest(Path(__file__)),
              'protocol_sha256': base.digest(args.protocol_dir / 'protocol.json'),
              'model': args.model, 'validation_target_flops': protocol['validation_target_flops'],
              'validation_updates': validation_updates(protocol, args.model),
              'learning_rate_policy': protocol['learning_rate_policy']}
    target = args.output / 'runtime_recipe.json'
    if target.exists() and json.loads(target.read_text()) != recipe:
        raise ValueError('Resume adapter recipe changed')
    base.atomic_json(target, recipe)
    if args.model == 'unet':
        import train_unet48_budget_pilot
        train_unet48_budget_pilot.main()
    else:
        base.main()


if __name__ == '__main__':
    main()
