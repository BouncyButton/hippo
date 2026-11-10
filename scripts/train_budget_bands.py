"""Apply calibrated bands without changing each baseline architecture or LR policy."""
import argparse
import json
from pathlib import Path

import train_compute_matched_constraints as trainer


def validation_updates(protocol, model, max_steps=None):
    schedule = protocol['models'][model]['objectives']['dice_bands']
    cap = min(schedule['max_steps'], max_steps) if max_steps is not None else schedule['max_steps']
    prefix, cost, baseline, calibration = (schedule[k] for k in
        ['prefix_updates', 'flops_per_update', 'baseline_flops_per_update', 'calibration_flops'])
    if cap < 1:
        raise ValueError('Invalid update cap')
    result = {cap}
    for target in protocol['validation_target_flops']:
        if not isinstance(target, int) or not 0 < target <= protocol['budget_flops']:
            raise ValueError('Invalid validation compute target')
        if target <= prefix * baseline + calibration:
            updates = min(prefix, target // baseline)
        else:
            updates = prefix + (target - prefix * baseline - calibration) // cost
        result.add(max(1, min(cap, updates)))
    return sorted(result)


def configure(protocol, model):
    if protocol['campaign_adapter_sha256'] != trainer.base.digest(Path(__file__)):
        raise ValueError('Bands adapter differs from frozen protocol')
    if protocol.get('candidate_name') == 'basic_unet16':
        from baselines.simple_segmentation import CANDIDATES, build_candidate
        factory = Path(__file__).resolve().parents[1] / 'baselines/simple_segmentation.py'
        # JSON turns tuples into lists; compare the serialized architecture recipe.
        candidate = json.loads(json.dumps(CANDIDATES['basic_unet16']))
        if (model != 'unet' or protocol['candidate'] != candidate
                or protocol['simple_factory_sha256'] != trainer.base.digest(factory)
                or protocol['initial_lrs']['unet'] != .0003):
            raise ValueError('Plain width-16 U-Net differs from the paired baseline')
        trainer.base.build_model = lambda name, architecture: build_candidate('basic_unet16')
    elif model == 'unet':
        from baselines.residual_unet import build_unet48, absolute_learning_rate
        factory = Path(__file__).resolve().parents[1] / 'baselines/residual_unet.py'
        if (protocol['unet_factory_sha256'] != trainer.base.digest(factory)
                or protocol['unet_output_head'] != 'linear'
                or protocol['unet_lr_policy'] != 'absolute_warmup_constant'
                or protocol['unet_warmup_updates'] != 100):
            raise ValueError('Corrected U-Net architecture or schedule differs')
        trainer.base.build_model = lambda name, architecture: build_unet48()
        trainer.base.learning_rate = lambda name, step, max_steps, initial, **kwargs: absolute_learning_rate(step, initial, 100)
    trainer.base.validation_steps = lambda max_steps, count=20: validation_updates(protocol, model, max_steps)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', choices=['unet', 'swinunetr', 'nnunet'], required=True)
    parser.add_argument('--objective', choices=['dice_bands'], required=True)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text())
    configure(protocol, args.model)
    recipe = {'adapter_sha256': trainer.base.digest(Path(__file__)),
              'protocol_sha256': trainer.base.digest(args.protocol_dir / 'protocol.json'),
              'model': args.model, 'objective': 'dice_bands',
              'validation_target_flops': protocol['validation_target_flops'],
              'validation_updates': validation_updates(protocol, args.model),
              'learning_rate_policy': protocol['learning_rate_policy'],
              'baseline_job_id': protocol['baseline_job_id']}
    if 'candidate_name' in protocol:
        recipe.update(candidate=protocol['candidate_name'], architecture=protocol['candidate'])
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'runtime_recipe.json'
    if target.exists() and json.loads(target.read_text()) != recipe:
        raise ValueError('Resume adapter recipe changed')
    trainer.base.atomic_json(target, recipe)
    trainer.main()


if __name__ == '__main__':
    main()
