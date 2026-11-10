"""Bind frozen architecture, fixed LR trajectory, and common compute checkpoints."""
import argparse
import json
from pathlib import Path

import train_optimization90_core as trainer
from campaign90 import BUDGET, MODELS, ROOT, digest
from optimization90 import SCOPE, validation_updates


def configure(protocol, arm):
    identity = protocol['model_identity']
    spec = MODELS[identity]
    if protocol['budget_flops'] != BUDGET or protocol['lr_horizon_updates'] != spec['lr_horizon']:
        raise ValueError('Campaign budget or LR trajectory changed')
    original_factory = trainer.base.build_model
    if spec['candidate']:
        from baselines.simple_segmentation import build_candidate
        trainer.base.build_model = lambda name, architecture: build_candidate(spec['candidate'])
    else:
        trainer.base.build_model = original_factory
    original_lr = trainer.base.learning_rate
    trainer.base.learning_rate = lambda name, step, max_steps, initial, **kwargs: original_lr(
        name, step, spec['lr_horizon'], initial, protocol.get('nnunet_warmup_fraction', 0.0))
    trainer.base.validation_steps = lambda max_steps, count=20: validation_updates(protocol, arm, max_steps)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--model', required=True)
    args, _ = parser.parse_known_args()
    path = args.protocol_dir / 'protocol.json'
    protocol = json.loads(path.read_text())
    for name, sha in protocol['training_source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Frozen training source changed: ' + name)
    if args.model != protocol['trainer_alias'] or args.objective not in protocol['requested_arm_order']:
        raise ValueError('Run identity differs from protocol')
    if protocol.get('budget_scope') != SCOPE:
        raise ValueError('Optimization-only accounting must be explicit')
    configure(protocol, args.objective)
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {'model_identity': protocol['model_identity'], 'objective': args.objective,
              'protocol_sha256': digest(path), 'adapter_sha256': digest(Path(__file__)),
              'lr_horizon_updates': protocol['lr_horizon_updates'],
              'learning_rate_policy': protocol['learning_rate_policy'],
              'validation_updates': validation_updates(protocol, args.objective),
              'validation_target_flops': protocol['validation_target_flops'],
              'fresh_initialization': True, 'pretrained_weights': False,
              'budget_scope': SCOPE}
    target = args.output / 'runtime_recipe.json'
    if target.exists() and json.loads(target.read_text()) != recipe:
        raise ValueError('Resume adapter recipe changed')
    trainer.base.atomic_json(target, recipe)
    trainer.main()


if __name__ == '__main__':
    main()
