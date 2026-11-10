"""Bind the unchanged nnU-Net LR trajectory to the isolated SDF timing pilot."""
import argparse
import json
from pathlib import Path

from campaign90 import BUDGET, ROOT, digest
from delayed_sdf_only_schedule import validation_updates
import train_delayed_sdf_only_core as trainer


def configure(protocol, arm):
    if (protocol['budget_flops'] != BUDGET or protocol['lr_horizon_updates'] != 174
            or protocol['model_identity'] != 'nnunet' or arm != 'dice_sdf'
            or protocol['constraint_warmup_updates'] != 9):
        raise ValueError('Frozen pilot identity differs')
    original_lr = trainer.base.learning_rate
    trainer.base.learning_rate = lambda name, step, maximum, initial, **kwargs: original_lr(
        name, step, 174, initial, 0.)
    trainer.base.validation_steps = lambda maximum, count=20: validation_updates(protocol, arm, maximum)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--objective',required=True)
    args, _ = parser.parse_known_args()
    path = args.protocol_dir / 'protocol.json'
    protocol = json.loads(path.read_text())
    if args.model != 'nnunet':
        raise ValueError('nnU-Net only')
    for name, sha in protocol['training_source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Training source changed: ' + name)
    configure(protocol,args.objective)
    args.output.mkdir(parents=True,exist_ok=True)
    recipe = dict(model_identity='nnunet',objective=args.objective,
                  protocol_sha256=digest(path),adapter_sha256=digest(Path(__file__)),
                  lr_horizon_updates=174,learning_rate_policy=protocol['learning_rate_policy'],
                  validation_updates=validation_updates(protocol,args.objective),
                  validation_target_flops=protocol['validation_target_flops'],
                  fresh_initialization=True,pretrained_weights=False,budget_scope=protocol['budget_scope'],
                  sdf_prefix_updates=50,sdf_warmup_updates=9,
                  calibration_reference_sha256=protocol['calibration_reference_sha256'])
    target = args.output / 'runtime_recipe.json'
    if target.exists() and json.loads(target.read_text()) != recipe:
        raise ValueError('Resume recipe differs')
    trainer.base.atomic_json(target,recipe)
    trainer.main()


if __name__ == '__main__':
    main()
