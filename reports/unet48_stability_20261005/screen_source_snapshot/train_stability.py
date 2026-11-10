"""Isolated, protocol-bound interventions on the original width-48 trainer."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def fixed_learning_rate(step, initial, warmup_updates=100):
    """Absolute warmup followed by constant LR, independent of the stop budget."""
    if step < 0 or warmup_updates < 1:
        raise ValueError('Invalid update or warmup')
    return initial * min(1., (step + 1) / warmup_updates)


def install(recipe):
    import torch
    from monai.losses import DiceLoss
    import compute_constraint_objectives as objectives
    import train_compute_matched_constraints as trainer
    from train_unet48_budget_pilot import KWARGS

    def build_model(name, architecture):
        assert name == 'unet'
        return trainer.base.UNet(**KWARGS)

    def build_loss_optimizer(name, model, protocol):
        assert name == 'unet'
        return (DiceLoss(to_onehot_y=True, softmax=True, include_background=recipe['include_background']),
                torch.optim.AdamW(model.parameters(), lr=protocol['initial_lrs'][name],
                                  betas=tuple(recipe['betas']), weight_decay=protocol['adamw_weight_decay'],
                                  foreach=False))

    def learning_rate(name, step, max_steps, initial, **kwargs):
        assert name == 'unet'
        return fixed_learning_rate(step, initial, recipe['warmup_updates'])

    trainer.base.build_model = build_model
    trainer.base.build_loss_and_optimizer = build_loss_optimizer
    trainer.base.learning_rate = learning_rate
    objectives.DiceLoss = partial(DiceLoss, include_background=recipe['include_background'])
    return trainer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text())
    assert protocol['stability_adapter_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    recipe = protocol['stability_recipe']
    assert recipe['schedule'] == 'absolute_warmup_constant'
    assert recipe['warmup_updates'] == 100
    assert len(recipe['betas']) == 2 and all(0 < b < 1 for b in recipe['betas'])
    assert isinstance(recipe['include_background'], bool)
    trainer = install(recipe)
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output / 'runtime_recipe.json'
    runtime = {'recipe': recipe, 'adapter_sha256': protocol['stability_adapter_sha256'],
               'protocol_sha256': trainer.base.digest(args.protocol_dir / 'protocol.json')}
    if runtime_path.exists():
        assert json.loads(runtime_path.read_text()) == runtime
    else:
        trainer.base.atomic_json(runtime_path, runtime)
    trainer.main()


if __name__ == '__main__':
    main()
