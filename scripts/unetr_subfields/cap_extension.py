"""Bind the MNI architecture, optimizer and learning-rate schedule."""
import math

from scripts.cross_dataset_campaign.common import read

from scripts.unetr_subfields.contracts import DATASETS

def lr_at(step, maximum, initial, identity, policy):
    """A bounded, non-cycling schedule; long budget must not revive cosine LR."""
    if policy == 'constant': return initial
    if identity == 'nnunet':
        hold = maximum//2 if policy == 'hold_half' else 0
        progress = max(0., min(1., (step-hold)/max(1, maximum-hold)))
        return initial * (1-progress)**.9
    warmup = 24 if identity == 'unetr_vit_b16' else max(1, math.ceil(.05*maximum))
    if step < warmup: return initial*(step+1)/warmup
    decay_start=max(warmup,maximum//2) if policy=='hold_half' else warmup
    if step<decay_start:return initial
    progress = min(1., max(0., (step-decay_start)/max(1, maximum-decay_start-1)))
    return initial*(.01+.99*(1+math.cos(math.pi*progress))/2)

def bind(args):
    from functools import partial
    from scripts.mni_constraints import unetr_recipe
    from scripts.unetr_subfields import baseline_runtime, core, log_squared_dice

    protocol = read(args.protocol_dir / 'protocol.json')
    identity = protocol['model_identity']
    spec = DATASETS['MNI']
    base = core.base
    if identity == 'unetr_vit_b16':
        unetr_recipe.validate(protocol)
        base.build_model = lambda alias, arch: unetr_recipe.build_model(protocol, args.output)
    else:
        base.build_model = lambda alias, arch: baseline_runtime.build_model(identity, arch)
    if identity == 'swinunetr' and protocol['dice_loss_arguments'] != log_squared_dice.LOSS_ARGUMENTS:
        raise ValueError('SwinUNETR native loss differs from the recorded protocol')

    def check_augmentation(actual):
        expected = 'xy_d4_v1' if identity == 'swinunetr' else 'none'
        if actual['augmentation'] != expected:
            raise ValueError(f'Expected {expected} augmentation for MNI {identity}')

    core.trainer.check_no_augmentation = check_augmentation
    base.build_loss_and_optimizer = baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER
    base.PreparedPatients = partial(core.PreparedSubfields, spec=spec)
    base.evaluate = partial(core.evaluate, classes=spec['classes'])
    base.learning_rate = lambda alias, step, maximum, initial: lr_at(
        step, protocol['lr_horizon_updates'], initial, identity, protocol['lr_policy'])
    return core.trainer
