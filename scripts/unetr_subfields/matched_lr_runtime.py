"""MNI Dice+CE baselines with learning-rate horizons matched to the FLOP cap."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import fcntl
import traceback
from monai.networks.nets import BasicUNet, SwinUNETR
from baselines.simple_segmentation import CANDIDATES
from scripts.unetr_subfields import core
from scripts.capped_depth_delay import objectives
from scripts.cross_dataset_campaign import conventional
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.unetr_subfields.contracts import DATASETS

MODELS = {
    'nnunet': dict(alias='nnunet', lr=.01, loss='Native foreground Dice+CE with deep supervision; SGD/Nesterov momentum0.95'),
    'swinunetr': dict(alias='swinunetr', lr=.0001, loss='Existing MONAI softmax Dice including background + mean cross-entropy, weights1:1; AdamW'),
}


def build_model(identity, architecture, classes=4):
    if identity == 'swinunetr':
        return SwinUNETR(in_channels=1, out_channels=classes, use_checkpoint=False, drop_rate=0.)
    if identity == 'basic_unet16':
        kwargs = deepcopy(CANDIDATES[identity]['kwargs']); kwargs['out_channels'] = classes
        return BasicUNet(**kwargs)
    if identity != 'nnunet' or architecture['kwargs']['num_classes'] != classes:
        raise ValueError('Unexpected baseline architecture')
    return conventional.ORIGINAL_BUILDER('nnunet', architecture)


def bind(protocol):
    identity = protocol['model_identity']; spec = DATASETS['MNI']; model = MODELS[identity]
    if protocol['dataset'] != 'MNI' or protocol['requested_arm_order'] != ['dice'] or protocol['augmentation'] != 'none' or protocol['constraints']:
        raise ValueError('Only MNI native baseline recipes are allowed')
    if protocol['classes'] != spec['classes'] or protocol['spatial_size'] != list(spec['spatial_size']):
        raise ValueError('MNI geometry or classes changed')
    base = core.base
    base.build_model = lambda alias, architecture: build_model(identity, architecture)
    base.build_loss_and_optimizer = conventional.ORIGINAL_LOSS_OPTIMIZER
    base.PreparedPatients = partial(core.PreparedSubfields, spec=spec)
    base.evaluate = partial(core.evaluate, classes=spec['classes'])
    horizon = protocol['models'][model['alias']]['objectives']['dice']['max_steps']
    if protocol['lr_horizon_updates'] != horizon:
        raise ValueError('LR horizon must equal the FLOP-limited update count')
    base.learning_rate = lambda name, step, maximum, initial: core.ORIGINAL_LR(name, step, horizon, initial, 0.)
    cost = protocol['models'][model['alias']]['objectives']['dice']['flops_per_update']
    base.validation_steps = lambda maximum, count=20: sorted({min(maximum,max(1,target//cost)) for target in protocol['validation_target_flops']} | {maximum})
    from scripts.unetr_subfields.ce_loss import training_loss as dice_ce
    core.trainer.training_loss = dice_ce if identity == 'swinunetr' else objectives.training_loss
    core.trainer.ARMS = ('dice',)
    return core.trainer


def train(args):
    protocol=read(args.protocol_dir/'protocol.json');trainer=bind(protocol)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        with (args.output/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write(args.output/'runtime_recipe.json',dict(dataset='MNI',model=protocol['model_identity'],
                objective=protocol['training_loss'], loss_recipe='dice_ce', augmentation='none', constraints=False,
                lr_horizon_updates=protocol['lr_horizon_updates'], learning_rate_policy=protocol['learning_rate_policy'],
                initial_lr=protocol['initial_lrs'][args.model],
                initialization='fresh_random_from_declared_seed', primary_checkpoint='final_budget',
                protocol_sha256=digest(args.protocol_dir/'protocol.json'), adapter_sha256=digest(Path(__file__))))
            trainer.run(args)
    except Exception:
        write(args.output/'failure.json',dict(traceback=traceback.format_exc()));raise
