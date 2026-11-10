"""Plain UNETR with the existing foreground squared Dice base loss.

Original auxiliary definitions remain in objectives.py. A process-local base
binding applies identically to optimization and training-only calibration.
"""
from contextlib import contextmanager
from pathlib import Path
import torch
from monai.networks.nets import UNETR

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.unetr_subfields import cap_extension as batch,foreground_squared_dice as dice
from scripts.mni_constraints import campaign,objectives
from scripts.mni_constraints import unetr_simple_recipe_runtime as plain

ORIGINAL_BIND=batch.bind
ORIGINAL_TRAIN=campaign.train
ORIGINAL_BASE_LOSS=objectives.base_loss
LOSS_RECIPE='unetr_foreground_squared_dice'


def validate(protocol):
    assert protocol['model_identity']=='unetr_vit_b16'
    assert protocol['loss_recipe']==LOSS_RECIPE
    assert protocol['loss_arguments']==dice.LOSS_ARGUMENTS
    assert protocol['base_loss_adapter_sha256']==digest(Path(__file__))
    assert protocol['base_loss_source_sha256']==digest(Path(dice.__file__))
    assert protocol['dropout_rate']==0. and protocol['augmentation']=='none'
    assert protocol['unetr_simplification']==dict(dropout_rate=0.,augmentation='none')
    assert protocol['initial_lrs']=={'unet':.0006} and protocol['lr_policy']=='hold_half'


def base_loss(outputs,labels,identity,criterion=None):
    if identity=='unetr_vit_b16':
        assert criterion is None
        return dice.training_loss(outputs,labels,'dice',{},1.)
    return ORIGINAL_BASE_LOSS(outputs,labels,identity,criterion)


@contextmanager
def use_base_loss():
    previous=objectives.base_loss
    objectives.base_loss=base_loss
    try:yield
    finally:objectives.base_loss=previous


def build_model(protocol,output=None):
    validate(protocol)
    kwargs=plain.model_kwargs(0.)
    model=UNETR(**kwargs)
    dropouts={name:module.p for name,module in model.named_modules() if isinstance(module,torch.nn.Dropout)}
    assert dropouts and all(p==0. for p in dropouts.values())
    assert sum(p.numel() for p in model.parameters())==92685572
    if output is not None:
        write(output/'model_recipe.json',dict(model='unetr_vit_b16',declared=protocol['unetr_simplification'],
            kwargs=kwargs,actual_dropout_modules=dropouts,parameters=92685572,
            runtime_source_sha256=digest(Path(__file__)),loss_recipe=LOSS_RECIPE,
            scope='Actual plain model; foreground squared Dice base; original auxiliary definitions unchanged.'))
    return model


def bind(args):
    protocol=read(args.protocol_dir/'protocol.json');validate(protocol)
    trainer=ORIGINAL_BIND(args)
    trainer.base.build_model=lambda alias,architecture:build_model(protocol,getattr(args,'output',None))
    trainer.check_no_augmentation=validate
    return trainer


def train(args):
    protocol=read(args.protocol_dir/'protocol.json');validate(protocol)
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'base_loss_recipe.json',dict(recipe=LOSS_RECIPE,arguments=dice.LOSS_ARGUMENTS,
        adapter_sha256=digest(Path(__file__)),loss_source_sha256=digest(Path(dice.__file__)),
        original_auxiliary_source_sha256=digest(Path(objectives.__file__)),
        protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        calibration='Same Dice-only base gradient; training crops only; original sequential calibration.',
        dense_cross_entropy_in_base=False))
    previous=batch.bind;batch.bind=bind
    try:
        with use_base_loss():ORIGINAL_TRAIN(args)
    finally:batch.bind=previous
