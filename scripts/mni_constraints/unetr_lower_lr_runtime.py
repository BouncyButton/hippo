"""Protocol-bound plain UNETR with reduced peak learning rates and original DiceCE."""
from pathlib import Path
import torch
from monai.networks.nets import UNETR

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import campaign
from scripts.unetr_subfields import cap_extension as batch,geo_recipe

ORIGINAL_BIND=batch.bind
ORIGINAL_TRAIN=campaign.train


def validate(protocol):
    assert protocol['model_identity']=='unetr_vit_b16'
    declared=protocol['unetr_simplification']
    assert declared==dict(dropout_rate=protocol['dropout_rate'],augmentation=protocol['augmentation'])
    assert declared['dropout_rate'] in (0.,.1) and declared['augmentation'] in ('none','xy_d4_v1')
    assert protocol['loss_arguments']==geo_recipe.LOSS_ARGUMENTS
    assert protocol['unetr_peak_lr'] in (.00045,.0003)
    assert protocol['initial_lrs']=={'unet':protocol['unetr_peak_lr']}
    assert declared==dict(dropout_rate=0.,augmentation='none')
    assert protocol['lr_policy']=='cap_matched'
    return declared


def model_kwargs(dropout):
    assert dropout in (0.,.1)
    return dict(geo_recipe.model_kwargs(4,(64,64,64)),dropout_rate=dropout)


def build_model(protocol,output=None):
    declared=validate(protocol);kwargs=model_kwargs(declared['dropout_rate'])
    model=UNETR(**kwargs)
    dropouts={name:module.p for name,module in model.named_modules() if isinstance(module,torch.nn.Dropout)}
    assert dropouts and all(value==declared['dropout_rate'] for value in dropouts.values())
    assert sum(p.numel() for p in model.parameters())==92685572
    if output is not None:
        write(output/'model_recipe.json',dict(model='unetr_vit_b16',declared=declared,kwargs=kwargs,
            actual_dropout_modules=dropouts,parameters=92685572,
            runtime_source_sha256=digest(Path(__file__)),peak_lr=protocol['unetr_peak_lr'],scope='Actual constructed model; base loss, AdamW and LR policy retained from protocol.'))
    return model


def bind(args):
    protocol=read(args.protocol_dir/'protocol.json');validate(protocol)
    trainer=ORIGINAL_BIND(args)
    trainer.base.build_model=lambda alias,architecture:build_model(protocol,getattr(args,'output',None))
    def check(actual):
        validate(actual)
        assert actual['unetr_simplification']==protocol['unetr_simplification']
    trainer.check_no_augmentation=check
    return trainer


def train(args):
    previous=batch.bind;batch.bind=bind
    try:ORIGINAL_TRAIN(args)
    finally:batch.bind=previous
