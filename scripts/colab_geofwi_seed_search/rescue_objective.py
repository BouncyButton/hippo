"""Frozen rescue recipe shared by CPU profiling and CUDA training."""
from copy import deepcopy
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR

LOSS_ARGUMENTS = dict(include_background=False, to_onehot_y=True, softmax=True,
    squared_pred=True, smooth_nr=0., smooth_dr=1e-6, lambda_dice=1., lambda_ce=1.)


def build_model(identity, architecture):
    if identity != 'unetr_vit_b16':
        raise ValueError('UNETR-only rescue')
    from baselines.simple_segmentation import CANDIDATES
    kwargs = deepcopy(CANDIDATES[identity]['kwargs'])
    kwargs.update(out_channels=2, img_size=(96,96,96), dropout_rate=.1)
    return UNETR(**kwargs)


def training_loss(outputs, labels, arm, weights, scale, sdf=None,
                  criterion=None, depth=None, sdf_scale=0.):
    if arm != 'dice' or criterion is not None or weights:
        raise ValueError('Rescue permits only the frozen foreground Dice+CE recipe')
    return DiceCELoss(**LOSS_ARGUMENTS)(outputs.float(), labels)


def dropout_audit(model):
    values = {name:m.p for name,m in model.named_modules() if isinstance(m,torch.nn.Dropout)}
    if not values or set(values.values()) != {.1}:
        raise ValueError('Dropout wiring changed')
    return values
