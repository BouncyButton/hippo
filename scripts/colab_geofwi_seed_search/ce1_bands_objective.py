"""Frozen rescue recipe shared by CPU profiling and CUDA training."""
from copy import deepcopy
import math
import torch
from scripts.adni_constraints.objectives import bands
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
    if arm not in ('dice','dice_bands') or criterion is not None or sdf is not None or depth is not None:
        raise ValueError('Only Dice+CE and matched binary BANDS are allowed')
    loss = DiceCELoss(**LOSS_ARGUMENTS)(outputs.float(), labels)
    if arm == 'dice_bands':
        if set(weights) != {'bands'} or not math.isfinite(weights['bands']) or weights['bands'] <= 0:
            raise ValueError('Missing or invalid calibrated BANDS coefficient')
        loss = loss + float(scale) * weights['bands'] * bands()(outputs.float(), labels).loss
    elif weights:
        raise ValueError('Baseline must not contain constraint weights')
    return loss


def calibration_case(outputs, labels, arm, sdf=None, criterion=None, depth=None):
    if arm != 'dice_bands' or any(x is not None for x in (sdf,criterion,depth)):
        raise ValueError('Binary BANDS calibration only')
    z=outputs.detach().float().requires_grad_(True)
    base=training_loss(z,labels,'dice',{},0.)
    gradient,=torch.autograd.grad(base,z)
    base_rms=gradient.square().mean().sqrt()
    result=bands()(z,labels)
    if not bool(result.details['valid'].all()):raise ValueError('Invalid calibration geometry')
    gradient,=torch.autograd.grad(result.loss,z)
    band_rms=gradient.square().mean().sqrt()
    row=dict(dice_rms=float(base_rms),bands_rms=float(band_rms),bands_over_dice=float(band_rms/base_rms))
    if any(not math.isfinite(v) or v<=0 for v in row.values()):raise ValueError('Invalid gradient ratio')
    return row


def dropout_audit(model):
    values = {name:m.p for name,m in model.named_modules() if isinstance(m,torch.nn.Dropout)}
    if not values or set(values.values()) != {.1}:
        raise ValueError('Dropout wiring changed')
    return values
