"""Add the existing bands to nnU-Net's native deep-supervised objective."""
import math

import torch

from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.constraint_result import differentiable_zero


def native_loss(outputs, labels, criterion):
    targets = [torch.nn.functional.interpolate(labels.float(), size=o.shape[2:],
                                              mode='nearest').long() for o in outputs]
    return criterion(outputs, targets)


def training_loss(outputs, labels, arm, weights, scale, criterion):
    loss = native_loss(outputs, labels, criterion)
    if arm == 'dice':
        return loss
    if arm != 'dice_bands':
        raise ValueError('Only native baseline plus bands is supported for nnU-Net')
    logits = outputs[0]
    auxiliary = differentiable_zero(logits.float())
    auxiliary = auxiliary + weights['bands'] * OuterBoundaryBandLoss()(logits, labels).loss
    return loss + scale * auxiliary


def calibration_case(outputs, labels, arm, criterion):
    """Compare both gradients in the full-resolution logit space of the bands."""
    if arm != 'dice_bands':
        raise ValueError('nnU-Net calibration requires the bands objective')
    detached = [o.detach().float() for o in outputs]
    z = detached[0].requires_grad_(True)
    gd = torch.autograd.grad(native_loss(detached, labels, criterion), z)[0]
    dr = gd.square().mean().sqrt()
    band = OuterBoundaryBandLoss()(z, labels)
    if not bool(band.details['valid'].all()):
        raise ValueError('Calibration requires valid inner and outer bands')
    gb = torch.autograd.grad(band.loss, z)[0]
    br = gb.square().mean().sqrt()
    result = {'dice_rms': float(dr), 'bands_rms': float(br),
              'bands_over_dice': float(br / dr)}
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError('Nonfinite or degenerate calibration gradients')
    return result
