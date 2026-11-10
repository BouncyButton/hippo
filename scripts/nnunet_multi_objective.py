"""Full-resolution constraints alongside the unchanged native nnU-Net objective."""
import math
import torch

from nnunet_band_objective import native_loss
from compute_constraint_objectives import ARMS
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.constraint_result import differentiable_zero
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_loss


def training_loss(outputs, labels, arm, weights, scale, criterion, target_sdf=None):
    if arm not in ARMS:
        raise ValueError('Unknown constraint objective')
    loss = native_loss(outputs, labels, criterion)
    if arm == 'dice':
        return loss
    logits = outputs[0]
    auxiliary = differentiable_zero(logits.float())
    if 'bands' in arm:
        auxiliary = auxiliary + weights['bands'] * OuterBoundaryBandLoss()(logits, labels).loss
    if 'edge' in arm:
        auxiliary = auxiliary + weights['edge'] * BoundaryEdgeConsistencyLoss()(logits, labels)
    if 'sdf' in arm:
        auxiliary = auxiliary + weights['sdf'] * signed_distance_loss(logits, labels, target_sdf)
    return loss + scale * auxiliary


def calibration_case(outputs, labels, arm, criterion, target_sdf=None):
    """Calibrate against the native gradient, including its deep-supervision weight."""
    detached = [output.detach().float() for output in outputs]
    z = detached[0].requires_grad_(True)
    gd = torch.autograd.grad(native_loss(detached, labels, criterion), z)[0]
    dr = gd.square().mean().sqrt()
    result = {'dice_rms': float(dr)}
    if 'bands' in arm:
        band = OuterBoundaryBandLoss()(z, labels)
        if not bool(band.details['valid'].all()):
            raise ValueError('Calibration requires valid inner and outer bands')
        gb = torch.autograd.grad(band.loss, z)[0]
        br = gb.square().mean().sqrt()
        result.update(bands_rms=float(br), bands_over_dice=float(br / dr))
        if 'edge' in arm or 'sdf' in arm:
            result['dice_bands_cross_mean'] = float((gd * gb).mean())
    if 'edge' in arm:
        ge = torch.autograd.grad(BoundaryEdgeConsistencyLoss()(z, labels), z)[0]
        er = ge.square().mean().sqrt()
        result['edge_rms'] = float(er)
        if 'bands' not in arm:
            result['edge_over_base'] = float(er / dr)
    if 'sdf' in arm:
        gs = torch.autograd.grad(signed_distance_loss(z, labels, target_sdf), z)[0]
        result['sdf_rms'] = float(gs.square().mean().sqrt())
        if 'edge' in arm:
            result['dice_edge_cross_mean'] = float((gd * ge).mean())
            if 'bands' in arm:
                result['bands_edge_cross_mean'] = float((gb * ge).mean())
    if not all(math.isfinite(v) for v in result.values()) or any(
        v <= 0 for k, v in result.items() if not k.endswith('cross_mean')
    ):
        raise ValueError('Nonfinite or degenerate calibration gradients')
    return result
