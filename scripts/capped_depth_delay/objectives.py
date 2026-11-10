"""Original objectives plus the annotation-based depth-weighted CE term."""
from itertools import combinations
import math
import numpy as np
import torch
import torch.nn.functional as F

from compute_constraint_objectives import calibration_case as old_calibration
from compute_constraint_objectives import select_weights as old_select, calibrated_weight
from compute_constraint_objectives import training_loss as old_loss
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_loss
from scripts.depth_supervision.geometry import build_geometry, GeometryConfig

SDF_ARMS = ('dice_sdf', 'dice_bands_sdf', 'dice_bands_edge_sdf')
DEPTH_ARMS = tuple('dice' + ''.join('_' + name for name in ('bands', 'edge', 'sdf') if name in names) + '_depth'
                   for count in range(4) for names in combinations(('bands', 'edge', 'sdf'), count))
ARMS = SDF_ARMS + DEPTH_ARMS


def base_arm(arm):
    return arm.removesuffix('_depth')


def geometry_weights(labels, affine, settings):
    array = labels.detach().cpu().numpy()[0, 0].astype(np.uint8)
    geometry = build_geometry(array, np.asarray(affine), GeometryConfig(**settings), fields=False)
    return torch.as_tensor(geometry['weights'], device=labels.device, dtype=torch.float32)


def augmented_affine(affine, rng_state, statistics, shape):
    """Map augmented voxels back to the same physical anatomy and units.

    Replay only the original augmentation's first 16 CPU random draws using a
    separate generator. No model/data RNG state or input tensor is changed.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if not statistics['spatial_applied']:
        return affine
    generator = torch.Generator()
    generator.set_state(rng_state)
    u = torch.rand(16, generator=generator).tolist()
    a, b, c = [math.radians(20 * v - 10) for v in u[2:5]]
    sa, ca, sb, cb, sc, cc = math.sin(a), math.cos(a), math.sin(b), math.cos(b), math.sin(c), math.cos(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    inverse = np.linalg.inv((0.9 + 0.2 * u[5]) * (rz @ ry @ rx))
    reverse = np.eye(3)[::-1]
    matrix = reverse @ inverse @ reverse
    center = (np.asarray(shape, dtype=float) - 1) / 2
    shift_xyz = np.array([4 * v - 2 for v in u[6:9]])
    transform = np.eye(4)
    transform[:3, :3] = matrix
    transform[:3, 3] = center - matrix @ center - reverse @ inverse @ shift_xyz
    return affine @ transform


def depth_loss(logits, labels, depth_weights):
    if logits.shape != (1, 3, 64, 64, 64) or labels.shape != (1, 1, 64, 64, 64):
        raise ValueError('Expected the frozen three-class, batch-one 64-cube recipe')
    if depth_weights is None or depth_weights.shape != logits.shape[2:]:
        raise ValueError('Depth weights missing or shape differs')
    return (F.cross_entropy(logits.float(), labels[:, 0].long(), reduction='none')[0] * depth_weights).sum()


def full_loss(outputs, labels, arm, weights, scale, criterion=None, target_sdf=None, depth_weights=None):
    original = base_arm(arm)
    if criterion is None:
        loss = old_loss(outputs, labels, original, weights, scale, target_sdf)
        logits = outputs
    else:
        from nnunet_multi_objective import training_loss
        loss = training_loss(outputs, labels, original, weights, scale, criterion, target_sdf)
        logits = outputs[0]
    if 'depth' in arm:
        loss = loss + (weights['depth'] * scale) * depth_loss(logits, labels, depth_weights)
    return loss


def training_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0.):
    if 'sdf' in arm:
        if scale != 1. or not 0 < sdf_scale <= 1:
            raise ValueError('SDF must activate after the existing early ramp')
        weights = {**weights, 'sdf': weights['sdf'] * sdf_scale}
    return full_loss(outputs, labels, arm, weights, scale, criterion, target_sdf, depth_weights)


def calibration_case(outputs, labels, arm, target_sdf=None, criterion=None, depth_weights=None):
    original = base_arm(arm)
    if criterion is None:
        row = old_calibration(outputs, labels, original, target_sdf)
    else:
        from nnunet_multi_objective import calibration_case as native_calibration
        row = native_calibration(outputs, labels, original, criterion, target_sdf)
    if 'depth' not in arm:
        return row
    if criterion is None:
        from monai.losses import DiceLoss
        z = outputs.detach().float().requires_grad_(True)
        region = DiceLoss(to_onehot_y=True, softmax=True)(z, labels)
    else:
        from nnunet_band_objective import native_loss
        detached = [value.detach().float() for value in outputs]
        z = detached[0].requires_grad_(True)
        region = native_loss(detached, labels, criterion)
    gradients = {'dice': torch.autograd.grad(region, z)[0]}
    terms = {'bands': lambda: OuterBoundaryBandLoss()(z, labels).loss,
             'edge': lambda: BoundaryEdgeConsistencyLoss()(z, labels),
             'sdf': lambda: signed_distance_loss(z, labels, target_sdf),
             'depth': lambda: depth_loss(z, labels, depth_weights)}
    for name, term in terms.items():
        if name in arm:
            gradients[name] = torch.autograd.grad(term(), z)[0]
    row['depth_rms'] = float(gradients['depth'].square().mean().sqrt())
    row['_depth_pair_means'] = {left + ':' + right: float((gleft * gright).mean())
                               for left, gleft in gradients.items()
                               for right, gright in gradients.items()}
    return row


def select_weights(rows, arm):
    weights, statistics = old_select(rows, base_arm(arm))
    if 'depth' in arm:
        coefficients = {'dice': 1., **weights}
        for row in rows:
            mse = sum(wleft * wright * row['_depth_pair_means'][left + ':' + right]
                      for left, wleft in coefficients.items() for right, wright in coefficients.items())
            if not math.isfinite(mse) or mse <= 0:
                raise ValueError('Degenerate joint gradient for depth calibration')
            row['depth_over_base'] = row['depth_rms'] / math.sqrt(mse)
        weights['depth'], statistics['depth'] = calibrated_weight([r['depth_over_base'] for r in rows])
    return weights, statistics
