"""Original losses adapted to one foreground class and explicit voxel units."""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.adni_baselines import core as adni
import compute_constraint_objectives as original
import nnunet_multi_objective as native
from scripts.capped_depth_delay import objectives as combined
from scripts.depth_supervision.geometry import build_geometry, GeometryConfig, signed_distance
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss, build_boundary_bands
from thesis.new_constraints.edge_consistency import signed_face_error

ARMS = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_depth', 'dice_bands_depth', 'dice_edge_depth',
        'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
        'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth')
GEOMETRY = dict(band_voxels=2., roi_margin_voxels=8., sparse_band_voxels=64)


def bands():
    return OuterBoundaryBandLoss(foreground_class_ids=(1,), complement_class_ids=(0,))


class BinaryEdge(torch.nn.Module):
    """Original uniform signed-face error, with a single hippocampus probability."""
    def forward(self, logits, labels):
        if logits.ndim != 5 or logits.shape[1] != 2 or labels.shape != logits[:, :1].shape:
            raise ValueError('Expected binary ADNI logits and matching labels')
        foreground = labels > 0
        with torch.no_grad():
            inner, outer = build_boundary_bands(foreground, steps=2)
        probability = logits.float().softmax(1)[:, 1:].sum(1, keepdim=True)
        total, count = signed_face_error(probability, foreground, inner | outer)
        valid = count > 0
        return ((total / count.clamp_min(1)) * valid).sum() / valid.sum().clamp_min(1)


def signed_distance_maps(labels):
    values = labels.detach().cpu().numpy()[:, 0]
    if labels.ndim != 5 or labels.shape[1] != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError('Expected binary ADNI masks')
    fields = np.stack([signed_distance(v == 1, (1., 1., 1.), 8.) / 8. for v in values])
    return torch.from_numpy(fields[:, None]).to(labels.device)


def sdf_loss(logits, labels, target_sdf):
    if logits.shape[1] != 2 or target_sdf is None or target_sdf.shape != logits[:, 1:].shape:
        raise ValueError('Expected one foreground distance field')
    return ((logits.float().softmax(1)[:, 1:] - (labels == 1).float()) * target_sdf.detach().float()).mean()


def depth_loss(logits, labels, weights):
    if logits.shape != (1, 2, 64, 64, 64) or labels.shape != (1, 1, 64, 64, 64):
        raise ValueError('Expected batch-one binary 64-cube recipe')
    if weights is None or weights.shape != logits.shape[2:]:
        raise ValueError('Depth weights missing')
    return (F.cross_entropy(logits.float(), labels[:, 0].long(), reduction='none')[0] * weights).sum()


def geometry_weights(labels, affine, settings):
    """Depth in source ARRAY coordinates, without claiming anatomical RAS or mm.

    Keep the original projected-axis plane rule: the normal is array axis 0
    projected orthogonally to foreground PC1. The affine maps augmented voxels
    back into the original unit voxel grid. Translation has no effect.
    The shared geometry helper's historical *_mm keys carry VOXEL units here.
    """
    array = labels.detach().cpu().numpy()[0, 0].astype(np.uint8)
    config = GeometryConfig(band_mm=settings['band_voxels'],
                            roi_margin_mm=settings['roi_margin_voxels'],
                            sparse_band_voxels=settings['sparse_band_voxels'])
    geometry = build_geometry(array, np.asarray(affine), config, fields=False)
    weights = geometry['weights']
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1., atol=1e-6):
        raise ValueError('Invalid normalized depth weights')
    return torch.as_tensor(weights, device=labels.device, dtype=torch.float32)


def bind():
    for module in (original, native, combined):
        module.OuterBoundaryBandLoss = bands
        module.BoundaryEdgeConsistencyLoss = BinaryEdge
        module.signed_distance_loss = sdf_loss
    combined.depth_loss = depth_loss


bind()
calibration_case = combined.calibration_case
select_weights = combined.select_weights


def training_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0., policy='delayed_5_of_9'):
    if 'sdf' in arm and policy == 'original_shared_ramp':
        adjusted = {**weights, 'sdf': weights['sdf'] * 1.0}
        return combined.full_loss(outputs, labels, arm, adjusted, scale, criterion, target_sdf, depth_weights)
    return combined.training_loss(outputs, labels, arm, weights, scale, target_sdf,
                                  criterion, depth_weights, sdf_scale)
