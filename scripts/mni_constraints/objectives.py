"""Original constraints adapted to MNI, preserving each winning base objective."""
from functools import lru_cache
import hashlib
import json
import math
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from scripts.unetr_subfields import geo_recipe, log_squared_dice
from scripts.capped_depth_delay import objectives as historical
from scripts.depth_supervision.geometry import build_geometry, GeometryConfig
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss, build_boundary_bands
from thesis.new_constraints.edge_consistency import signed_face_error
from compute_constraint_objectives import calibrated_weight

TERMS = ('bands', 'edge', 'sdf', 'depth')


def base_loss(outputs, labels, identity, criterion=None):
    if identity == 'unetr_vit_b16':
        return geo_recipe.training_loss(outputs, labels, 'dice', {}, .2)
    if identity == 'swinunetr':
        return log_squared_dice.training_loss(outputs, labels, 'dice', {}, .2)
    return historical.training_loss(outputs, labels, 'dice', {}, .2, criterion=criterion)


def edge_loss(logits, labels):
    foreground = labels > 0
    with torch.no_grad():
        inner, outer = build_boundary_bands(foreground, steps=2)
    probability = logits.float().softmax(1)[:, 1:].sum(1, keepdim=True)
    total, count = signed_face_error(probability, foreground, inner | outer)
    valid = count > 0
    return ((total / count.clamp_min(1)) * valid).sum() / valid.sum().clamp_min(1)


def signed_distance_maps(labels, clip=8.):
    values = labels.detach().cpu().numpy()[:, 0]
    if labels.ndim != 5 or labels.shape[1] != 1 or not np.isin(values, [0, 1, 2, 3]).all():
        raise ValueError('Expected single-channel MNI labels 0..3')
    maps = np.empty((len(values), 3, *values.shape[1:]), dtype=np.float32)
    for batch, volume in enumerate(values):
        for channel, class_id in enumerate((1, 2, 3)):
            mask = volume == class_id
            if not mask.any():
                field = np.full(mask.shape, clip)
            elif mask.all():
                field = np.full(mask.shape, -clip)
            else:
                field = np.where(mask, -(ndimage.distance_transform_edt(mask) - .5),
                                 ndimage.distance_transform_edt(~mask) - .5)
            maps[batch, channel] = np.clip(field, -clip, clip) / clip
    return torch.from_numpy(maps).to(labels.device)


@lru_cache(maxsize=64)
def _geometry(shape, binary_bytes, affine_json, settings_json):
    array = np.frombuffer(binary_bytes, dtype=np.uint8).reshape(shape)
    result = build_geometry(array, np.asarray(json.loads(affine_json)),
                            GeometryConfig(**json.loads(settings_json)), fields=False)
    return torch.tensor(result['weights'], dtype=torch.float32)


def geometry_weights(labels, affine, settings):
    # Geometry depends on the foreground union, not the number of subfields.
    array = (labels.detach().cpu().numpy()[0, 0] > 0).astype(np.uint8)
    return _geometry(array.shape, array.tobytes(), json.dumps(np.asarray(affine).tolist()),
                     json.dumps(settings, sort_keys=True)).to(labels.device)


def transform_depth(weights, seed, update):
    # Transform native annotation weights with exactly the same voxel permutation.
    # Re-estimating PCA in rotated array coordinates would change the objective.
    index = geo_recipe.transform_id(seed, update)
    result = torch.rot90(weights, index % 4, dims=(0, 1))
    if index >= 4:
        result = torch.flip(result, dims=(0,))
    return result.contiguous()


def term(name, logits, labels, target_sdf=None, depth_weights=None):
    if logits.shape[1] != 4 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected four-class MNI logits and aligned labels')
    if name == 'bands':
        return OuterBoundaryBandLoss(foreground_class_ids=(1, 2, 3), complement_class_ids=(0,))(logits, labels).loss
    if name == 'edge':
        return edge_loss(logits, labels)
    if name == 'sdf':
        if target_sdf is None or target_sdf.shape != logits[:, 1:].shape:
            raise ValueError('Three foreground SDF maps required')
        truth = torch.cat([labels == c for c in (1, 2, 3)], dim=1).float()
        return ((logits.float().softmax(1)[:, 1:] - truth) * target_sdf.detach().float()).mean()
    if name == 'depth':
        if depth_weights is None or depth_weights.shape != logits.shape[2:]:
            raise ValueError('Aligned native-geometry weights required')
        return (F.cross_entropy(logits.float(), labels[:, 0].long(), reduction='none')[0] * depth_weights).sum()
    raise ValueError(name)


def training_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0., *, identity):
    loss = base_loss(outputs, labels, identity, criterion)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    for name in TERMS:
        if name in arm:
            factor = sdf_scale if name == 'sdf' else scale
            loss = loss + (weights[name] * factor) * term(name, logits, labels, target_sdf, depth_weights)
    return loss


def calibration_case(outputs, labels, arm, target_sdf=None, criterion=None,
                     depth_weights=None, *, identity):
    detached = [o.detach().float() for o in outputs] if isinstance(outputs, (tuple, list)) else outputs.detach().float()
    z = detached[0] if isinstance(detached, list) else detached
    z.requires_grad_(True)
    gradients = {'base': torch.autograd.grad(base_loss(detached, labels, identity, criterion), z)[0]}
    for name in TERMS:
        if name in arm:
            gradients[name] = torch.autograd.grad(term(name, z, labels, target_sdf, depth_weights), z)[0]
    gram = {a + ':' + b: float((ga * gb).mean()) for a, ga in gradients.items() for b, gb in gradients.items()}
    if not all(math.isfinite(v) for v in gram.values()) or any(gram[n+':'+n] <= 0 for n in gradients):
        raise ValueError('Nonfinite or degenerate calibration gradients')
    return dict(base_loss_identity=identity, gradient_pair_means=gram,
                gradient_rms={n: math.sqrt(gram[n+':'+n]) for n in gradients})


def select_weights(rows, arm, target=.1, cap=.5):
    coefficients = {'base': 1.}
    statistics = {}
    for name in TERMS:
        if name not in arm:
            continue
        ratios = []
        for row in rows:
            gram = row['gradient_pair_means']
            mse = sum(wa * wb * gram[a+':'+b] for a, wa in coefficients.items() for b, wb in coefficients.items())
            if not math.isfinite(mse) or mse <= 0:
                raise ValueError('Degenerate joint gradient')
            ratio = row['gradient_rms'][name] / math.sqrt(mse)
            row[name+'_over_base'] = ratio
            ratios.append(ratio)
        coefficients[name], statistics[name] = calibrated_weight(ratios, target, cap)
    return {k: v for k, v in coefficients.items() if k != 'base'}, statistics
