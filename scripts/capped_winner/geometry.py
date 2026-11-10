"""Rebuild label-only targets after the unchanged capped augmentation."""
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.band_chamfer.geometry import edge_grid, surface_points
from scripts.band_focus.geometry import pair_geometry, move
from scripts.capped_depth_delay.objectives import geometry_weights, augmented_affine
from scripts.edge_ablation.objectives import valid_field
from .objectives import PAIR

_CACHE = OrderedDict()


def transformed_valid(shape, native_shape, original_affine, current_affine):
    original = valid_field(shape, native_shape)
    transform = np.linalg.solve(np.asarray(original_affine), np.asarray(current_affine))
    if np.allclose(transform, np.eye(4)):
        return original
    xyz = np.indices(shape).reshape(3, -1)
    mapped = np.rint(transform[:3, :3] @ xyz + transform[:3, 3:]).astype(int)
    bounded = ((mapped >= 0) & (mapped < np.asarray(shape)[:, None])).all(0)
    clipped = np.clip(mapped, 0, np.asarray(shape)[:, None]-1)
    return (bounded & original[tuple(clipped)]).reshape(shape)


def make_targets(labels, original_affine, current_affine, native_shape, settings, device):
    y = labels.detach().cpu().numpy()[0, 0].astype(np.uint8)
    valid = transformed_valid(y.shape, native_shape, original_affine, current_affine)
    if np.any((y > 0) & ~valid):
        raise ValueError('Transformed foreground falls outside the transformed native FOV')
    key = hashlib.sha256(y.tobytes()+valid.tobytes()+np.asarray(current_affine).tobytes()).hexdigest()
    if key not in _CACHE:
        spacing = np.ones(3)
        grid = edge_grid(valid, spacing)
        gt = surface_points(torch.from_numpy(y > 0).float(), grid)
        xyz = np.indices(y.shape).reshape(3, -1).T
        distance = (cKDTree(gt.numpy()).query(xyz, workers=1)[0].reshape(y.shape)
                    if len(gt) else np.full(y.shape, np.inf))
        _CACHE[key] = dict(valid=torch.from_numpy(valid), foreground=torch.from_numpy(y > 0),
            distance=torch.tensor(distance, dtype=torch.float32),
            pairs=pair_geometry(y, valid, spacing, PAIR),
            depth=geometry_weights(labels.cpu(), current_affine, settings).cpu())
        if len(_CACHE) > 24:
            _CACHE.popitem(last=False)
    return move(_CACHE[key], device)


def worst_case_targets(shape=(64, 64, 64)):
    """Arithmetic bound: N voxels/side and six directed anchors/voxel/scale."""
    n = int(np.prod(shape))
    index = torch.arange(n).repeat(6)
    group = {side: dict(index=index, weight=torch.full((6*n,), 1/(6*n)))
             for side in ('inside', 'outside')}
    return dict(valid=torch.ones(shape, dtype=torch.bool),
        foreground=torch.ones(shape, dtype=torch.bool), distance=torch.ones(shape),
        pairs=[group, group], depth=torch.full(shape, 1/n))
