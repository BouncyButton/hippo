"""Explicit physical geometry for future conventional HFH SDF/depth arms.

BAND remains a separately declared morphological voxel-step objective.
This adapter is not installed by the active BAND-only worker.
"""
import numpy as np
import torch
from functools import lru_cache
from scripts.depth_supervision.geometry import GeometryConfig, build_geometry, signed_distance

SPACING_MM = (1., 2., 1.)
DEPTH = dict(band_mm=2., roi_margin_mm=8., sparse_band_voxels=64)


@lru_cache(maxsize=64)
def _sdf_fields(shape, annotation_bytes):
    values=np.frombuffer(annotation_bytes,dtype=np.uint8).reshape(shape)
    return np.stack([signed_distance(v==1,SPACING_MM,8.)/8. for v in values])


@lru_cache(maxsize=64)
def _depth_field(annotation_bytes, affine_values):
    values=np.frombuffer(annotation_bytes,dtype=np.uint8).reshape(64,64,64)
    affine=np.asarray(affine_values,dtype=np.float64).reshape(4,4)
    return build_geometry(values,affine,GeometryConfig(**DEPTH),fields=False)['weights']


def sdf_targets(labels):
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError('Expected B,1,X,Y,Z labels')
    values = labels.detach().cpu().numpy()[:, 0]
    if not np.isin(values, [0, 1]).all():
        raise ValueError('Expected binary HFH annotations')
    # Preserve the project's discrete half-minimum-spacing EDT convention;
    # this is an anisotropic distance estimate, not an exact surface distance.
    fields = _sdf_fields(values.shape,values.astype(np.uint8).tobytes()).copy()
    return torch.as_tensor(fields[:, None], device=labels.device, dtype=torch.float32)


def depth_weights(labels, affine, settings):
    if labels.shape != (1, 1, 64, 64, 64):
        raise ValueError('Expected padded batch-one HFH grid')
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.allclose(affine[:3, :3], np.diag(SPACING_MM)):
        raise ValueError('Expected HFH source-axis physical affine, 1x2x1 mm')
    if settings != DEPTH:
        raise ValueError('Depth geometry differs from the declared physical protocol')
    values = labels.detach().cpu().numpy()[0, 0]
    if not np.isin(values, [0, 1]).all() or not values.any():
        raise ValueError('Expected a nonempty binary hippocampus')
    # Cache by annotation content and complete affine, not tensor addresses.
    # A fresh copy prevents an in-place consumer from mutating cached targets.
    weights = _depth_field(values.astype(np.uint8).tobytes(),tuple(affine.ravel())).copy()
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-6):
        raise ValueError('Invalid normalized HFH depth weights')
    return torch.as_tensor(weights, device=labels.device, dtype=torch.float32)


def install(trainer, protocol, arm):
    if not any(component in arm for component in ('sdf', 'depth')):
        return
    if protocol.get('constraint_geometry_units') != 'physical_mm_hfh_1x2x1':
        raise ValueError('HFH physical constraint units must be frozen explicitly')
    trainer.signed_distance_maps = sdf_targets
    trainer.geometry_weights = depth_weights
