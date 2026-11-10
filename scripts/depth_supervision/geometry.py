"""Fixed annotation geometry in physical coordinates; no predicted geometry."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GeometryConfig:
    band_mm: float = 2.0
    roi_margin_mm: float = 8.0
    distance_clip_mm: float = 8.0
    layer_grid_mm: float = 1.0
    sparse_band_voxels: int = 64


def world_points(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return np.asarray(indices) @ affine[:3, :3].T + affine[:3, 3]


def longitudinal_plane(labels: np.ndarray, affine: np.ndarray,
                       right_axis=None, anterior_axis=None) -> dict:
    """Centroid plane containing PC1, with normal closest to anatomical right."""
    right = np.array([1., 0., 0.] if right_axis is None else right_axis)
    anterior = np.array([0., 1., 0.] if anterior_axis is None else anterior_axis)
    points = world_points(np.argwhere(labels > 0), affine)
    if len(points) < 4:
        raise ValueError("Cannot orient a hippocampus with fewer than four foreground voxels")
    center = points.mean(0)
    covariance = (points - center).T @ (points - center) / len(points)
    eigenvalues, axes = np.linalg.eigh(covariance)
    longitudinal = axes[:, -1]
    if longitudinal @ anterior < 0:
        longitudinal = -longitudinal
    normal = right - (right @ longitudinal) * longitudinal
    if np.linalg.norm(normal) < .1:
        raise ValueError("Longitudinal axis nearly parallel to anatomical left-right; inspect geometry")
    normal /= np.linalg.norm(normal)
    transverse = np.cross(normal, longitudinal)
    basis = np.column_stack((longitudinal, transverse, normal))
    return dict(center_mm=center, normal=normal, basis=basis,
                eigenvalues_mm2=eigenvalues,
                principal_axis_gap=float((eigenvalues[-1] - eigenvalues[-2]) / eigenvalues[-1]))


def signed_distance(mask: np.ndarray, spacing: tuple[float, ...], clip: float) -> np.ndarray:
    """Half-voxel EDT, negative inside; matches existing SDF at isotropic 1 mm."""
    if not mask.any():
        return np.full(mask.shape, clip, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -clip, dtype=np.float32)
    offset = .5 * min(spacing)
    field = np.where(mask, -(ndimage.distance_transform_edt(mask, sampling=spacing) - offset),
                     ndimage.distance_transform_edt(~mask, sampling=spacing) - offset)
    return np.clip(field, -clip, clip).astype(np.float32)


def build_geometry(labels: np.ndarray, affine: np.ndarray,
                   config: GeometryConfig = GeometryConfig(), *, fields: bool = True) -> dict:
    """Voxel weights plus matched 3-D and cross-sectional classwise distances.

    Cross-sections use a physical 1-mm oblique lattice. Absolute distances are
    interpolated back, then their signs are set by the ORIGINAL class labels.
    This preserves the correct gradient direction despite rasterization near
    oblique boundaries. These are discrete distance estimates, not exact SDFs.
    """
    labels = np.asarray(labels)
    affine = np.asarray(affine, dtype=np.float64)
    if labels.ndim != 3 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected a 3-D label array containing 0/1/2")
    if min(config.band_mm, config.roi_margin_mm, config.distance_clip_mm,
           config.layer_grid_mm, config.sparse_band_voxels) <= 0:
        raise ValueError("Geometry scales must be positive")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Sheared image affine is unsupported")
    plane = longitudinal_plane(labels, affine)
    indices = np.indices(labels.shape).reshape(3, -1).T
    local = (world_points(indices, affine) - plane['center_mm']) @ plane['basis']
    depth = local[:, 2].reshape(labels.shape)
    bin_ids = np.floor(depth / config.band_mm).astype(np.int16)
    foreground = labels > 0
    roi = ndimage.distance_transform_edt(~foreground, sampling=spacing) <= config.roi_margin_mm
    bins, counts = np.unique(bin_ids[roi], return_counts=True)
    masses = np.minimum(counts / config.sparse_band_voxels, 1.)
    weights = np.zeros(labels.shape, dtype=np.float32)
    for b, count, mass in zip(bins, counts, masses):
        weights[roi & (bin_ids == b)] = mass / (masses.sum() * count)
    result = dict(weights=weights, signed_depth_mm=depth.astype(np.float32), bin_ids=bin_ids,
                  center_mm=plane['center_mm'], normal=plane['normal'], basis=plane['basis'],
                  eigenvalues_mm2=plane['eigenvalues_mm2'], bin_values=bins, bin_counts=counts,
                  bin_masses=masses / masses.sum(), roi_voxels=np.array(roi.sum()),
                  principal_axis_gap=np.array(plane['principal_axis_gap']))
    if not fields:
        return result
    result['sdf_3d'] = np.stack([signed_distance(labels == c, tuple(spacing), config.distance_clip_mm)
                                 for c in (1, 2)]) / config.distance_clip_mm
    # Cover the ROI plus an EDT halo, independent of predictions. Each grid
    # location has original physical coordinates, unlike display-chart U/V.
    step = config.layer_grid_mm
    support = local[roi.ravel()]
    low = np.floor((support.min(0) - config.distance_clip_mm - 2 * step) / step) * step
    high = np.ceil((support.max(0) + config.distance_clip_mm + 2 * step) / step) * step
    grid_shape = (np.rint((high - low) / step).astype(int) + 1)
    if np.prod(grid_shape) > 8_000_000:
        raise ValueError("Unexpectedly large oblique grid")
    uv = np.stack(np.meshgrid(low[0] + np.arange(grid_shape[0]) * step,
                              low[1] + np.arange(grid_shape[1]) * step, indexing='ij'), -1)
    plane_labels = np.empty(tuple(grid_shape), dtype=np.uint8)
    inverse = np.linalg.inv(affine)
    for k in range(grid_shape[2]):
        coordinates = np.concatenate((uv, np.full((*uv.shape[:2], 1), low[2] + k * step)), -1)
        physical = coordinates @ plane['basis'].T + plane['center_mm']
        voxels = world_points(physical, inverse)
        plane_labels[:, :, k] = ndimage.map_coordinates(labels, voxels.reshape(-1, 3).T,
                                                        order=0, mode='constant', cval=0).reshape(uv.shape[:2])
    query = ((local - low) / step).T
    output, mismatches = [], []
    for c in (1, 2):
        lattice = np.empty(tuple(grid_shape), dtype=np.float32)
        for k in range(grid_shape[2]):
            lattice[:, :, k] = signed_distance(plane_labels[:, :, k] == c, (step, step),
                                                config.distance_clip_mm)
        # Measure sign disagreement separately; never turn a correct native
        # target into a different class because of the oblique rasterization.
        signed = ndimage.map_coordinates(lattice, query, order=1, mode='nearest').reshape(labels.shape)
        mismatch = ((signed < 0) != (labels == c)) & roi
        mismatches.append(int(mismatch.sum()))
        magnitude = ndimage.map_coordinates(np.abs(lattice), query, order=1,
                                             mode='nearest').reshape(labels.shape)
        output.append(np.where(labels == c, -magnitude, magnitude) / config.distance_clip_mm)
    result['sdf_layer'] = np.stack(output).astype(np.float32)
    result['layer_raster_sign_disagreement'] = np.array(mismatches)
    result['layer_grid_shape'] = grid_shape
    for name in ('weights', 'sdf_3d', 'sdf_layer'):
        if not np.isfinite(result[name]).all():
            raise FloatingPointError(name)
    if not np.isclose(weights.sum(), 1., atol=1e-6):
        raise AssertionError('Depth weights do not sum to one')
    return result
