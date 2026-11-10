"""Analytically invertible, prediction-guided straightening in physical space."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline

from scripts.depth_supervision.geometry import longitudinal_plane, world_points


@dataclass
class Straightening:
    center: np.ndarray
    basis: np.ndarray
    knots: np.ndarray
    offsets: np.ndarray
    raw_s: np.ndarray
    raw_t: np.ndarray
    component_fraction: float

    def __post_init__(self):
        self.curve = CubicSpline(self.knots, self.offsets, bc_type=((1, 0.), (1, 0.)))

    def offset(self, s, derivative=0):
        s = np.asarray(s)
        values = self.curve(np.clip(s, self.knots[0], self.knots[-1]), derivative)
        if derivative:
            values = np.where((s < self.knots[0]) | (s > self.knots[-1]), 0., values)
        return values

    def world_to_local(self, xyz):
        return (np.asarray(xyz) - self.center) @ self.basis

    def world_to_straight(self, xyz):
        q = self.world_to_local(xyz)
        q = q.copy()
        q[..., 1] -= self.offset(q[..., 0])
        return q

    def straight_to_world(self, q):
        local = np.asarray(q).copy()
        local[..., 1] += self.offset(local[..., 0])
        return local @ self.basis.T + self.center

    def local_to_world(self, q):
        return np.asarray(q) @ self.basis.T + self.center

    def metadata(self):
        return dict(center_mm=self.center.tolist(), basis_columns_s_t_d=self.basis.tolist(),
                    knot_s_mm=self.knots.tolist(), offset_t_mm=self.offsets.tolist(),
                    raw_section_s_mm=self.raw_s.tolist(), raw_section_t_mm=self.raw_t.tolist(),
                    largest_prediction_component_fraction=self.component_fraction,
                    mapping='local=(s, t + c(s), d); world=center + basis @ local',
                    inverse='s=local_s; t=local_t-c(local_s); d=local_d',
                    depth='signed physical distance from prediction-derived longitudinal plane',
                    jacobian_determinant_analytic=1.,
                    interpretation='PCA-longitudinal straightening by a transverse shear; not arc-length or laminar unfolding')


def fit_straightening(prediction, affine, *, section_mm=2., smoothing_mm=3.):
    """Fit from predicted foreground only; annotation and intensities are not inputs."""
    mask = np.asarray(prediction) > 0
    labels, number = ndi.label(mask, ndi.generate_binary_structure(3, 3))
    if not number:
        raise ValueError('Empty foreground prediction: no valid anatomical guide')
    counts = np.bincount(labels.ravel())
    largest = int(np.argmax(counts[1:]) + 1)
    guide = labels == largest
    plane = longitudinal_plane(guide, affine)
    xyz = world_points(np.argwhere(guide), affine)
    local = (xyz - plane['center_mm']) @ plane['basis']
    s, t = local[:, 0], local[:, 1]
    bins = np.floor((s-s.min())/section_mm).astype(int)
    raw_s, raw_t, raw_n = [], [], []
    for b in np.unique(bins):
        keep = bins == b
        if keep.sum() >= 4:
            raw_s.append(np.median(s[keep])); raw_t.append(np.median(t[keep])); raw_n.append(keep.sum())
    if len(raw_s) < 4:
        raise ValueError('Insufficient longitudinal support for a smooth guide')
    raw_s, raw_t, raw_n = map(np.asarray, (raw_s, raw_t, raw_n))
    knots = np.linspace(s.min(), s.max(), max(8, int(np.ceil(np.ptp(s))) + 1))
    kernel = np.exp(-.5*((knots[:, None]-raw_s[None])/smoothing_mm)**2) * np.sqrt(raw_n[None])
    offsets = (kernel @ raw_t) / kernel.sum(1)
    return Straightening(plane['center_mm'], plane['basis'], knots, offsets, raw_s, raw_t,
                         float(guide.sum()/mask.sum()))


def matched_axes(transform, roi_world, size=64, halo_mm=2.):
    """Matched control/straight grid extents, covering the predicted foreground ROI."""
    both = np.concatenate([transform.world_to_local(roi_world), transform.world_to_straight(roi_world)])
    low, high = both.min(0)-halo_mm, both.max(0)+halo_mm
    center = (low+high)/2
    center[2] = 0.  # retain an explicitly signed plane-depth axis
    step = np.maximum(1., 2*np.maximum(high-center, center-low)/(size-1))
    return [center[i] + (np.arange(size)-(size-1)/2)*step[i] for i in range(3)]


def sample_world(array, affine, xyz, *, order=1):
    inverse = np.linalg.inv(affine)
    voxels = world_points(np.asarray(xyz).reshape(-1, 3), inverse)
    values = ndi.map_coordinates(array, voxels.T, order=order, mode='constant', cval=0., prefilter=False)
    return values.reshape(np.asarray(xyz).shape[:-1])


def sample_grid(array, axes, q, *, order=1):
    q = np.asarray(q)
    indices = np.stack([(q[..., i]-axes[i][0])/(axes[i][1]-axes[i][0]) for i in range(3)], -1)
    values = ndi.map_coordinates(array, indices.reshape(-1, 3).T, order=order, mode='constant', cval=0., prefilter=False)
    return values.reshape(q.shape[:-1])


def inside_grid(axes, q, margin_voxels=0.):
    return np.logical_and.reduce([(q[..., i] >= a[0]+margin_voxels*(a[1]-a[0])) &
                                  (q[..., i] <= a[-1]-margin_voxels*(a[1]-a[0])) for i, a in enumerate(axes)])
