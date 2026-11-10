"""Mass-preserving class profiles in a frozen anatomical coordinate frame."""
from pathlib import Path
import numpy as np
import torch


def make_geometry(shape, native_shape, affine, center, basis, band_mm=2., longitudinal_mm=2.):
    shape, native = np.asarray(shape, int), np.asarray(native_shape, int)
    affine, center, basis = [np.asarray(v, float) for v in (affine, center, basis)]
    if np.any(native <= 0) or np.any(native > shape):
        raise ValueError('Native field of view must fit the padded grid')
    if not np.isfinite(affine).all() or not np.isfinite(center).all():
        raise ValueError('Nonfinite physical frame')
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
        raise ValueError('Expected orthonormal longitudinal/transverse/normal basis')
    if min(band_mm, longitudinal_mm) <= 0:
        raise ValueError('Positive physical bin widths required')
    before = (shape-native)//2
    indices = np.indices(tuple(native)).reshape(3, -1).T + before
    xyz = indices @ affine[:3, :3].T + affine[:3, 3]
    local = (xyz-center) @ basis
    longitudinal = np.floor(local[:, 0]/longitudinal_mm).astype(np.int64)
    depth = np.floor(local[:, 2]/band_mm).astype(np.int64)
    lmin, dmin = longitudinal.min(), depth.min()
    nl, nd = int(longitudinal.max()-lmin+1), int(depth.max()-dmin+1)
    groups = (depth-dmin)*nl + longitudinal-lmin
    order = np.argsort(groups, kind='stable')
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    if voxel_mm3 <= 0:
        raise ValueError('Singular affine')
    return dict(indices=np.ravel_multi_index(indices.T, tuple(shape))[order].copy(),
                lengths=np.bincount(groups, minlength=nd*nl), n_long=nl, n_depth=nd,
                voxel_mm3=voxel_mm3, longitudinal_mm=longitudinal_mm,
                longitudinal_start_mm=float(lmin*longitudinal_mm),
                signed_depth_start_mm=float(dmin*band_mm), valid_voxels=int(np.prod(native)))


def tissue_mass(foreground, geometry):
    """Return [class, signed-depth bin, longitudinal bin] tissue volumes in mm³."""
    if foreground.shape[0] != 2:
        raise ValueError('Separate anterior/posterior probabilities required')
    g = geometry
    indices = torch.as_tensor(g['indices'], device=foreground.device, dtype=torch.long)
    lengths = torch.as_tensor(g['lengths'], device=foreground.device, dtype=torch.long)
    values = foreground.float().reshape(2, -1)[:, indices].contiguous()
    # Sorted segments avoid the unordered CUDA atomic sums of scatter_add.
    mass = torch.segment_reduce(values, 'sum', lengths=lengths[None].expand(2, -1), axis=1)
    return mass.reshape(2, g['n_depth'], g['n_long']) * g['voxel_mm3']


def profile_distance(predicted_mass, target_mass):
    """Symmetric cumulative discrepancy; one global GT mass scale, no band weights.

    Both directions are used so missing tissue has no arbitrary dependence on
    which end of the longitudinal axis is the origin. Profiles retain absolute
    masses; prediction and target are never independently normalized.
    """
    if predicted_mass.shape != target_mass.shape or predicted_mass.ndim != 3:
        raise ValueError('Expected matched [class, depth, longitudinal] profiles')
    delta = predicted_mass-target_mass
    forward = delta.cumsum(-1)
    reverse = delta.flip(-1).cumsum(-1).flip(-1)
    denominator = target_mass.sum().clamp_min(1.) * delta.shape[-1]
    return .5*(forward.abs().sum()+reverse.abs().sum())/denominator


def profile_loss(logits, labels, geometry):
    if logits.shape[0] != 1 or logits.shape[1] != 3 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected batch-one three-class logits and labels')
    probabilities = logits.float().softmax(1)[0, 1:]
    truth = torch.stack((labels[0, 0] == 1, labels[0, 0] == 2)).float()
    return profile_distance(tissue_mass(probabilities, geometry), tissue_mass(truth, geometry))


class ProfileCache:
    def __init__(self, root, protocol, fraction):
        self.root, self.protocol, self.fraction = Path(root), protocol, fraction
        self.items = {}

    def get(self, item):
        name = item['case_name']
        if name not in self.items:
            with np.load(self.root/'atlas'/self.fraction/(name+'.npz'), allow_pickle=False) as atlas:
                center, basis = atlas['center'], atlas['basis']
            self.items[name] = make_geometry(item['image'].shape[1:],
                self.protocol['native_geometry'][name]['native_shape'], item['affine'], center, basis,
                **self.protocol['profile_geometry'])
        return self.items[name]
