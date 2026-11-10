"""Longitudinal tissue profiles retaining transverse strips at two resolutions."""
from pathlib import Path
import numpy as np
import torch
from .profiles import make_geometry, profile_distance


DEFAULT_SCALES = [dict(band_mm=2., longitudinal_mm=4., transverse_mm=4.),
                  dict(band_mm=2., longitudinal_mm=2., transverse_mm=2.)]


def make_strip_geometry(shape, native_shape, affine, center, basis,
                        band_mm=2., longitudinal_mm=2., transverse_mm=2.):
    """Keep one index per source voxel; bin in the shared physical frame."""
    if not np.isfinite(transverse_mm) or transverse_mm <= 0:
        raise ValueError('Positive transverse strip width required')
    # Reuse the original frame/FOV validation and physical volume convention.
    base = make_geometry(shape, native_shape, affine, center, basis, band_mm, longitudinal_mm)
    indices = np.array(np.unravel_index(base['indices'], tuple(shape))).T
    affine, basis = np.asarray(affine, float), np.asarray(basis, float)
    local = (indices @ affine[:3, :3].T + affine[:3, 3]-center) @ basis
    longitudinal = np.floor(local[:, 0]/longitudinal_mm).astype(np.int64)
    transverse = np.floor(local[:, 1]/transverse_mm).astype(np.int64)
    depth = np.floor(local[:, 2]/band_mm).astype(np.int64)
    nl = int(np.ptp(longitudinal)+1)
    nt = int(np.ptp(transverse)+1)
    nd = int(np.ptp(depth)+1)
    groups = ((depth-depth.min())*nt+transverse-transverse.min())*nl+longitudinal-longitudinal.min()
    order = np.argsort(groups, kind='stable')
    return dict(indices=base['indices'][order].copy(), lengths=np.bincount(groups, minlength=nd*nt*nl),
                n_long=nl, n_depth=nd, n_transverse=nt, voxel_mm3=base['voxel_mm3'],
                valid_voxels=base['valid_voxels'], longitudinal_mm=longitudinal_mm,
                transverse_mm=transverse_mm, band_mm=band_mm)


def strip_mass(foreground, geometry):
    """Return [class, signed depth, transverse strip, longitudinal] mm³."""
    if foreground.shape[0] != 2:
        raise ValueError('Separate anterior/posterior probabilities required')
    g = geometry
    indices = torch.as_tensor(g['indices'], device=foreground.device, dtype=torch.long)
    lengths = torch.as_tensor(g['lengths'], device=foreground.device, dtype=torch.long)
    values = foreground.float().reshape(2, -1)[:, indices].contiguous()
    mass = torch.segment_reduce(values, 'sum', lengths=lengths[None].expand(2, -1), axis=1)
    return mass.reshape(2, g['n_depth'], g['n_transverse'], g['n_long'])*g['voxel_mm3']


def multiscale_distance(probabilities, truth, geometries):
    """Equal mean of scale losses; never normalize individual strips/bands."""
    if not geometries:
        raise ValueError('At least one anatomical resolution required')
    losses = []
    for geometry in geometries:
        predicted, target = [strip_mass(x, geometry) for x in (probabilities, truth)]
        # Flatten only the row index, retaining every independent depth/strip.
        # Every scale uses the same total physical GT mass, not cell counts.
        losses.append(profile_distance(predicted.flatten(1, 2), target.flatten(1, 2)))
    return torch.stack(losses).mean()


def multiscale_loss(logits, labels, geometries):
    if logits.shape[0] != 1 or logits.shape[1] != 3 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected batch-one three-class logits and labels')
    truth = torch.stack((labels[0, 0] == 1, labels[0, 0] == 2)).float()
    return multiscale_distance(logits.float().softmax(1)[0, 1:], truth, geometries)


class MultiscaleCache:
    def __init__(self, root, protocol, fraction):
        self.root, self.protocol, self.fraction = Path(root), protocol, fraction
        self.items = {}

    def get(self, item):
        name = item['case_name']
        if name not in self.items:
            with np.load(self.root/'atlas'/self.fraction/(name+'.npz'), allow_pickle=False) as atlas:
                center, basis = atlas['center'], atlas['basis']
            self.items[name] = [make_strip_geometry(item['image'].shape[1:],
                self.protocol['native_geometry'][name]['native_shape'], item['affine'], center, basis, **scale)
                for scale in self.protocol['multiscale_geometry']]
        return self.items[name]
