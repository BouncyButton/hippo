"""Frozen GT pair geometry and explicit probability/score edge objectives."""
from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss, build_boundary_bands
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss


def experiment_matrix():
    controls = [dict(id=name, kind=name) for name in
                ('bands', 'legacy_edge', 'same_only', 'split_equal')]
    variants = []
    for form, distances, reduction, ap in itertools.product(
            ('probability', 'score'), ((1.,), (1., 3.)), ('pairs', 'anchors'), (False, True)):
        name = f"{form}_{'1mm' if len(distances) == 1 else '1and3mm'}_{reduction}_{'outer_ap' if ap else 'outer'}"
        variants.append(dict(id=name, kind='cross', form=form, distances_mm=list(distances),
                             reduction=reduction, ap=ap))
    return controls + variants


def valid_field(shape, native_shape):
    shape, native = np.asarray(shape, int), np.asarray(native_shape, int)
    if (native <= 0).any() or (native > shape).any():
        raise ValueError('Invalid native field of view')
    before = (shape-native)//2
    valid = np.zeros(tuple(shape), bool)
    valid[tuple(slice(int(a), int(a+b)) for a, b in zip(before, native))] = True
    return valid


def pair_weights(anchors, scales, reduction):
    if not len(anchors):
        return np.empty(0, np.float32)
    if reduction == 'pairs':
        levels, inverse, counts = np.unique(scales, return_inverse=True, return_counts=True)
        weights = 1. / (len(levels)*counts[inverse])
    elif reduction == 'anchors':
        keys, inverse, counts = np.unique(np.column_stack([anchors, scales]), axis=0,
                                          return_inverse=True, return_counts=True)
        locations, location_index = np.unique(keys[:, 0], return_inverse=True)
        available_scales = np.bincount(location_index)
        weights = 1. / (len(locations)*available_scales[location_index[inverse]]*counts[inverse])
    else:
        raise ValueError('Unknown pair reduction')
    if not np.isclose(weights.sum(), 1.):
        raise AssertionError('Pair weights must sum to one')
    return weights.astype(np.float32)


def make_pairs(labels, valid, spacing, interface, distances=(1.,), reduction='pairs'):
    """Extend a GT crossing symmetrically, requiring exactly one label transition.

    A 1 mm crossing i -> j becomes i-u -> j+u at 3 mm. Every intervening
    voxel must remain in its prescribed class and in the real field of view.
    Anchors are the original inside (or anterior) boundary voxel, not predictions.
    """
    labels, valid, spacing = np.asarray(labels), np.asarray(valid, bool), np.asarray(spacing, float)
    if labels.ndim != 3 or valid.shape != labels.shape or spacing.shape != (3,) or (spacing <= 0).any():
        raise ValueError('Invalid pair geometry')
    if interface == 'outer':
        inside, outside = labels > 0, labels == 0
    elif interface == 'ap':
        inside, outside = labels == 1, labels == 2
    else:
        raise ValueError('Unknown interface')
    shape = np.asarray(labels.shape)
    all_i, all_j, all_anchor, all_scale = [], [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            direction = np.zeros(3, int)
            direction[axis] = sign
            coords = np.argwhere(inside & valid)
            neighbors = coords + direction
            keep = ((neighbors >= 0) & (neighbors < shape)).all(1)
            coords, neighbors = coords[keep], neighbors[keep]
            keep = outside[tuple(neighbors.T)] & valid[tuple(neighbors.T)]
            anchors = coords[keep]
            for level, distance in enumerate(distances):
                n_float = (float(distance)/spacing[axis]-1)/2
                n = int(round(n_float))
                if n < 0 or not np.isclose(n_float, n, atol=1e-6):
                    continue  # Do not resample hard labels to invent this physical distance.
                left, right = anchors-n*direction, anchors+(n+1)*direction
                keep = ((left >= 0) & (left < shape) & (right >= 0) & (right < shape)).all(1)
                base = anchors[keep]
                if not len(base):
                    continue
                okay = np.ones(len(base), bool)
                for offset in range(-n, n+2):
                    points = base+offset*direction
                    group = inside if offset <= 0 else outside
                    okay &= valid[tuple(points.T)] & group[tuple(points.T)]
                base = base[okay]
                if not len(base):
                    continue
                all_i.append(np.ravel_multi_index((base-n*direction).T, labels.shape))
                all_j.append(np.ravel_multi_index((base+(n+1)*direction).T, labels.shape))
                all_anchor.append(np.ravel_multi_index(base.T, labels.shape))
                all_scale.append(np.full(len(base), level, dtype=np.int64))
    arrays = [np.concatenate(parts) if parts else np.empty(0, np.int64)
              for parts in (all_i, all_j, all_anchor, all_scale)]
    i, j, anchor, scale = arrays
    weights = pair_weights(anchor, scale, reduction)
    stats = dict(interface=interface, reduction=reduction, pairs=len(i),
                 anchors=len(np.unique(anchor)), weight_sum=float(weights.sum()),
                 by_distance={str(d): int((scale == k).sum()) for k, d in enumerate(distances)})
    return dict(i=torch.from_numpy(i), j=torch.from_numpy(j), weight=torch.from_numpy(weights),
                anchor=torch.from_numpy(anchor), scale=torch.from_numpy(scale), stats=stats)


def same_pairs(labels):
    foreground = labels > 0
    inner, outer = build_boundary_bands(foreground[None, None], steps=2)
    support = (inner | outer)[0, 0].numpy()
    fg = foreground.numpy()
    index = np.arange(fg.size).reshape(fg.shape)
    all_i, all_j = [], []
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        mask = support[left] & support[right] & (fg[left] == fg[right])
        all_i.append(index[left][mask]); all_j.append(index[right][mask])
    return dict(i=torch.from_numpy(np.concatenate(all_i)), j=torch.from_numpy(np.concatenate(all_j)))


class Targets:
    def __init__(self, root, protocol, entry):
        self.protocol, self.entry, self.cached = protocol, entry, {}

    def get(self, item, device):
        name = item['case_name']
        if name not in self.cached:
            labels = item['label'][0].long().cpu()
            info = self.protocol['native_geometry'][name]
            valid = valid_field(labels.shape, info['native_shape'])
            if np.any((labels.numpy() > 0) & ~valid):
                raise ValueError('Annotated tissue outside the native image')
            spacing = np.linalg.norm(item['affine'][:3, :3], axis=0)
            if not np.allclose(spacing, 1.):
                raise ValueError('This frozen campaign requires verified 1 mm native voxels')
            target = dict(same=same_pairs(labels))
            for interface in ('outer', 'ap'):
                target[interface] = make_pairs(labels.numpy(), valid, spacing, interface,
                    self.entry.get('distances_mm', (1.,)), self.entry.get('reduction', 'pairs'))
            # Split-control crossings retain the legacy padded-grid definition.
            target['legacy_cross'] = make_pairs(labels.numpy(), np.ones(labels.shape, bool),
                                               spacing, 'outer')
            self.cached[name] = target
        return {k: {a: b.to(device) if torch.is_tensor(b) else b for a, b in v.items()}
                for k, v in self.cached[name].items()}

    def statistics(self):
        return {name: {key: value.get('stats', dict(pairs=len(value['i'])))
                       for key, value in target.items()} for name, target in self.cached.items()}


def crossing_loss(logits, pairs, interface, form):
    z = logits.float()[0]
    score = torch.logaddexp(z[1], z[2])-z[0] if interface == 'outer' else z[1]-z[2]
    score = score.reshape(-1)
    if not pairs['i'].numel():
        return score.sum()*0
    if form == 'probability':
        probability = score.sigmoid()
        errors = (1-(probability[pairs['i']]-probability[pairs['j']])).square()
    elif form == 'score':
        errors = F.softplus(score[pairs['j']]-score[pairs['i']])
    else:
        raise ValueError('Unknown comparison')
    return (errors*pairs['weight']).sum()


def edge_value(logits, labels, entry, target):
    kind = entry['kind']
    if kind == 'legacy_edge':
        return BoundaryEdgeConsistencyLoss()(logits, labels)
    if kind in ('same_only', 'split_equal'):
        p = logits.float().softmax(1)[0, 1:].sum(0).reshape(-1)
        pair = target['same']
        same = (p[pair['i']]-p[pair['j']]).square().mean() if len(pair['i']) else p.sum()*0
        if kind == 'same_only':
            return same
        cross = crossing_loss(logits, target['legacy_cross'], 'outer', 'probability')
        return .5*(same+cross)
    if kind != 'cross':
        raise ValueError('No edge objective for '+kind)
    outer = crossing_loss(logits, target['outer'], 'outer', entry['form'])
    if not entry['ap']:
        return outer
    # One fixed total edge-gradient budget; do not silently double it with AP.
    if not target['ap']['i'].numel():
        return outer
    return .5*(outer+crossing_loss(logits, target['ap'], 'ap', entry['form']))


def components(logits, labels, entry, target, active=True):
    if logits.shape[0] != 1 or logits.shape[1] != 3 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected batch-one three-class segmentation')
    values = dict(dice=DiceLoss(to_onehot_y=True, softmax=True)(logits.float(), labels))
    if active:
        band = OuterBoundaryBandLoss()(logits, labels)
        if not bool(band.details['valid'].all()):
            raise ValueError('Missing band support')
        values['bands'] = band.loss
        if entry['kind'] != 'bands':
            values['edge'] = edge_value(logits, labels, entry, target)
    return values
