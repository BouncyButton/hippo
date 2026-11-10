"""Training-label geometry and piecewise differentiable isosurface samples."""
import hashlib
from collections import OrderedDict
import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.edge_ablation.objectives import make_pairs, valid_field
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands

# Shared immutable CPU label geometry avoids recomputing the same 64-cubed
# nearest-distance maps for every objective during preflight/calibration. Keys
# include label bytes, FOV, spacing and shape; never depend on case names alone.
_GEOMETRY_CACHE = OrderedDict()
_AP_CACHE = OrderedDict()


def geometry_key(labels, valid, spacing):
    return (labels.shape, hashlib.sha256(labels.astype(np.uint8).tobytes()+valid.tobytes()+spacing.tobytes()).hexdigest())


def retain(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > 12:
        cache.popitem(last=False)
    return value


def edge_grid(valid, spacing):
    """All positive-axis native-FOV edges; no artificial crop/padding surfaces."""
    index = np.arange(valid.size).reshape(valid.shape)
    ii, jj, xyz, direction = [], [], [], []
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        keep = valid[left] & valid[right]
        a, b = index[left][keep], index[right][keep]
        ii.append(a); jj.append(b)
        xyz.append(np.stack(np.unravel_index(a, valid.shape), 1)*spacing)
        step = np.zeros((len(a), 3), np.float32)
        step[:, axis] = spacing[axis]
        direction.append(step)
    return dict(i=torch.from_numpy(np.concatenate(ii)), j=torch.from_numpy(np.concatenate(jj)),
                xyz=torch.tensor(np.concatenate(xyz), dtype=torch.float32),
                step=torch.tensor(np.concatenate(direction), dtype=torch.float32))


def surface_points(probability, grid):
    """Linear p=0.5 edge intersections, differentiable within fixed topology.

    The crossing decisions are discrete. Coordinates retain their dependence on
    both endpoint probabilities; nearest-neighbor decisions later are discrete
    too. This is a piecewise differentiable point-sample Chamfer, not a mesh-area
    integral. Degenerate shallow slopes use a documented 1e-6 denominator floor.
    """
    p = probability.reshape(-1)
    a, b = p[grid['i']], p[grid['j']]
    crossing = (a.detach() >= .5) != (b.detach() >= .5)
    a, b = a[crossing], b[crossing]
    delta = b-a
    safe = torch.where(delta >= 0, delta.clamp_min(1e-6), delta.clamp_max(-1e-6))
    fraction = ((.5-a)/safe).clamp(0., 1.)
    return grid['xyz'][crossing].to(p.dtype)+fraction[:, None]*grid['step'][crossing].to(p.dtype)


def pair_geometry(labels, valid, spacing, spec):
    """Control eligibility independently of scale weights and endpoint shifts."""
    if spec['support'] == 'original':
        pair = make_pairs(labels, valid, spacing, 'outer', tuple(spec['distances']), 'pairs')
        out = {k: pair[k] for k in ('i', 'j', 'scale')}
    else:
        one = make_pairs(labels, valid, spacing, 'outer', (1.,), 'pairs')
        a = np.stack(np.unravel_index(one['i'].numpy(), labels.shape), 1)
        b = np.stack(np.unravel_index(one['j'].numpy(), labels.shape), 1)
        direction = b-a
        extent = (-2, 3) if spec['support'] == 'common_shift' else (-1, 2)
        okay = np.ones(len(a), bool)
        for offset in range(extent[0], extent[1]+1):
            x = a+offset*direction
            bounded = ((x >= 0) & (x < np.asarray(labels.shape))).all(1)
            x = np.clip(x, 0, np.asarray(labels.shape)-1)
            okay &= bounded & valid[tuple(x.T)] & ((labels[tuple(x.T)] > 0) == (offset <= 0))
        a, direction = a[okay], direction[okay]
        ii, jj, levels = [], [], []
        for level, distance in enumerate(spec['distances']):
            offsets = (0, 1) if distance == 1 else dict(symmetric=(-1, 2), outward=(0, 3), inward=(-2, 1))[spec.get('shift', 'symmetric')]
            ii.append(np.ravel_multi_index((a+offsets[0]*direction).T, labels.shape))
            jj.append(np.ravel_multi_index((a+offsets[1]*direction).T, labels.shape))
            levels.append(np.full(len(a), level, np.int64))
        out = {k: torch.from_numpy(np.concatenate(v)) for k, v in zip(('i', 'j', 'scale'), (ii, jj, levels))}
    weights = torch.zeros(len(out['i']), dtype=torch.float32)
    available = [(k, w) for k, w in enumerate(spec['weights']) if (out['scale'] == k).any()]
    for k, w in available:
        keep = out['scale'] == k
        weights[keep] = w/sum(q for _, q in available)/int(keep.sum())
    out['weight'] = weights
    return out


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


class Targets:
    def __init__(self, root, protocol, entry):
        self.protocol, self.entry, self.cached, self.stats = protocol, entry, {}, {}

    def get(self, item, device):
        name = item['case_name']
        if name not in self.cached:
            labels = item['label'][0].long().cpu()
            y = labels.numpy()
            valid = valid_field(y.shape, self.protocol['native_geometry'][name]['native_shape'])
            spacing = np.linalg.norm(item['affine'][:3, :3], axis=0)
            if not np.allclose(spacing, 1.) or np.any((y > 0) & ~valid):
                raise ValueError('Require verified native 1 mm data and annotations inside the FOV')
            key = geometry_key(y, valid, spacing)
            if key not in _GEOMETRY_CACHE:
                grid = edge_grid(valid, spacing)
                gt = surface_points((labels > 0).float(), grid)
                points = np.stack(np.indices(y.shape), -1).reshape(-1, 3)*spacing
                distance = cKDTree(gt.numpy()).query(points, workers=1)[0].reshape(y.shape) if len(gt) else np.full(y.shape, np.inf)
                retain(_GEOMETRY_CACHE, key, dict(valid=torch.from_numpy(valid), foreground=labels > 0,
                       distance=torch.tensor(distance, dtype=torch.float32), grid=grid, gt=gt))
            target = dict(_GEOMETRY_CACHE[key])
            gt = target['gt']
            band = self.entry['band']
            if band['kind'] == 'morph':
                inner, outer = build_boundary_bands((labels > 0)[None, None], steps=band['width'])
                target['inner'] = inner[0, 0] & target['valid']
                target['outer'] = outer[0, 0] & target['valid']
            if self.entry['pair']:
                target['pair'] = pair_geometry(y, valid, spacing, self.entry['pair'])
            if self.entry.get('ap') or self.entry.get('ap_ce'):
                from .ap import ap_target
                if not np.allclose(item['affine'][:3, :3], np.eye(3)):
                    raise ValueError('A/P objectives require verified RAS identity axes')
                if key not in _AP_CACHE:
                    retain(_AP_CACHE, key, ap_target(y, valid, spacing))
                target['ap'] = _AP_CACHE[key]
            self.stats[name] = dict(gt_surface_points=len(gt), valid_voxels=int(valid.sum()),
                                   foreground_voxels=int((y > 0).sum()), surface_calls=0,
                                   empty_prediction_calls=0, empty_target_calls=0,
                                   pair_counts={str(d): int((target['pair']['scale'] == k).sum())
                                                for k, d in enumerate(self.entry['pair']['distances'])} if self.entry['pair'] else {})
            if 'ap' in target:
                self.stats[name]['ap_gt_surface_points'] = len(target['ap']['gt'])
                self.stats[name]['ap_target_y_bins'] = int((target['ap']['target_distribution'] > 0).sum())
            self.cached[name] = target
        target = move(self.cached[name], device)
        target['runtime'] = self.stats[name]
        return target

    def statistics(self):
        return self.stats
