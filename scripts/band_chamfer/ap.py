"""A/P cut location from conditional probabilities, never predicted FG gating."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from .geometry import edge_grid, surface_points
from scripts.edge_ablation.objectives import make_pairs
from thesis.new_constraints.ap_plane.conditional_ce import OriginalLabelAPConditionalCELoss


def ap_target(labels, valid, spacing):
    """Preserve the actual interface, including nonplanar/mixed-slice labels."""
    fg = (labels > 0) & valid
    grid = edge_grid(fg, spacing)
    gt = surface_points(torch.tensor(labels == 1, dtype=torch.float32), grid)
    coordinates = np.stack(np.indices(labels.shape), -1).reshape(-1, 3)*spacing
    distance = cKDTree(gt.numpy()).query(coordinates, workers=1)[0].reshape(labels.shape) if len(gt) else np.full(labels.shape, np.inf)
    occupied = np.flatnonzero(fg.any((0, 2)))
    candidates = np.arange(occupied[0], occupied[-1], dtype=np.int64) if len(occupied) else np.empty(0, np.int64)
    distribution = np.zeros(len(candidates), np.float32)
    if len(gt) and len(candidates):
        # Axis-edge samples can lie at integer or half-integer Y. Linear mass
        # assignment retains a mixed-slice interface instead of projecting labels.
        position = gt[:, 1].numpy()/spacing[1]-.5-candidates[0]
        lower = np.floor(position).astype(int)
        fraction = position-lower
        for offset, mass in ((0, 1-fraction), (1, fraction)):
            np.add.at(distribution, np.clip(lower+offset, 0, len(candidates)-1), mass)
        distribution /= distribution.sum()
    return dict(grid=grid, gt=gt, distance=torch.tensor(distance, dtype=torch.float32),
                support=torch.tensor(fg), anterior=torch.tensor(labels == 1),
                candidates=torch.from_numpy(candidates), target_distribution=torch.from_numpy(distribution),
                pair=make_pairs(labels, valid, spacing, 'ap', (1., 3.), 'pairs'))


def cut_distribution(score, target, temperature):
    """Existing cut-score construction generalized to actual interface targets.

    Lower Y is posterior; higher Y is anterior. Candidates span the entire
    occupied GT support, not a window chosen around the correct cut.
    """
    support = target['support'].to(score.dtype)
    counts = support.sum((0, 2))
    anterior = (-F.softplus(-score)*support).sum((0, 2))/counts.clamp_min(1)
    posterior = (-F.softplus(score)*support).sum((0, 2))/counts.clamp_min(1)
    candidates = target['candidates']
    scores = (posterior.cumsum(0)[candidates]+anterior.flip(0).cumsum(0).flip(0)[candidates+1])/(counts > 0).sum().clamp_min(1)
    return (scores/temperature).softmax(0)


def column_chamfer(probability, target, spec, runtime):
    """Same-X/Z Y-crossing comparison on columns containing a GT Y transition."""
    from .objectives import point_chamfer
    grid = target['grid']
    selected = grid['step'][:, 1] != 0
    vertical = {k: v[selected] for k, v in grid.items()}
    truth = surface_points(target['anterior'].float(), vertical)
    predicted = surface_points(probability, vertical)
    gt_xy = truth[:, (0, 2)].detach().cpu().numpy()
    pred_xy = predicted[:, (0, 2)].detach().cpu().numpy()
    columns = np.unique(gt_xy, axis=0)
    values, missing = [], 0
    for column in columns:
        a = np.flatnonzero((pred_xy == column).all(1))
        b = np.flatnonzero((gt_xy == column).all(1))
        if not len(a):
            missing += 1
            continue
        a = torch.tensor(a, device=probability.device)
        b = torch.tensor(b, device=probability.device)
        values.append(point_chamfer(predicted[a], truth[b], spec))
    if runtime is not None:
        runtime['ap_column_calls'] = runtime.get('ap_column_calls', 0)+1
        runtime['ap_columns_total'] = runtime.get('ap_columns_total', 0)+len(columns)
        runtime['ap_columns_missing'] = runtime.get('ap_columns_missing', 0)+missing
    return torch.stack(values).mean() if values else probability.sum()*0


def ap_loss(logits, spec, target, runtime=None):
    from .objectives import pair_loss, point_chamfer, weighted_sides
    score = logits.float()[0, 1]-logits.float()[0, 2]
    probability = score.sigmoid()
    kind = spec['kind']
    if kind == 'pair':
        return pair_loss(probability, target['pair'], dict(form='coupled'))
    if kind in ('cut_w1', 'cut_expected_l1'):
        if not len(target['gt']) or not len(target['candidates']):
            return score.sum()*0
        q = cut_distribution(score, target, spec['temperature'])
        r = target['target_distribution']
        if kind == 'cut_w1':
            return (q.cumsum(0)-r.cumsum(0)).abs()[:-1].sum()  # spacing_y=1 mm verified
        locations = target['candidates'].to(q.dtype)+.5
        return (q[:, None]*r[None, :]*(locations[:, None]-locations[None, :]).abs()).sum()
    if kind == 'sdf':
        error = torch.where(target['anterior'], 1-probability, probability)
        weight = target['distance'].clamp_max(spec['clip_mm'])/spec['clip_mm']
        return weighted_sides(error*weight, target['support'] & target['anterior'], target['support'] & ~target['anterior'])
    if kind == 'column_chamfer':
        return column_chamfer(probability, target, spec, runtime)
    if kind != 'chamfer':
        raise ValueError('Unknown A/P objective')
    points = surface_points(probability, target['grid'])
    if runtime is not None:
        runtime['ap_surface_calls'] = runtime.get('ap_surface_calls', 0)+1
        runtime['ap_empty_surface_calls'] = runtime.get('ap_empty_surface_calls', 0)+int(not len(points) or not len(target['gt']))
    if not len(points) or not len(target['gt']):
        return score.sum()*0
    return point_chamfer(points, target['gt'], spec)


def ap_ce_loss(logits, labels, spec, target):
    if spec['kind'] == 'full':
        return OriginalLabelAPConditionalCELoss()(logits, labels).loss
    support = target['support'] & (target['distance'] <= spec['width_mm'])
    score = logits.float()[0, 1]-logits.float()[0, 2]
    if not support.any():
        return score.sum()*0
    return F.binary_cross_entropy_with_logits(score[support], target['anterior'][support].float())
