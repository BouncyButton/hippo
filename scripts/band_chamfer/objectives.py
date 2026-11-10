"""Band, pair and surface losses on ordinary SwinUNETR probabilities."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from scipy.spatial import cKDTree
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from scripts.compute_constraint_objectives import calibrated_weight
from .geometry import Targets, surface_points


def foreground_score(logits):
    z = logits.float()[0]
    return torch.logaddexp(z[1], z[2])-z[0]


def weighted_sides(values, inner, outer, outer_weight=.5):
    sides = [(mask, weight) for mask, weight in ((inner, 1-outer_weight), (outer, outer_weight))
             if mask.any() and weight > 0]
    if not sides:
        return values.sum()*0
    return sum(weight*values[mask].mean() for mask, weight in sides)/sum(weight for _, weight in sides)


def band_loss(logits, labels, spec, target):
    if spec['kind'] == 'canonical':
        return OuterBoundaryBandLoss()(logits, labels).loss
    score = foreground_score(logits)
    fg, valid, distance = [target[k] for k in ('foreground', 'valid', 'distance')]
    truth = fg.float()
    if spec.get('soft_mm'):
        signed = torch.where(fg, -distance, distance)
        truth = (-signed/spec['soft_mm']).sigmoid()
    error = F.binary_cross_entropy_with_logits(score, truth, reduction='none')
    if spec['kind'] == 'morph':
        return weighted_sides(error, target['inner'], target['outer'])
    inner = fg & valid & (distance <= spec['inner'])
    outer = ~fg & valid & (distance <= spec['outer'])
    if spec.get('shells'):
        parts, lower = [], 0.
        for upper in spec['shells']:
            shell = (distance > lower) & (distance <= upper)
            if (inner & shell).any() or (outer & shell).any():
                parts.append(weighted_sides(error, inner & shell, outer & shell))
            lower = upper
        return torch.stack(parts).mean() if parts else score.sum()*0
    return weighted_sides(error, inner, outer, spec.get('outer_weight', .5))


def pair_loss(probability, pair, spec):
    p = probability.reshape(-1)
    if not len(pair['i']):
        return p.sum()*0
    a, b = p[pair['i']], p[pair['j']]
    if spec['form'] == 'coupled':
        error = (1-(a-b)).square()
    elif spec['form'] == 'independent':
        error = .5*((1-a).square()+b.square())
    elif spec['form'] == 'hinge':
        error = F.relu(spec['margin']-(a-b)).square()
    else:
        raise ValueError('Unknown pair form')
    return (error*pair['weight']).sum()


def point_chamfer(predicted, truth, spec):
    """Nearest assignments detached; distances retain gradients to coordinates.

    cKDTree computes exact nearest indices on CPU. Recomputing selected distances
    in torch gives the correct derivative away from nearest-neighbor ties and
    topology changes. Includes both directions, no point subsampling.
    """
    if not len(predicted) or not len(truth):
        return predicted.sum()*0
    pred_np, truth_np = predicted.detach().cpu().numpy(), truth.detach().cpu().numpy()
    to_truth = torch.as_tensor(cKDTree(truth_np).query(pred_np, workers=1)[1], device=predicted.device)
    to_pred = torch.as_tensor(cKDTree(pred_np).query(truth_np, workers=1)[1], device=predicted.device)
    distances = (torch.linalg.vector_norm(predicted-truth[to_truth], dim=1),
                 torch.linalg.vector_norm(truth-predicted[to_pred], dim=1))
    parts = []
    for distance in distances:
        distance = F.relu(distance-spec.get('tolerance_mm', 0.))
        penalty = spec.get('penalty', 'linear')
        if penalty == 'squared':
            value = distance.square()
        elif penalty == 'huber':
            value = torch.where(distance < 1., .5*distance.square(), distance-.5)
        elif penalty == 'linear':
            value = distance
        else:
            raise ValueError('Unknown Chamfer penalty')
        mean = value.mean()
        if spec.get('tail_fraction'):
            k = max(1, math.ceil(len(value)*spec['tail_fraction']))
            mean = .5*(mean+value.topk(k).values.mean())
        parts.append(mean)
    weight = spec.get('predicted_weight', .5)
    return weight*parts[0]+(1-weight)*parts[1]


def surface_loss(probability, spec, target):
    fg, valid, distance = [target[k] for k in ('foreground', 'valid', 'distance')]
    if spec['kind'] != 'chamfer':
        weight = distance.clamp_max(spec['clip_mm'])/spec['clip_mm']
        error = torch.where(fg, 1-probability, probability)
        if spec['kind'] == 'distance_brier':
            error = error.square()
            weight = weight.pow(spec['power'])
        elif spec['kind'] != 'sdf':
            raise ValueError('Unknown surface loss')
        return weighted_sides(error*weight, fg & valid, ~fg & valid)
    points = surface_points(probability, target['grid'])
    runtime = target.get('runtime')
    if runtime is not None:
        runtime['surface_calls'] += 1
        runtime['empty_prediction_calls'] += int(len(points) == 0)
        runtime['empty_target_calls'] += int(len(target['gt']) == 0)
    if not len(points) or not len(target['gt']):
        # Chamfer is undefined on empty point sets: zero this component only.
        # Dice and bands remain active; telemetry makes every fallback visible.
        return probability.sum()*0
    return point_chamfer(points, target['gt'], spec)


def components(logits, labels, entry, target, active=True):
    if logits.shape[0] != 1 or logits.shape[1] != 3 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected batch-one three-class segmentation')
    values = dict(dice=DiceLoss(to_onehot_y=True, softmax=True)(logits.float(), labels))
    if active:
        values['bands'] = band_loss(logits, labels, entry['band'], target)
        probability = foreground_score(logits).sigmoid()
        if entry['pair']:
            values['pair'] = pair_loss(probability, target['pair'], entry['pair'])
        if entry['surface']:
            values['surface'] = surface_loss(probability, entry['surface'], target)
        if entry.get('ap') or entry.get('ap_ce'):
            from .ap import ap_loss, ap_ce_loss
            if entry.get('ap'):
                values['ap'] = ap_loss(logits, entry['ap'], target['ap'], target.get('runtime'))
            if entry.get('ap_ce'):
                values['ap_ce'] = ap_ce_loss(logits, labels, entry['ap_ce'], target['ap'])
    return values


def select_weights(rows, entry, settings):
    """Calibrate bands, then one shared extra budget for pair/surface terms."""
    statistics = {}
    band_ratios = [math.sqrt(r['gradient_gram']['bands:bands']/r['gradient_gram']['dice:dice']) for r in rows]
    band, statistics['bands'] = calibrated_weight(band_ratios, settings['target_ratio'], settings['cap_ratio'])
    weights = dict(bands=band)
    extras = [k for k in ('pair', 'surface', 'ap', 'ap_ce') if entry.get(k)]
    if extras:
        base = []
        for row in rows:
            g = row['gradient_gram']
            norm = g['dice:dice']+2*band*g['dice:bands']+band*band*g['bands:bands']
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Degenerate calibration base')
            base.append(norm)
        normalized = {}
        for key in extras:
            ratios = [math.sqrt(row['gradient_gram'][key+':'+key]/b) for row, b in zip(rows, base)]
            normalized[key], statistics[key+'_initial'] = calibrated_weight(ratios, settings['target_ratio'], settings['cap_ratio'])
            if entry.get('extra_shares'):
                normalized[key] *= entry['extra_shares'][key]
        if len(extras) == 1:
            weights.update(normalized)
            statistics['extra_total'] = statistics[extras[0]+'_initial']
        else:
            ratios = []
            for row, b in zip(rows, base):
                g = row['gradient_gram']
                norm = sum(normalized[a]*normalized[c]*g[a+':'+c] for a in extras for c in extras)
                ratios.append(math.sqrt(max(norm, 0.)/b))
            joint, statistics['extra_total'] = calibrated_weight(ratios, settings['target_ratio'], settings['cap_ratio'])
            weights.update({k: v*joint for k, v in normalized.items()})
    return dict(coefficients=weights, statistics=statistics, cases=rows,
                basis='Training-only logit-gradient RMS; bands then ONE joint extra budget; fixed after common warmup',
                checkpoint_epoch=settings['prefix_epochs'])
