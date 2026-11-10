"""Original loss definitions on the final native logits, with fixed calibration."""
import math
import torch
from monai.losses import DiceLoss
from scripts.compute_constraint_objectives import calibrated_weight
from scripts.depth_supervision.losses import components as depth_components
from scripts.depth_supervision.train import GeometryCache
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_maps, signed_distance_loss

BASE_OBJECTIVES = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf',
                   'dice_bands_edge', 'dice_bands_edge_sdf')
OBJECTIVES = tuple(base + suffix for base in BASE_OBJECTIVES for suffix in ('', '_depth'))
TERM_ORDER = ('bands', 'edge', 'sdf', 'depth')


def terms(objective):
    if objective not in OBJECTIVES:
        raise ValueError('Objective is outside the authorized twelve-arm matrix')
    return tuple(name for name in TERM_ORDER if name in objective.split('_'))


class Targets:
    """Cache training-label targets only; branch inputs never consume these."""
    def __init__(self, root, protocol, objective):
        self.terms = terms(objective)
        self.geometry = GeometryCache(root/'loss_geometry', protocol['geometry']) if 'depth' in self.terms else None
        self.cached = {}

    def get(self, item, device):
        name = item['case_name']
        if name not in self.cached:
            target = {}
            if 'sdf' in self.terms:
                target['sdf'] = signed_distance_maps(item['label'][None], clip=8.)
            if self.geometry is not None:
                target['weights'] = torch.as_tensor(self.geometry.get(item, fields=False)['weights'])
            self.cached[name] = target
        return {k: v.to(device) for k, v in self.cached[name].items()}


def components(logits, labels, objective, target):
    requested = terms(objective)
    values = {'dice': DiceLoss(to_onehot_y=True, softmax=True)(logits.float(), labels)}
    if 'bands' in requested:
        result = OuterBoundaryBandLoss()(logits, labels)
        if not bool(result.details['valid'].all()):
            raise ValueError('Missing inner/outer band support')
        values['bands'] = result.loss
    if 'edge' in requested:
        values['edge'] = BoundaryEdgeConsistencyLoss()(logits, labels)
    if 'sdf' in requested:
        values['sdf'] = signed_distance_loss(logits, labels, target['sdf'])
    if 'depth' in requested:
        values['depth'] = depth_components(logits, labels, target, 'depth')['depth']
    return values


def gradient_record(values, logits):
    gradients = {name: torch.autograd.grad(value, logits, retain_graph=True)[0]
                 for name, value in values.items()}
    gram = {a+':'+b: float((left*right).mean())
            for a, left in gradients.items() for b, right in gradients.items()}
    if not all(math.isfinite(v) for v in gram.values()):
        raise ValueError('Nonfinite calibration gradient')
    if any(gram[k+':'+k] <= 0 for k in gradients):
        raise ValueError('Zero calibration gradient; never silently disable a term')
    return dict(gradient_gram=gram, losses={k: float(v.detach()) for k, v in values.items()})


def select_weights(rows, objective, settings):
    """Match the original sequential Dice -> bands -> edge -> SDF -> depth rule."""
    weights, statistics = {'dice': 1.}, {}
    for term in terms(objective):
        ratios = []
        for row in rows:
            gram = row['gradient_gram']
            base = sum(wa*wb*gram[a+':'+b] for a, wa in weights.items() for b, wb in weights.items())
            if base <= 0 or not math.isfinite(base):
                raise ValueError('Degenerate combined calibration gradient')
            ratios.append(math.sqrt(gram[term+':'+term]/base))
        weights[term], statistics[term] = calibrated_weight(
            ratios, settings['target_ratio'], settings['cap_ratio'])
    del weights['dice']
    return dict(coefficients=weights, statistics=statistics, cases=rows,
                basis='Training-only final-logit gradient RMS, sequential combined-base calibration',
                checkpoint_epoch=settings['prefix_epochs'])
