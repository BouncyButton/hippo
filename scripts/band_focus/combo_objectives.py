"""Established SDF, edge and depth terms; preserve winning outer coefficients."""
import math
import torch
from scripts.compute_constraint_objectives import calibrated_weight
from scripts.depth_supervision.losses import components as depth_components
from scripts.depth_supervision.utils import write_history as original_write_history
from scripts.native_branch_constraints.objectives import Targets as ExtraTargets
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_loss
from .geometry import Targets as OuterTargets
from .winner_objectives import components as base_components
from .combo_matrix import VERSION, TERMS


class Targets:
    def __init__(self, root, protocol, entry):
        self.outer = OuterTargets(root, protocol, entry['base_objective'])
        # This name selects the original target builder; it does not add original bands.
        name = 'dice_bands' + ''.join('_'+k for k in ('edge', 'sdf', 'depth')
                                    if k in entry['combo']['terms'])
        self.extra = ExtraTargets(root, protocol, name)

    def get(self, item, device):
        return dict(base=self.outer.get(item, device), extra=self.extra.get(item, device))

    def statistics(self):
        return self.outer.statistics()


def components(logits, labels, entry, target, active=True):
    assert entry['objective_version'] == VERSION
    values = base_components(logits, labels, entry['base_objective'], target.get('base', {}), active)
    if not active:
        return values
    terms = entry['combo']['terms']
    assert len(set(terms)) == len(terms) and set(terms) <= set(TERMS)
    extra = target['extra']
    if 'sdf' in terms:
        values['sdf'] = signed_distance_loss(logits, labels, extra['sdf'])
    if 'edge' in terms:
        values['edge'] = BoundaryEdgeConsistencyLoss()(logits, labels)
    if 'depth' in terms:
        values['depth'] = depth_components(logits, labels, extra, 'depth')['depth']
    return values


def select_weights(rows, entry, settings):
    frozen = dict(entry['base_coefficients'])
    terms = entry['combo']['terms']
    stats = {}
    if terms:
        base = dict(dice=1., **frozen)
        denominators = []
        for row in rows:
            gram = row['gradient_gram']
            norm = sum(wa*wb*gram[a+':'+b] for a, wa in base.items() for b, wb in base.items())
            if norm <= 0 or not math.isfinite(norm):
                raise ValueError('Invalid frozen-base calibration gradient')
            denominators.append(norm)
        ratio = entry['combo']['gradient_ratio']
        if ratio <= 0:
            raise ValueError('Requested auxiliary has no gradient budget')
        weights = {}
        # Equal shares after individual gradient normalization; one total budget.
        for term in terms:
            raw = [math.sqrt(r['gradient_gram'][term+':'+term]/d) for r, d in zip(rows, denominators)]
            weight, stats[term] = calibrated_weight(raw, ratio, settings['cap_ratio'])
            weights[term] = weight/len(terms)
        joint = []
        for row, denominator in zip(rows, denominators):
            gram = row['gradient_gram']
            norm = sum(wa*wb*gram[a+':'+b] for a, wa in weights.items() for b, wb in weights.items())
            joint.append(math.sqrt(max(0., norm)/denominator))
        multiplier, stats['added_joint'] = calibrated_weight(joint, ratio, settings['cap_ratio'])
        frozen.update({k: v*multiplier for k, v in weights.items()})
    return dict(coefficients=frozen, statistics=stats, cases=rows,
                checkpoint_epoch=settings['prefix_epochs'],
                basis='Frozen Dice+outer bands+endpoints; equally normalized auxiliaries share one joint logit-gradient budget')


def write_history(path, history):
    """The copied Dice warmup predates the new auxiliary columns."""
    keys = list(dict.fromkeys(k for row in history for k in row))
    original_write_history(path, [{k: row.get(k, 0.) for k in keys} for row in history])
