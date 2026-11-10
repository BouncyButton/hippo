"""Keep original loss definitions and the established winner unchanged."""
from copy import deepcopy
import math

from monai.losses import DiceLoss

from scripts.native_branch_constraints import objectives as original
from scripts.band_focus import combo_objectives
from scripts.capped_winner.objectives import BAND, PAIR, ENTRY

ROWS = (
    ('dice', ()),
    ('dice_bands', ('bands',)),
    ('dice_sdf', ('sdf',)),
    ('dice_bands_sdf', ('bands', 'sdf')),
    ('dice_bands_edge', ('bands', 'edge')),
    ('dice_bands_edge_sdf', ('bands', 'edge', 'sdf')),
    ('dice_depth', ('depth',)),
    ('dice_bands_depth', ('bands', 'depth')),
    ('dice_edge_depth', ('edge', 'depth')),
    ('dice_sdf_depth', ('sdf', 'depth')),
    ('dice_bands_edge_depth', ('bands', 'edge', 'depth')),
    ('dice_bands_sdf_depth', ('bands', 'sdf', 'depth')),
    ('dice_edge_sdf_depth', ('edge', 'sdf', 'depth')),
    ('dice_bands_edge_sdf_depth', ('bands', 'edge', 'sdf', 'depth')),
)


def entries():
    rows = [dict(id=name, terms=list(terms), objective_version='original_contenders_v1',
                 arm='ordinary_swinunetr', fraction='5pct', fold=0, seed=0,
                 combo=dict(terms=list(terms))) for name, terms in ROWS]
    base = dict(deepcopy(ENTRY), id='w81_strength_qi70_qo10_mi65_mo85_b125_p75',
                objective_version='winner_v1', ap_ce=None)
    winner = dict(id='winning_constraint', objective_version='outer_combo_v1',
                  arm='ordinary_swinunetr', fraction='5pct', fold=0, seed=0,
                  base_objective=base,
                  base_coefficients=dict(bands=.040404281348296614, pair=.006424836560449451),
                  combo=dict(terms=['depth'], gradient_ratio=.1))
    # Fresh baseline and winner replay first; remaining rows retain image order.
    return rows[:1] + [winner] + rows[1:]


class Targets:
    def __init__(self, root, protocol, entry):
        self.winner = entry['objective_version'] == 'outer_combo_v1'
        if self.winner:
            self.source = combo_objectives.Targets(root, protocol, entry)
        else:
            # The original builder only creates SDF/depth targets. Its authorized
            # name includes bands even for the two new edge-without-bands rows.
            name = 'dice_bands' + ''.join('_'+k for k in ('edge', 'sdf', 'depth')
                                         if k in entry['terms'])
            self.source = original.Targets(root, protocol, name)

    def get(self, item, device):
        return self.source.get(item, device)

    def statistics(self):
        return self.source.statistics() if self.winner else dict(kind='original_constraints')


def components(logits, labels, entry, target, active=True):
    if entry['objective_version'] == 'outer_combo_v1':
        return combo_objectives.components(logits, labels, entry, target, active)
    values = dict(dice=DiceLoss(to_onehot_y=True, softmax=True)(logits.float(), labels))
    if not active:
        return values
    requested = entry['terms']
    assert requested == [k for k in original.TERM_ORDER if k in requested]
    if 'bands' in requested:
        result = original.OuterBoundaryBandLoss()(logits, labels)
        if not bool(result.details['valid'].all()):
            raise ValueError('Missing original inner/outer band support')
        values['bands'] = result.loss
    if 'edge' in requested:
        values['edge'] = original.BoundaryEdgeConsistencyLoss()(logits, labels)
    if 'sdf' in requested:
        values['sdf'] = original.signed_distance_loss(logits, labels, target['sdf'])
    if 'depth' in requested:
        values['depth'] = original.depth_components(logits, labels, target, 'depth')['depth']
    return values


def select_weights(rows, entry, settings):
    if entry['objective_version'] == 'outer_combo_v1':
        return combo_objectives.select_weights(rows, entry, settings)
    weights, statistics = {'dice': 1.}, {}
    for term in entry['terms']:
        ratios = []
        for row in rows:
            gram = row['gradient_gram']
            base = sum(wa*wb*gram[a+':'+b] for a, wa in weights.items()
                       for b, wb in weights.items())
            if base <= 0 or not math.isfinite(base):
                raise ValueError('Invalid combined calibration gradient')
            ratios.append(math.sqrt(gram[term+':'+term]/base))
        weights[term], statistics[term] = original.calibrated_weight(
            ratios, settings['target_ratio'], settings['cap_ratio'])
    del weights['dice']
    return dict(coefficients=weights, statistics=statistics, cases=rows,
                checkpoint_epoch=settings['prefix_epochs'],
                basis='Original training-only sequential Dice -> bands -> edge -> SDF -> depth gradient calibration')
