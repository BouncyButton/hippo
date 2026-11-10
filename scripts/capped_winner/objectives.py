"""Selected loss, with each model's original region objective left intact."""
from copy import deepcopy
import torch
from scripts.capped_depth_delay.objectives import training_loss as baseline_loss, depth_loss
from scripts.band_focus.winner_objectives import regional_band
from scripts.band_focus.objectives import select_weights as outer_weights
from scripts.band_focus.combo_objectives import select_weights as combo_weights
from scripts.band_chamfer.objectives import foreground_score
from scripts.band_evolution.objectives import independent_loss
from scripts.native_branch_constraints.objectives import gradient_record

ARM = 'dice_focus_bands_pair_depth'
ARMS = ('dice', ARM)
BAND = dict(kind='regional', inner=3, outer=2, penalty='brier', inside_weight=.5,
            hard_fraction=.25, hard_mix=.75, inside_fraction=.7, outside_fraction=.1,
            inside_hard_mix=.65, outside_hard_mix=.85)
PAIR = dict(distances=[1, 3], weights=[.5, .5], support='all', form='independent')
ENTRY = dict(band=BAND, pair=PAIR, surface=None, ap=None, band_budget=1.25,
             extra_budget=.75, extra_shares=None)
SETTINGS = dict(prefix_epochs=5, target_ratio=.1, cap_ratio=.5)


def auxiliary(logits, labels, target, upper_bound=False):
    score = foreground_score(logits)
    if upper_bound:
        # Each side can contain at most N voxels. Summing the two complete
        # one-sided graphs dominates the actual two-sided reduction.
        inner = dict(target, foreground=torch.ones_like(target['foreground']))
        outer = dict(target, foreground=torch.zeros_like(target['foreground']))
        bands = regional_band(score, BAND, inner) + regional_band(score, BAND, outer)
    else:
        bands = regional_band(score, BAND, target)
    return dict(bands=bands, pair=independent_loss(score, target['pairs'], PAIR),
                depth=depth_loss(logits, labels, target['depth']))


def training_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0.):
    if arm not in ARMS:
        raise ValueError('Unexpected constrained objective')
    region = baseline_loss(outputs, labels, 'dice', {}, scale, criterion=criterion)
    if arm == 'dice':
        return region
    logits = outputs if criterion is None else outputs[0]
    terms = auxiliary(logits, labels, depth_weights)
    for name in ('bands', 'pair', 'depth'):
        region = region + (weights[name] * scale) * terms[name]
    return region


def calibration_case(outputs, labels, arm, target_sdf=None, criterion=None, depth_weights=None,
                     upper_bound=False):
    detached = (outputs.detach().float().requires_grad_(True) if criterion is None else
                [value.detach().float() for value in outputs])
    z = detached if criterion is None else detached[0].requires_grad_(True)
    values = dict(dice=baseline_loss(detached, labels, 'dice', {}, 1., criterion=criterion))
    values.update(auxiliary(z, labels, depth_weights, upper_bound))
    return gradient_record(values, z)


def select_weights(rows, arm):
    if arm != ARM:
        raise ValueError('Calibration is only defined for the frozen winner')
    base = outer_weights(rows, ENTRY, SETTINGS)
    entry = dict(base_coefficients=base['coefficients'], combo=dict(terms=['depth'], gradient_ratio=.1))
    result = combo_weights(rows, entry, SETTINGS)
    return result['coefficients'], dict(outer=base['statistics'], depth=result['statistics'])
