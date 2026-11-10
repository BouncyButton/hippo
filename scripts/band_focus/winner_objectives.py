"""Optional side-specific feedback; frozen objectives retain their original path."""
import torch
from . import objectives as original

VERSION = 'winner_v1'
SIDE_KEYS = ('hard_mix', 'penalty', 'bce_share', 'neighborhood', 'point_rank_share')


def regional_band(score, spec, target):
    """Reduce each side independently, then apply its explicit total loss mass."""
    if not any(name+'_'+key in spec for name in ('inside', 'outside') for key in SIDE_KEYS):
        return original.regional_band(score, spec, target)
    if spec.get('multiscale'):
        raise ValueError('Side overrides require explicit, single-scale bands')
    values = []
    for inside, name, width, mass in (
        (True, 'inside', spec['inner'], spec.get('inside_weight', .5)),
        (False, 'outside', spec['outer'], 1-spec.get('inside_weight', .5)),
    ):
        support = (target['foreground'] if inside else ~target['foreground']) & target['valid']
        support = support & (target['distance'] > 0) & (target['distance'] <= width)
        if mass <= 0 or not support.any():
            continue
        side = dict(spec, inside_weight=1. if inside else 0.)
        for key in SIDE_KEYS:
            if name+'_'+key in spec:
                side[key] = spec[name+'_'+key]
        values.append((mass, original.regional_band(score, side, target)))
    return sum(mass*value for mass, value in values)/sum(mass for mass, _ in values) if values else score.sum()*0


def components(logits, labels, entry, target, active=True):
    if entry.get('objective_version') != VERSION:
        return original.components(logits, labels, entry, target, active)
    values = dict(dice=original.DiceLoss(to_onehot_y=True, softmax=True)(logits.float(), labels))
    if not active:
        return values
    assert entry['band']['kind'] == 'regional'
    score = original.previous.foreground_score(logits)
    values['bands'] = regional_band(score, entry['band'], target)
    if entry['pair']:
        values['pair'] = original.independent_loss(score, target['pairs'], entry['pair'])
    assert entry['surface'] is None and entry.get('ap') is None and entry.get('ap_ce') is None
    return values


def install():
    """Use the extension only in its own worker process; no frozen files change."""
    from . import train
    train.objective_components = components
