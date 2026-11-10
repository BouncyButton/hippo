"""Independent A/P supervision, added without recalibrating the winning loss."""
import math
import torch
from .ap82_matrix import VERSION
from .winner_objectives import components as base_components, regional_band
from .objectives import select_weights as original_weights
from scripts.band_evolution.objectives import independent_loss
from scripts.compute_constraint_objectives import calibrated_weight


def components(logits, labels, entry, target, active=True):
    if entry.get('objective_version') != VERSION:
        return base_components(logits, labels, entry, target, active)
    values = base_components(logits, labels, entry['base_objective'], target.get('base', {}), active)
    if not active: return values
    s, t = entry['ap82'], target['ap82']
    score = logits.float()[0,1]-logits.float()[0,2]
    if s['band_share'] > 0:
        band = dict(inner=s['width'], outer=s['width'], inside_weight=s['anterior_mass'],
                    inside_fraction=s['anterior_fraction'], outside_fraction=s['posterior_fraction'],
                    inside_hard_mix=s['anterior_emphasis'], outside_hard_mix=s['posterior_emphasis'],
                    penalty='brier')
        values['ap_band'] = regional_band(score, band, t)
    if s['band_share'] < 1:
        values['ap_pair'] = independent_loss(score, t['pairs'], dict(weights=s['endpoint_weights'],
            inside_weight=s['anterior_mass'], penalty='brier'))
    return values


def select_weights(rows, entry, settings):
    if entry.get('objective_version') != VERSION:
        return original_weights(rows, entry, settings)
    frozen = dict(entry['base_coefficients'])
    base = dict(dice=1., **frozen)
    denominators = []
    for row in rows:
        gram = row['gradient_gram']
        norm = sum(wa*wb*gram[a+':'+b] for a,wa in base.items() for b,wb in base.items())
        if norm <= 0 or not math.isfinite(norm): raise ValueError('Invalid background gradient')
        denominators.append(norm)
    s = entry['ap82']; ratio = s['gradient_ratio']; weights, stats = {}, {}
    for term, share in (('ap_band',s['band_share']), ('ap_pair',1-s['band_share'])):
        if share <= 0: continue
        ratios = [math.sqrt(r['gradient_gram'][term+':'+term]/d) for r,d in zip(rows,denominators)]
        coefficient, stats[term] = calibrated_weight(ratios, ratio, settings['cap_ratio'])
        weights[term] = coefficient*share
    joint = []
    for row,d in zip(rows,denominators):
        gram = row['gradient_gram']
        norm = sum(wa*wb*gram[a+':'+b] for a,wa in weights.items() for b,wb in weights.items())
        joint.append(math.sqrt(max(norm,0.)/d))
    multiplier, stats['ap_joint'] = calibrated_weight(joint, ratio, settings['cap_ratio'])
    frozen.update({k:v*multiplier for k,v in weights.items()})
    return dict(coefficients=frozen, statistics=stats, cases=rows, checkpoint_epoch=settings['prefix_epochs'],
                basis='Frozen background coefficients; one jointly calibrated A/P budget relative to complete background loss')


def install():
    from . import train
    from .ap82_geometry import Targets
    from scripts.depth_supervision.utils import write_history
    def extended_history(path, history):
        if not history: return
        keys = list(dict.fromkeys(k for row in history for k in row))
        write_history(path, [{k:row.get(k,0.) for k in keys} for row in history])
    train.objective_components = components
    train.select_weights = select_weights
    train.Targets = Targets
    train.TERM_ORDER = ('bands','pair','surface','ap','ap_ce','ap_band','ap_pair')
    train.write_history = extended_history
