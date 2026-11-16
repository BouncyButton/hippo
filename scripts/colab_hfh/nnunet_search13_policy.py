"""Bounded f0/s0 search with an explicit nonzero gradient contribution rule."""
from copy import deepcopy
import math
import numpy as np

from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.constraint_schedule import training_flops

NAME = 'nnunet_search13_a100_f0s0_v1'
CAP = 63952901334180
ARMS = ('dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_depth', 'dice_bands_depth', 'dice_edge_depth',
        'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
        'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth')
ACTIVITY = dict(minimum_native_ratio=.01, minimum_component_share=.10,
                maximum_sum_ratio=.50, maximum_anchor_multiplier=64.)


def candidates():
    # Recipe-major ordering covers every combination before deeper searches.
    result = []
    for label, target, ramp in [('standard', .10, 9), ('gentle', .05, 9),
                                 ('strong', .20, 9), ('slow', .10, 30)]:
        for arm in ARMS:
            result.append(dict(id=arm+'__'+label, arm=arm, target=target, ramp=ramp,
                               emphasis=None))
    for arm in ARMS:
        if len(components(arm)) > 1:
            for name in components(arm):
                result.append(dict(id=arm+'__emphasize_'+name, arm=arm, target=.10,
                                   ramp=9, emphasis=name))
    return result


def probe_allowance(arm):
    # Detached full-resolution logits only: no additional model forward/backward.
    # 8192 operations/logit/term covers softmax, native DS loss, two-step
    # morphology, BCE/CE, face differences, logit backwards and Gram reductions.
    # Intentionally conservative; included in EVERY active optimization charge.
    return 8192 * (2 * 64**3) * (len(components(arm)) + 1) + 1000000


def make_protocol(reference, recipe, native_anchor):
    p = deepcopy(reference)
    arm = recipe['arm']
    if p['active_objective'] != arm or p['budget_flops'] != CAP:
        raise ValueError('Wrong canonical long-budget reference')
    names = components(arm)
    p.update(name=NAME+'__'+recipe['id'], folds=[0], seeds=[0],
             a100_native_anchor=str(native_anchor), search_recipe=recipe,
             activity_policy=deepcopy(ACTIVITY), constraint_warmup_updates=recipe['ramp'],
             calibration_target_ratio=recipe['target'], constraint_joint_target=recipe['target'],
             constraint_component_targets={n:recipe['target']*(2 if n == recipe['emphasis'] else 1)
                                           for n in names},
             endpoint='Final capped endpoint; exploratory selection on f0/s0 validation only',
             constraint_definition='Unchanged canonical losses and native long recipe; frozen anchor weights plus per-batch activity floor and conservative combined cap; all components share ramp')
    s = p['models']['nnunet']['objectives'][arm]
    s.update(constraint_warmup_updates=recipe['ramp'], sdf_warmup_updates=recipe['ramp'],
             activity_probe_flops_per_update=probe_allowance(arm))
    s['early_flops_per_update'] += probe_allowance(arm)
    s['flops_per_update'] += probe_allowance(arm)
    steps = 0
    while training_flops(steps+1,s) <= CAP:
        steps += 1
    used = training_flops(steps,s)
    s.update(max_steps=steps, consumed_training_flops=used, consumed_flops=used,
             consumed_total_flops=used+s['calibration_flops'], unused_flops=CAP-used,
             patient_pass_epochs=steps/10)
    if s['prefix_updates'] != 60 or s['sdf_prefix_updates'] != 60 or steps < 150:
        raise ValueError('Long native prefix or useful constraint exposure changed')
    return p


def active_coefficients(gram, anchor, names, policy=ACTIVITY):
    """Keep anchor weights when feasible; otherwise project RMS contributions.

    Guarantees apply before the shared ramp, in full-resolution logit space.
    The triangle bound prevents cancellation from hiding excessive pressure.
    This is not a guarantee about anatomy, parameter gradients or SGD momentum.
    """
    g = np.asarray(gram, dtype=np.float64)
    w = np.asarray([anchor[n] for n in names], dtype=np.float64)
    if (g.shape != (len(names)+1,)*2 or not np.isfinite(g).all()
            or np.any(np.diag(g) <= 0) or not np.isfinite(w).all() or np.any(w <= 0)
            or not np.allclose(g,g.T,rtol=1e-6,atol=1e-30)
            or np.linalg.eigvalsh(g).min() < -1e-5*np.diag(g).max()):
        raise ValueError('Invalid activity gradients or anchor coefficients')
    raw = np.sqrt(np.diag(g)[1:]/g[0,0])
    initial = w * raw
    total = min(max(float(initial.sum()), len(names)*policy['minimum_native_ratio']),
                policy['maximum_sum_ratio'])
    floor = max(policy['minimum_native_ratio'],policy['minimum_component_share']*total)
    if len(names)*floor > total+1e-12:
        raise ValueError('Infeasible activity envelope')
    if (initial.min() >= floor and initial.sum() <= policy['maximum_sum_ratio']):
        ratios = initial.copy()
    else:
        # Euclidean projection onto a simplex with an equal lower bound.
        remaining = total-len(names)*floor
        v = initial-floor
        ordered = np.sort(v)[::-1]
        css = np.cumsum(ordered)-remaining
        indices = np.flatnonzero(ordered-css/np.arange(1,len(names)+1) > 0)
        theta = css[indices[-1]]/(indices[-1]+1) if len(indices) else float(ordered[0])
        ratios = floor + np.maximum(v-theta,0.)
    effective = ratios/raw
    if np.any(effective > w*policy['maximum_anchor_multiplier']*(1+1e-9)):
        raise ValueError('Activity requires more than 64x anchor weight; no component is disabled')
    joint = math.sqrt(max(0.,float(effective@g[1:,1:]@effective/g[0,0])))
    if ratios.min() < policy['minimum_native_ratio']-1e-10 or (ratios/ratios.sum()).min() < policy['minimum_component_share']-1e-10 or joint > policy['maximum_sum_ratio']+1e-9:
        raise ValueError('Activity floor/cap failed')
    return dict(zip(names,map(float,effective))), dict(
        component_ratios=dict(zip(names,map(float,ratios))),
        component_shares=dict(zip(names,map(float,ratios/ratios.sum()))),
        joint_ratio=joint, sum_ratio=float(ratios.sum()),
        adjusted=not np.allclose(effective,w,rtol=1e-10,atol=0.))
