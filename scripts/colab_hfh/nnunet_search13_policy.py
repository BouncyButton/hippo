"""Runtime utilities for the archived HFH protocol."""
import math

import numpy as np

from scripts.colab_hfh.constraint_calibration import components

ACTIVITY = dict(minimum_native_ratio=.01, minimum_component_share=.10,
                maximum_sum_ratio=.50, maximum_anchor_multiplier=64.)

def probe_allowance(arm):
    # Detached full-resolution logits only: no additional model forward/backward.
    # 8192 operations/logit/term covers softmax, native DS loss, two-step
    # morphology, BCE/CE, face differences, logit backwards and Gram reductions.
    # Intentionally conservative; included in EVERY active optimization charge.
    return 8192 * (2 * 64**3) * (len(components(arm)) + 1) + 1000000

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
