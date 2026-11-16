"""Runtime utilities for the archived HFH protocol."""
import itertools

import numpy as np

from scripts.colab_hfh.constraint_calibration import components, select_joint_weights

from scripts.colab_hfh.constraint_diagnostics import fixed_weight_statistics

ANCHOR = {'after_update': 60, 'model_state_sha256':
          '4809a3cda62d96b53fb043cf131f648d91e577d6522cb70f4835799cc0070136'}

def subset_envelope(rows, arm, weights):
    names = components(arm)
    # Use the canonical validator, including PSD and finite-value checks.
    select_joint_weights(rows, arm, targets={n: .1 for n in names},
                         component_cap=.5, joint_target=.1, joint_cap=.5)
    grams = np.asarray([r['gradient_gram'] for r in rows], dtype=np.float64)
    coefficients = np.array([weights[n] for n in names], dtype=np.float64)
    if set(weights) != set(names) or not np.isfinite(coefficients).all() or np.any(coefficients <= 0):
        raise ValueError('Incomplete or invalid fixed coefficients')
    phases = []
    for mask in itertools.product((0., 1.), repeat=len(names)):
        if not any(mask):
            continue
        active = coefficients * mask
        squared = np.einsum('i,nij,j->n', active, grams[:, 1:, 1:], active)
        phases.append(np.sqrt(np.maximum(squared, 0.) / grams[:, 0, 0]))
    worst = np.max(phases, axis=0)
    return dict(median=float(np.median(worst)), p95=float(np.quantile(worst, .95)))

def select_fixed_weights(rows, arm, specification):
    if arm != specification['objective']:
        raise ValueError('Composition objective differs')
    weights = dict(specification['weights'])
    bounds = subset_envelope(rows, arm, weights)
    measured = fixed_weight_statistics(rows, arm, weights)
    stats = {n: dict(weighted_median_ratio=v['median'], weighted_p95_ratio=v['p95'])
             for n, v in measured.items()}
    stats['joint'].update(target=.1, p95_cap=.5,
        worst_ramp_median_ratio=bounds['median'], worst_ramp_p95_ratio=bounds['p95'],
        gradient_space='detached_fp32_full_resolution_logits',
        interpretation='Fixed coefficients. Envelope enforced at the native anchor, not clipped at later diagnostics.')
    return weights, stats

def check_anchor(report, specification):
    if report['anchor_model_sha256'] != ANCHOR['model_state_sha256']:
        raise ValueError('Matched native model differs')
    actual, expected = report['cases'], specification['anchor_rows']
    if len(actual) != 10 or [r['case_name'] for r in actual] != [r['case_name'] for r in expected]:
        raise ValueError('Calibration training cases differ')
    if any(a['gradient_names'] != b['gradient_names'] for a,b in zip(actual,expected)):
        raise ValueError('Calibration gradient names differ')
    if not np.allclose([r['gradient_gram'] for r in actual], [r['gradient_gram'] for r in expected], rtol=1e-6, atol=1e-25):
        raise ValueError('Calibration gradients differ from frozen references')
    if report['weights'] != specification['weights']:
        raise ValueError('Frozen coefficients changed')
    bounds = subset_envelope(actual, specification['objective'], report['weights'])
    if bounds['median'] > .1+1e-7 or bounds['p95'] > .5+1e-7:
        raise ValueError('Anchor envelope exceeded')
