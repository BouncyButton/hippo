"""Frozen long-budget composition ablations; never changes the native loss."""
from copy import deepcopy
import itertools

import numpy as np

from scripts.colab_hfh.constraint_calibration import components, select_joint_weights
from scripts.colab_hfh.constraint_diagnostics import fixed_weight_statistics

NAME = 'nnunet_composition_v1'
CAP = 63952901334180
ANCHOR = {'after_update': 60, 'model_state_sha256':
          '4809a3cda62d96b53fb043cf131f648d91e577d6522cb70f4835799cc0070136'}
ARMS = {'attenuated_band_edge': 'dice_bands_edge',
        'group_band_edge_depth': 'dice_bands_edge_depth'}


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


def derive_specification(parent, triple, variant):
    if variant not in ARMS or parent['anchor_model_sha256'] != ANCHOR['model_state_sha256'] or triple['anchor_model_sha256'] != ANCHOR['model_state_sha256']:
        raise ValueError('Unexpected variant or native anchor')
    if [r['case_name'] for r in parent['cases']] != [r['case_name'] for r in triple['cases']]:
        raise ValueError('Reference training panels differ')
    pg = np.array([r['gradient_gram'] for r in parent['cases']])
    tg = np.array([r['gradient_gram'] for r in triple['cases']])
    if not np.allclose(pg, tg[:, :3, :3], rtol=1e-9, atol=0):
        raise ValueError('Reference shared gradients differ')
    if variant == 'attenuated_band_edge':
        weights = {n: triple['weights'][n] for n in ('bands', 'edge')}
        rows = parent['cases']
        rule = 'Retain the losing triple BAND/EDGE coefficients and omit DEPTH'
    else:
        rows = triple['cases']
        def candidate(scale):
            return dict(bands=scale*triple['weights']['bands'],
                        edge=parent['weights']['edge'], depth=scale*triple['weights']['depth'])
        low, high = 0., 1.
        # All reference cross-products are nonnegative; the envelope is monotone
        # in this common B+D multiplier. Fail closed if that assumption changes.
        if np.any(tg[:, 1:, 1:] < 0):
            raise ValueError('Group calibration requires nonnegative anchor cross-products')
        for _ in range(100):
            mid = (low+high)/2
            bounds = subset_envelope(rows, ARMS[variant], candidate(mid))
            if bounds['median'] <= .1 and bounds['p95'] <= .5:
                low = mid
            else:
                high = mid
        weights = candidate(low)
        rule = 'Preserve parent EDGE; scale B+D together to original joint/ramp envelope'
    bounds = subset_envelope(rows, ARMS[variant], weights)
    if bounds['median'] > .1+1e-12 or bounds['p95'] > .5+1e-12:
        raise ValueError('Proposed anchor envelope exceeded')
    return dict(variant=variant, objective=ARMS[variant], rule=rule, weights=weights,
                anchor_rows=deepcopy(rows), anchor_model_sha256=ANCHOR['model_state_sha256'],
                expected_envelope=bounds, coefficient_policy='Frozen training-anchor coefficients; no online retuning')


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


def make_protocol(reference, specification):
    p = deepcopy(reference)
    arm = specification['objective']
    if (p['active_objective'] != arm or p['budget_flops'] != CAP or
            p['seeds'] != [0] or p['folds'] != [0] or p['augmentation'] != 'none' or
            p['matched_readiness_anchor'] != ANCHOR):
        raise ValueError('Archived long protocol differs')
    p.update(name=NAME+'_'+specification['variant'], composition=specification,
             constraint_calibration='frozen_composition_anchor_v1',
             constraint_definition=specification['rule']+'; unchanged long native objective/schedule',
             endpoint='Final cap endpoint, f0s0 only, no validation selection')
    # Conservative separate calibration allowance for fixed-weight/envelope checks.
    schedule = p['models']['nnunet']['objectives'][arm]
    schedule['calibration_flops'] += 10000000
    schedule['composition_calibration_extra_flops'] = 10000000
    schedule['consumed_total_flops'] = schedule['consumed_training_flops'] + schedule['calibration_flops']
    return p
