"""Checks for the new multi-plane geometric proxies and ranking protocol."""
import numpy as np

from .audit_cue_consistency import choose, fit_scalar, return_features, scalar_stats, topology


def test_return_requires_anterior_bridge_and_checks_direction():
    union = np.zeros((8, 10, 12), bool)
    union[2:6, 3:8, 2:4] = True
    union[2:6, 3:8, 7:9] = True
    image = union.astype(float)
    assert return_features(union, image, 3, 2)['sagittal_bridge_count'] == 0
    union[2:6, 5, 4:7] = True
    result = return_features(union, image, 3, 2)
    assert result['sagittal_bridge_count'] == 4
    assert result['sagittal_anterior_only'] == 4
    assert result['sagittal_gap_darkness'] == 1
    assert return_features(union[::-1], image[::-1], 3, 2) == result
    union[2:6, 2, 2:9] = True
    assert return_features(union, image, 3, 2)['sagittal_anterior_only'] == 0


def test_thin_upper_lip_is_resolution_sensitive():
    union = np.zeros((8, 10, 12), bool)
    union[2:6, 3:8, 2:4] = True
    union[2:6, 3:8, 7] = True
    union[2:6, 5, 4:7] = True
    assert return_features(union, union.astype(float), 3, 1)['sagittal_bridge_count'] == 4
    assert return_features(union, union.astype(float), 3, 2)['sagittal_bridge_count'] == 0


def test_secondary_profile_disappearance_is_spatial():
    mask = np.zeros((10, 10), bool)
    mask[1:5, 1:5] = True
    mask[7:9, 7:9] = True
    assert topology(mask, mask)['secondary_disappears'] == 0
    previous = mask.copy()
    previous[7:9, 7:9] = False
    assert topology(mask, previous)['secondary_disappears'] == 1
    previous[8, 8] = True
    assert topology(mask, previous)['secondary_disappears'] == 0


def test_connected_bridge_is_not_a_component_split():
    mask = np.zeros((10, 10), bool)
    mask[2:6, 1:3] = True
    mask[2:6, 6:8] = True
    assert topology(mask, mask)['component_count_min2'] == 2
    mask[2, 3:6] = True
    assert topology(mask, mask)['component_count_min2'] == 1
    assert topology(mask, mask)['vertical_double_min2'] == 3


def test_scalar_direction_learned_from_training_and_constant_cue_abstains():
    cases = [dict(candidates=list(range(12)), target=6,
                  X=[[-float(y == 6)] for y in range(12)]) for _ in range(3)]
    direction, threshold = fit_scalar(cases, 0)
    assert direction == -1 and 0 <= threshold < 1
    for c in cases:
        c['X'] = [[0.] for _ in range(12)]
    direction, threshold = fit_scalar(cases, 0)
    assert direction == 1 and threshold == 0


def test_tie_break_does_not_take_a_target_argument():
    assert choose(np.array([0., 1., 1., 1., 0.]), np.arange(5)) == 2


def test_broad_regional_cue_is_not_precise_localisation():
    case = dict(candidates=list(range(15)), target=7,
                X=[[float(abs(y-7)<=3)] for y in range(15)])
    result = scalar_stats([case], 0, 1, .5)
    assert result['mean_within_case_auc'] == 1
    assert result['local_auc'] == .5
