import numpy as np
import torch

from scripts.audit_degree_boundary import supports, error_states, transitions, score_case
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands


def test_audit_support_exactly_matches_training_bands():
    truth = np.zeros((13, 13, 13), dtype=np.uint8)
    truth[3:10, 3:10, 3:10] = 1
    truth[7:10, 3:10, 3:10] = 2
    truth[1:3, 6, 6] = 1
    masks = supports(truth)
    inner, outer = build_boundary_bands(torch.from_numpy(truth > 0)[None, None])
    assert np.array_equal(masks['inner'], inner[0, 0].numpy())
    assert np.array_equal(masks['outer'], outer[0, 0].numpy())
    assert not (masks['inner'] & masks['outer']).any()
    assert np.array_equal(masks['inner_layer1'] | masks['inner_layer2'], masks['inner'])
    assert np.array_equal(masks['outer_layer1'] | masks['outer_layer2'], masks['outer'])


def test_recovered_foreground_with_wrong_ap_label_is_not_class_correction():
    truth = np.array([1, 1, 0, 2, 2, 0], dtype=np.uint8)
    old = np.array([0, 1, 1, 1, 0, 0], dtype=np.uint8)
    new = np.array([2, 0, 0, 2, 2, 1], dtype=np.uint8)
    a, b = error_states(truth, old), error_states(truth, new)
    assert a.tolist() == [1, 0, 2, 3, 1, 0]
    result = transitions(a, b, np.ones_like(truth, dtype=bool))
    assert result['union_corrected'] == 3 and result['union_introduced'] == 2
    assert result['class_corrected'] == 3 and result['class_introduced'] == 2
    assert result['state_transition_matrix'][1][3] == 1
    assert result['union_net_corrected'] == int(((a == 1) | (a == 2)).sum() - ((b == 1) | (b == 2)).sum())


def test_perfect_segmentation_and_one_inner_miss_have_expected_metrics():
    truth = np.zeros((9, 9, 9), dtype=np.uint8)
    truth[2:7, 2:7, 2:7] = 1
    truth[5:7, 2:7, 2:7] = 2
    masks = supports(truth)
    perfect, _ = score_case(truth, truth, masks)
    assert perfect['metrics']['macro_dice'] == 1
    assert perfect['metrics']['whole/assd_mm'] == 0
    assert perfect['metrics']['balanced_boundary_error'] == 0
    pred = truth.copy()
    pred[2, 3, 3] = 0
    row, _ = score_case(truth, pred, masks)
    assert row['regions']['boundary']['fn'] == 1
    assert row['regions']['beyond_bands']['union_errors'] == 0
    assert row['metrics']['balanced_boundary_error'] == .5 / masks['inner'].sum()
