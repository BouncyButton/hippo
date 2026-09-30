import numpy as np
import pytest

from .audit_boundary_utility import partition_foreground, score_boundary


def test_ras_cut_preserves_support_and_includes_landmark_in_anterior():
    labels = np.ones((3, 8, 4), dtype=np.uint8)
    labels[0, :, 0] = 0
    pred = partition_foreground(labels, 4)
    assert np.array_equal(pred > 0, labels > 0)
    assert np.all(pred[1, :4] == 2)
    assert np.all(pred[1, 4:] == 1)


def test_fixed_band_measures_swaps_separately_from_foreground_misses():
    truth = partition_foreground(np.ones((2, 8, 2)), 4)
    shifted = partition_foreground(truth, 5)
    score = score_boundary(shifted, truth, 4, 1.0, 2.0)
    assert score["band_ap_swap_rate"] == .25
    assert score["union_dice"] == 1.0
    shifted[:, 4] = 0
    score = score_boundary(shifted, truth, 4, 1.0, 2.0)
    assert score["band_ap_swap_rate"] == 0.0
    assert score["band_gt_error_rate"] == .25
    assert score["band_mean_ap_dice"] < 1


def test_padding_does_not_change_metrics():
    truth = partition_foreground(np.ones((2, 8, 2)), 4)
    pred = partition_foreground(truth, 5)
    a = score_boundary(pred, truth, 4, 1., 2.)
    b = score_boundary(np.pad(pred, 3), np.pad(truth, 3), 7, 1., 2.)
    assert a == b


def test_band_uses_physical_spacing():
    truth = partition_foreground(np.ones((2, 8, 2)), 4)
    pred = partition_foreground(truth, 5)
    assert score_boundary(pred, truth, 4, 2., 2.)["band_ap_swap_rate"] == .5
    with pytest.raises(ValueError):
        partition_foreground(truth, 0)
