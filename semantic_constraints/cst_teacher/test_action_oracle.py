"""Synthetic checks for the non-deployable foreground-bias oracle probe."""

import numpy as np

from .probe_action_oracle import biased_prediction, evaluate_case


def test_bias_preserves_ap_order_and_changes_foreground():
    probabilities = np.zeros((3, 2, 2), dtype=np.float32)
    probabilities[0] = 0.51
    probabilities[1] = 0.40
    probabilities[2] = 0.09
    assert np.all(biased_prediction(probabilities, 0.0) == 0)
    assert np.all(biased_prediction(probabilities, 0.5) == 1)
    assert np.all(biased_prediction(probabilities, -0.5) == 0)


def test_oracle_case_can_recover_missing_tissue():
    labels = np.ones((3, 3, 3), dtype=np.int8)
    probabilities = np.zeros((3, 3, 3, 3), dtype=np.float32)
    probabilities[0] = 0.51
    probabilities[1] = 0.49
    result = evaluate_case(labels, probabilities, np.arange(3), np.array([False, True, False]), np.array([0, 1, 0]))
    assert result["selected_slices"] == 1
    assert result["slices_with_beneficial_oracle_action"] == 1
    assert result["best_per_slice_oracle_dice_gain"] > 0
