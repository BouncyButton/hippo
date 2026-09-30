"""Synthetic edit-audit accounting tests."""

import numpy as np

from .audit_local_correction import edit_counts


def test_edit_accounting():
    target = np.array([[[1, 0, 2, 1]]])
    baseline = np.array([[[0, 0, 1, 1]]])
    proposal = np.array([[[1, 1, 0, 1]]])
    confidence = np.array([[[0.5, 0.7, 0.9, 0.9]]])
    result = edit_counts(baseline, proposal, target, confidence)
    assert result["changed"] == 3
    assert result["corrected"] == 1
    assert result["introduced"] == 1
    assert result["wrong_to_wrong"] == 1
    assert result["confidence_bins"]["lt_0.6"]["corrected"] == 1
    assert result["confidence_bins"]["0.6_to_0.8"]["introduced"] == 1
    assert result["confidence_bins"]["ge_0.8"]["wrong_to_wrong"] == 1
