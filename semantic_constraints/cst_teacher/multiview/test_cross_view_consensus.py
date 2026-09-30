"""Tests for label-free cross-view cut summaries."""

import numpy as np

from .audit_cross_view_consensus import soft_global_cut, trace_cut_candidates


def test_trace_midpoints_and_soft_cut():
    prediction = np.zeros((3, 8, 3), dtype=np.int8)
    prediction[:, 1:4] = 2
    prediction[:, 4:7] = 1
    assert np.all(trace_cut_candidates(prediction) == 3)
    probabilities = np.eye(3, dtype=np.float32)[prediction].transpose(3, 0, 1, 2)
    assert soft_global_cut(probabilities) == 3
    assert soft_global_cut(probabilities, 0.8) == 3
