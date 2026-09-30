"""Synthetic unit tests for the operational coronal error taxonomy."""

import numpy as np
import pytest

from .error_types import audit_case, classify_counts, count_slice_errors


def test_count_and_classify():
    reference = np.array([[1, 1, 0], [2, 2, 0]])
    prediction = np.array([[0, 2, 1], [2, 0, 0]])
    assert count_slice_errors(reference, prediction).tolist() == [2, 1, 1]
    counts = np.array([[0, 0, 0], [6, 1, 0], [1, 8, 0], [0, 0, 7], [4, 4, 1]])
    assert classify_counts(counts).tolist() == [0, 1, 2, 3, 4]


def test_disappearance_and_terminal_extension():
    reference = np.zeros((3, 5, 3), dtype=np.int8)
    prediction = np.zeros_like(reference)
    reference[:, 1:4, :] = 1
    prediction[:, 1, :] = 1
    prediction[:, 3:5, :] = 1
    result = audit_case(reference, prediction, np.arange(5))
    assert result["isolated_disappearance"].tolist() == [False, False, True, False, False]
    assert result["terminal_extension"].tolist() == [False, False, False, False, True]


def test_bad_inputs():
    with pytest.raises(ValueError):
        count_slice_errors(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        classify_counts(np.ones((3, 4)))
    with pytest.raises(ValueError):
        audit_case(np.zeros((3, 5, 3)), np.zeros((3, 5, 3)), np.array([2, 1]))
