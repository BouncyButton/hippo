"""Tests for predicted-foreground audit preprocessing."""

from __future__ import annotations

import numpy as np

from .audit_predicted_foreground import (
    crop_model_volume_to_native,
    largest_foreground_component,
)


def test_crop_model_volume_to_native_undoes_symmetric_padding() -> None:
    native = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    padded = np.pad(native, ((2, 3), (1, 1), (4, 4)))
    result = crop_model_volume_to_native(padded, native.shape)
    np.testing.assert_array_equal(result, native)


def test_largest_foreground_component_removes_small_island() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    mask[0, 0, 0] = True
    largest, count, removed = largest_foreground_component(mask)
    assert count == 2
    assert removed == 1
    assert largest.sum() == 64
