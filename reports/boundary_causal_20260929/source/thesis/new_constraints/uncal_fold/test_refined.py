"""Tests for mirror invariance and localized transition measurements."""

from __future__ import annotations

import numpy as np

from .refined import extract_refined_slice_features, extract_transition_features


def _folded_mask() -> np.ndarray:
    mask = np.zeros((15, 13), dtype=bool)
    mask[2:7, 3:10] = True
    mask[7:13, 3:7] = True
    mask[8:11, 8:11] = True
    return mask


def test_refined_slice_features_are_mirror_invariant() -> None:
    mask = _folded_mask()
    first = extract_refined_slice_features(mask)
    second = extract_refined_slice_features(mask[::-1])
    np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-8)


def test_superior_notch_detects_two_lobes() -> None:
    folded = _folded_mask()
    flat = np.zeros_like(folded)
    flat[2:13, 3:8] = True
    assert (
        extract_refined_slice_features(folded)["superior_notch_depth"]
        > extract_refined_slice_features(flat)["superior_notch_depth"]
    )


def test_transition_detects_new_superior_tissue() -> None:
    previous = np.zeros((15, 13), dtype=bool)
    previous[3:12, 2:7] = True
    current = previous.copy()
    current[3:7, 8:11] = True
    features = extract_transition_features(previous, current)
    assert features["relative_area_change"] > 0
    assert features["novel_superior_fraction"] > 0
    assert features["superior_boundary_rise_max"] > 0
