"""Unit tests for label separation and foldedness descriptor invariants."""

from __future__ import annotations

import numpy as np

from .foldedness import best_fit_first_anterior_slice, extract_slice_features


def test_best_fit_cut_returns_first_anterior_slice() -> None:
    labels = np.zeros((5, 7, 5), dtype=np.uint8)
    labels[1:4, 1:3, 1:4] = 2
    labels[1:4, 3:6, 1:4] = 1
    cut, cost, gap = best_fit_first_anterior_slice(labels)
    assert cut == 3
    assert cost == 0
    assert gap > 0


def test_best_fit_cut_tolerates_one_nonplanar_voxel() -> None:
    labels = np.zeros((5, 7, 5), dtype=np.uint8)
    labels[1:4, 1:3, 1:4] = 2
    labels[1:4, 3:6, 1:4] = 1
    labels[1, 3, 1] = 2
    cut, cost, _ = best_fit_first_anterior_slice(labels)
    assert cut == 3
    assert cost == 1


def test_slice_features_are_translation_invariant() -> None:
    small = np.zeros((12, 11), dtype=bool)
    small[2:8, 3:6] = True
    small[5:9, 6:8] = True
    large = np.zeros((30, 29), dtype=bool)
    large[10:20, 12:21] = small[1:11, 1:10]
    first = extract_slice_features(small)
    second = extract_slice_features(large)
    assert first.keys() == second.keys()
    np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-8)


def test_height_asymmetry_detects_superior_extension() -> None:
    symmetric = np.zeros((10, 10), dtype=bool)
    symmetric[2:8, 2:6] = True
    folded = symmetric.copy()
    folded[2:5, 6:9] = True
    assert (
        extract_slice_features(folded)["top_asymmetry_abs"]
        > extract_slice_features(symmetric)["top_asymmetry_abs"]
    )
