"""Geometry and evaluation checks for the shoulder audit."""
import numpy as np
import pytest

from .audit_concavity_shoulder import (
    extract_shape, profile_shoulders, smooth_profile, error_metrics, paired,
)
from .foldedness import best_fit_first_anterior_slice


def toy_mask():
    mask = np.zeros((9, 44, 42), dtype=bool)
    # Exact steep-to-flat bend, with the first flat voxel at y=22.
    for y in range(4, 40):
        top = 34 - min(y - 4, 18)
        mask[2:7, y, 2:top + 1] = True
    return mask


def test_linear_slope_does_not_create_shoulder():
    p = 70.0 - np.arange(50) * 0.7
    scores = profile_shoulders(smooth_profile(p), np.arange(8, 42))
    assert np.max(scores) < 1e-7


def test_detects_steep_to_flat_bend():
    s = extract_shape(toy_mask())
    peak = s.candidates[np.argmax(s.consensus)]
    assert abs(peak - 22) <= 1
    assert s.consensus.max() > 0.5
    assert s.persistence[np.argmax(s.consensus)] == 1


def test_convex_bend_is_not_concavity():
    y = np.arange(60)
    p = 50 - np.maximum(y - 30, 0).astype(float)
    assert np.max(profile_shoulders(smooth_profile(p), np.arange(10, 50))) < 1e-7


def test_padding_and_native_coordinates_are_invariant():
    a = extract_shape(toy_mask())
    b = extract_shape(np.pad(toy_mask(), ((3, 4), (7, 9), (5, 2))), origin_y=-7)
    np.testing.assert_array_equal(a.candidates, b.candidates)
    np.testing.assert_allclose(a.consensus, b.consensus, atol=1e-12)
    assert (a.low, a.high) == (b.low, b.high)


def test_left_right_mirror_does_not_change_scores():
    a, b = extract_shape(toy_mask()), extract_shape(toy_mask()[::-1])
    np.testing.assert_allclose(a.consensus, b.consensus)


def test_gaps_are_not_interpolated():
    p = np.arange(30, dtype=float)
    p[15] = np.nan
    smoothed = smooth_profile(p)
    assert np.isnan(smoothed[15])
    assert np.isnan(profile_shoulders(smoothed, np.array([15, 16]))).all()


def test_empty_short_and_nonbinary_inputs_rejected():
    with pytest.raises(ValueError):
        extract_shape(np.zeros((10, 10, 10), dtype=bool))
    with pytest.raises(ValueError):
        extract_shape(np.ones((4, 5, 5), dtype=bool))
    with pytest.raises(ValueError):
        extract_shape(toy_mask().astype(int))


def test_target_change_cannot_change_shape_features():
    m = toy_mask()
    a, b = m.astype(np.uint8), m.astype(np.uint8)
    a[:, :22][m[:, :22]] = 2
    b[:, :28][m[:, :28]] = 2
    assert best_fit_first_anterior_slice(a)[:2] == (22, 0)
    assert best_fit_first_anterior_slice(b)[:2] == (28, 0)
    np.testing.assert_allclose(extract_shape(a > 0).consensus, extract_shape(b > 0).consensus)


def test_metrics_and_paired_direction():
    assert error_metrics([2, 5, 8], [2, 4, 5])["mae_mm"] == pytest.approx(4 / 3)
    p = paired([2, 5, 8], [2, 4, 5], [2, 4, 5])
    assert (p["helped"], p["equal"], p["harmed"]) == (2, 1, 0)
    assert p["second_minus_first_mae_mm"] < 0


def test_broad_hinge_finds_known_elbow_and_rejects_straight_line():
    from .audit_broad_shoulder import broad_score
    y = np.arange(60)
    z = 70 - y + 0.8 * np.maximum(y - 29.5, 0)
    candidates = np.arange(10, 50)
    scores = broad_score(y, z, candidates)
    assert candidates[np.argmax(scores)] == 30
    assert np.max(scores) == pytest.approx(1.0)
    assert np.all(broad_score(y, 70-y, candidates) == 0)
