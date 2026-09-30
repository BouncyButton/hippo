"""Analytical and exhaustive checks for the partition feasibility audit."""

import itertools

import numpy as np
import pytest

from evaluation.audit_ap_partition_topology import (
    compare_repair,
    partition_stats,
    plane_projection,
    ray_audit,
    single_switch,
    validate_labels,
)


def test_single_switch_matches_exhaustive_oracle():
    for n in range(1, 8):
        candidates = {
            tuple([left] * cut + [3 - left] * (n - cut))
            for left in (1, 2) for cut in range(n + 1)
        }
        for sequence in itertools.product((1, 2), repeat=n):
            costs = [sum(a != b for a, b in zip(sequence, candidate)) for candidate in candidates]
            fitted, cost, ties = single_switch(np.array(sequence))
            assert cost == min(costs)
            assert ties == costs.count(min(costs))
            assert np.count_nonzero(fitted != sequence) == cost
            assert np.count_nonzero(np.diff(fitted)) <= 1
            # Swapping semantic names must not alter the structural distance.
            assert single_switch(3 - np.array(sequence))[1] == cost


def test_background_splits_runs_and_pure_runs_are_valid():
    labels = np.array([1, 2, 0, 1, 2, 0, 2, 2], dtype=np.uint8)[None, :, None]
    stats, repaired = ray_audit(labels)
    assert stats["foreground_runs"] == 3
    assert stats["repeated_transition_runs"] == 0
    np.testing.assert_array_equal(repaired, labels)


def test_repeated_transition_repair_preserves_union_all_axes():
    labels = np.array([0, 1, 1, 2, 1, 1, 0], dtype=np.uint8)[None, :, None]
    for axis in range(3):
        rotated = np.moveaxis(labels, 1, axis)
        stats, repaired = ray_audit(rotated, axis)
        assert stats["repeated_transition_runs"] == 1
        assert stats["repair_voxels"] == 1
        np.testing.assert_array_equal(repaired > 0, rotated > 0)


def test_clean_shifted_boundary_is_invisible_to_structure():
    truth = np.zeros((7, 10, 7), dtype=np.uint8)
    truth[1:6, 1:5, 1:6] = 2
    truth[1:6, 5:9, 1:6] = 1
    prediction = truth.copy()
    prediction[1:6, 5:7, 1:6] = 2
    assert partition_stats(truth)["passes_connectivity_26"]
    assert partition_stats(prediction)["passes_connectivity_26"]
    rays, repaired = ray_audit(prediction)
    assert rays["repair_voxels"] == 0
    projected, fit = plane_projection(prediction)
    assert fit["disagreement"] == 0
    assert fit["cut"] == 7
    assert compare_repair(prediction, repaired, truth)["ap_swaps"] == 50
    np.testing.assert_array_equal(projected, prediction)
    oracle, _ = plane_projection(truth, prediction > 0)
    assert compare_repair(prediction, oracle, truth)["ap_swaps"] == 0


def test_component_connectivity_and_empty_contact_are_explicit():
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[0, 0, 0] = labels[1, 1, 1] = 1
    stats = partition_stats(labels)
    assert stats["a_components_6"] == 2
    assert stats["a_components_26"] == 1
    assert stats["contact_band_components_26"] == 0
    assert not stats["passes_connectivity_26"]
    _, fit = plane_projection(labels)
    assert not fit["valid"]


def test_plane_projection_matches_brute_force():
    rng = np.random.default_rng(45)
    for _ in range(30):
        labels = rng.integers(0, 3, size=(3, 6, 4), dtype=np.uint8)
        projected, fit = plane_projection(labels)
        occupied = np.flatnonzero((labels > 0).any(axis=(0, 2)))
        costs = []
        for left in (1, 2):
            for cut in range(occupied[0] + 1, occupied[-1] + 1):
                candidate = np.where(np.arange(6)[None, :, None] < cut, left, 3 - left)
                costs.append(np.count_nonzero((labels > 0) & (candidate != labels)))
        assert fit["disagreement"] == min(costs)
        assert np.count_nonzero(projected != labels) == min(costs)
        assert fit["ties"] == costs.count(min(costs))


def test_repairs_report_harm_and_reject_union_changes():
    truth = np.array([1, 2, 2], dtype=np.uint8)[None, :, None]
    prediction = np.array([1, 1, 2], dtype=np.uint8)[None, :, None]
    repaired = np.array([2, 2, 2], dtype=np.uint8)[None, :, None]
    result = compare_repair(prediction, repaired, truth)
    assert result["corrected"] == 1
    assert result["introduced"] == 1
    repaired[0, 0, 0] = 0
    with pytest.raises(ValueError, match="preserve foreground"):
        compare_repair(prediction, repaired, truth)


@pytest.mark.parametrize("labels", [np.zeros((2, 2)), np.full((2, 2, 2), 1.5), np.full((2, 2, 2), np.nan)])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        validate_labels(labels)
