"""Focused tests for the translation-equivariance closure analysis."""

from __future__ import annotations

import numpy as np
import torch

from thesis.new_constraints.equivariance.closure_metrics import (
    ValidityAwareViewAccumulator,
    aggregate_patient_shifts,
    classwise_segmentation_dice,
    describe_shift,
    paired_summary,
    signed_axis_shifts,
    validity_aware_tta,
)
from thesis.new_constraints.equivariance.evaluate_closure import (
    require_labels_for_cohort,
)
from thesis.new_constraints.equivariance.translation_equivariance import (
    restore_translation,
    translate_3d,
    translation_valid_mask,
)


def test_translation_round_trip_and_valid_mask_for_magnitudes_one_to_four() -> None:
    tensor = torch.arange(9 * 10 * 11).reshape(1, 1, 9, 10, 11).float()

    for shift in signed_axis_shifts((1, 2, 3, 4)):
        restored = restore_translation(translate_3d(tensor, shift), shift)
        valid = translation_valid_mask(tensor.shape[-3:], shift)
        axis, sign, magnitude = describe_shift(shift)

        assert axis in {"x", "y", "z"}
        assert sign in {-1, 1}
        assert magnitude in {1, 2, 3, 4}
        assert torch.equal(restored * valid, tensor * valid)
        expected = 1.0
        for length, amount in zip(tensor.shape[-3:], shift, strict=True):
            expected *= (length - abs(amount)) / length
        assert torch.isclose(valid.mean(), torch.tensor(expected))


def test_validity_aware_tta_uses_fewer_views_at_invalid_borders() -> None:
    base = torch.zeros(1, 3, 2, 2, 3)
    base[:, 0] = 1.0
    shifted = torch.zeros_like(base)
    shifted[:, 1] = 1.0
    valid = torch.ones(1, 1, 2, 2, 3)
    valid[..., 0] = 0.0

    mean, disagreement, count = validity_aware_tta(base, [shifted], [valid])

    assert torch.equal(count[..., 0], torch.ones_like(count[..., 0]))
    assert torch.equal(count[..., 1:], torch.full_like(count[..., 1:], 2.0))
    assert torch.equal(mean[:, 0, ..., 0], torch.ones_like(mean[:, 0, ..., 0]))
    assert torch.equal(mean[:, 1, ..., 0], torch.zeros_like(mean[:, 1, ..., 0]))
    assert torch.allclose(mean[:, :2, ..., 1:], torch.full_like(mean[:, :2, ..., 1:], 0.5))
    assert torch.equal(disagreement[..., 0], torch.zeros_like(disagreement[..., 0]))
    assert torch.all(disagreement[..., 1:] > 0)


def test_tta_probabilities_remain_normalized() -> None:
    torch.manual_seed(11)
    base = torch.softmax(torch.randn(1, 3, 4, 5, 6), dim=1)
    views = [torch.softmax(torch.randn_like(base), dim=1) for _ in range(2)]
    masks = [
        translation_valid_mask(base.shape[-3:], (1, 0, 0)),
        translation_valid_mask(base.shape[-3:], (0, -2, 0)),
    ]

    mean, _, _ = validity_aware_tta(base, views, masks)

    assert torch.allclose(mean.sum(dim=1), torch.ones_like(mean[:, 0]), atol=1e-6)


def test_disagreement_is_zero_for_identical_valid_views() -> None:
    probabilities = torch.softmax(torch.randn(1, 3, 4, 4, 4), dim=1)
    valid = torch.ones(1, 1, 4, 4, 4)

    _, disagreement, _ = validity_aware_tta(
        probabilities,
        [probabilities.clone(), probabilities.clone()],
        [valid, valid],
    )

    assert torch.allclose(disagreement, torch.zeros_like(disagreement), atol=1e-7)


def test_disagreement_is_positive_when_one_view_differs() -> None:
    base = torch.zeros(1, 3, 2, 2, 2)
    base[:, 0] = 1.0
    different = torch.zeros_like(base)
    different[:, 2] = 1.0
    accumulator = ValidityAwareViewAccumulator.from_base(base)
    accumulator.add(different, torch.ones(1, 1, 2, 2, 2))

    _, disagreement, _ = accumulator.finalize()

    assert torch.all(disagreement > 0)


def test_base_and_shift_dice_use_the_exact_same_mask() -> None:
    labels = torch.ones(1, 1, 2, 2, 4, dtype=torch.long)
    base = torch.zeros(1, 3, 2, 2, 4)
    shifted = torch.zeros_like(base)
    base[:, 1] = 1.0
    shifted[:, 1] = 1.0
    # The predictions differ only on a voxel outside the shared valid region.
    base[:, :, :, :, 0] = 0.0
    base[:, 0, :, :, 0] = 1.0
    valid = torch.ones(1, 1, 2, 2, 4)
    valid[..., 0] = 0.0

    base_dice = classwise_segmentation_dice(
        base, labels, valid_mask=valid, class_ids=(1,), hard=True
    )
    shift_dice = classwise_segmentation_dice(
        shifted, labels, valid_mask=valid, class_ids=(1,), hard=True
    )

    assert torch.equal(base_dice, shift_dice)
    assert base_dice.item() == 1.0


def test_patient_aggregation_uses_worst_shift_not_worst_row_globally() -> None:
    rows = [
        {
            "model": "none",
            "patient_id": "a",
            "pure_equivariance_mean": 0.9,
            "dice_delta_mean": -0.1,
        },
        {
            "model": "none",
            "patient_id": "a",
            "pure_equivariance_mean": 0.7,
            "dice_delta_mean": 0.05,
        },
        {
            "model": "none",
            "patient_id": "b",
            "pure_equivariance_mean": 0.8,
            "dice_delta_mean": -0.2,
        },
    ]

    aggregated = aggregate_patient_shifts(rows)
    patient_a = next(row for row in aggregated if row["patient_id"] == "a")
    patient_b = next(row for row in aggregated if row["patient_id"] == "b")

    assert patient_a["shift_count"] == 2
    assert patient_a["pure_equivariance_mean"] == 0.8
    assert patient_a["pure_equivariance_worst"] == 0.7
    assert patient_a["dice_delta_worst"] == -0.1
    assert patient_b["dice_delta_worst"] == -0.2


def test_paired_summary_bootstraps_patient_differences() -> None:
    summary = paired_summary(
        [0.1, 0.2, 0.3],
        [0.2, 0.2, 0.1],
        bootstrap_samples=200,
        rng=np.random.default_rng(5),
    )

    assert summary["patients"] == 3
    assert summary["improved"] == 1
    assert summary["tied"] == 1
    assert summary["worsened"] == 1
    assert len(summary["paired_bootstrap_95pct_ci"]) == 2


def test_unlabelled_cohort_rejects_label_required_analysis() -> None:
    try:
        require_labels_for_cohort("official-test", labels_required=True)
    except ValueError as error:
        assert "no locally available labels" in str(error)
    else:
        raise AssertionError("Expected the official test cohort to reject label metrics.")


def test_labelled_fold_accepts_label_required_analysis() -> None:
    assert require_labels_for_cohort("fold-val", labels_required=True) is True
