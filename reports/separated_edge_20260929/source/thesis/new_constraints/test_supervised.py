from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

from thesis.new_constraints.supervised import (
    CALIBRATION_STRATA,
    CalibrationDiagnostics,
    build_supervised_loss,
    calibration_diagnostics_config,
    supervised_loss_config,
)


def _sample() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.linspace(-2.0, 2.0, 2 * 3 * 3 * 4 * 5).reshape(2, 3, 3, 4, 5)
    labels = (torch.arange(2 * 3 * 4 * 5) % 3).reshape(2, 1, 3, 4, 5)
    return logits, labels


def _metric(row: dict, stratum: str, name: str) -> float | int | None:
    return row[f"calibration/{stratum}/{name}"]


def test_default_dice_preserves_exact_value_and_gradient() -> None:
    logits, labels = _sample()
    actual_logits = logits.clone().requires_grad_()
    legacy_logits = logits.clone().requires_grad_()
    loss = build_supervised_loss()
    assert type(loss) is DiceLoss
    actual = loss(actual_logits, labels)
    expected = DiceLoss(to_onehot_y=True, softmax=True)(legacy_logits, labels)
    assert torch.equal(actual, expected)
    actual_gradient, = torch.autograd.grad(actual, actual_logits)
    legacy_gradient, = torch.autograd.grad(expected, legacy_logits)
    assert torch.equal(actual_gradient, legacy_gradient)


@pytest.mark.parametrize("weight", [0.25, 1.0, 2.5])
@pytest.mark.parametrize("channel_target", [False, True])
def test_dice_ce_is_exact_weighted_multiclass_objective(weight: float, channel_target: bool) -> None:
    logits, labels = _sample()
    actual_logits = logits.clone().requires_grad_()
    reference_logits = logits.clone().requires_grad_()
    actual = build_supervised_loss("dice_ce", weight)(actual_logits, labels if channel_target else labels[:, 0])
    expected = DiceLoss(to_onehot_y=True, softmax=True)(reference_logits, labels)
    expected = expected + weight * F.cross_entropy(reference_logits.float(), labels[:, 0])
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual, actual_logits)[0],
        torch.autograd.grad(expected, reference_logits)[0],
    )


def test_ce_retains_gradient_for_confidently_wrong_voxels() -> None:
    logits = torch.tensor([80.0, -80.0, -80.0]).reshape(1, 3, 1, 1, 1).requires_grad_()
    labels = torch.ones((1, 1, 1, 1, 1), dtype=torch.long)
    dice = build_supervised_loss()(logits, labels)
    dice_gradient = torch.autograd.grad(dice, logits)[0]
    combined = build_supervised_loss("dice_ce", 0.75)(logits, labels)
    combined_gradient = torch.autograd.grad(combined, logits)[0]
    assert torch.isfinite(combined)
    assert dice_gradient.abs().max() < 1e-20
    assert torch.allclose(combined_gradient.flatten(), torch.tensor([0.75, -0.75, 0.0]))


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_dice_ce_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite, positive"):
        build_supervised_loss("dice_ce", weight)
    with pytest.raises(ValueError, match="finite, positive"):
        supervised_loss_config("dice_ce", weight)


def test_loss_identity_captures_defaults_and_effective_weight() -> None:
    legacy = supervised_loss_config()
    assert legacy == supervised_loss_config("dice", 7.0)
    assert legacy["dice"]["include_background"] is True
    assert legacy["dice"]["batch"] is False
    assert legacy["dice"]["squared_pred"] is False
    assert legacy["dice"]["smooth_nr"] == legacy["dice"]["smooth_dr"] == 1e-5
    assert legacy["dice"]["reduction"] == "mean"
    assert legacy["ce"] is None
    combined = supervised_loss_config("dice_ce", 0.5)
    assert combined["dice"] == legacy["dice"]
    assert combined["ce_weight"] == 0.5
    assert combined["ce"]["input"] == "full_multiclass_logits"
    json.dumps(combined, allow_nan=False)
    with pytest.raises(ValueError, match="Unknown supervised loss"):
        build_supervised_loss("ce")


@pytest.mark.parametrize("bad", ["nan", "inf", "fractional_label", "negative_label", "large_label", "onehot", "shape"])
def test_new_loss_and_diagnostics_reject_invalid_tensors(bad: str) -> None:
    logits, labels = _sample()
    if bad in {"nan", "inf"}:
        logits[0, 0, 0, 0, 0] = float(bad)
    elif bad == "fractional_label":
        labels = labels.float()
        labels[0, 0, 0, 0, 0] = 0.5
    elif bad == "negative_label":
        labels[0, 0, 0, 0, 0] = -1
    elif bad == "large_label":
        labels[0, 0, 0, 0, 0] = 3
    elif bad == "onehot":
        labels = F.one_hot(labels[:, 0], num_classes=3).movedim(-1, 1)
    else:
        labels = labels[:, :, :, :, :-1]
    with pytest.raises(ValueError):
        build_supervised_loss("dice_ce")(logits, labels)
    accumulator = CalibrationDiagnostics(3)
    before = accumulator.summary()
    with pytest.raises(ValueError):
        accumulator.update(logits, labels)
    assert accumulator.summary() == before


def test_multiclass_diagnostics_match_hand_computation_and_strata() -> None:
    # Two three-class voxels: one correct background, one confident mistake
    # between the two foreground classes. Grouped foreground Brier would miss it.
    probabilities = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    logits = probabilities.log().T.reshape(1, 3, 1, 1, 2).requires_grad_()
    labels = torch.tensor([0, 2]).reshape(1, 1, 1, 1, 2)
    before = logits.detach().clone()
    rng_before = torch.get_rng_state().clone()
    accumulator = CalibrationDiagnostics(3)
    rows = accumulator.update(logits, labels)
    row = rows[0]
    assert _metric(row, "whole", "voxel_count") == 2
    assert _metric(row, "whole", "nll") == pytest.approx((-math.log(0.8) - math.log(0.1)) / 2)
    assert _metric(row, "whole", "brier") == pytest.approx((0.06 + 1.46) / 2)
    entropy = -0.8 * math.log(0.8) - 0.2 * math.log(0.1)
    assert _metric(row, "whole", "entropy") == pytest.approx(entropy)
    assert _metric(row, "whole", "normalized_entropy") == pytest.approx(entropy / math.log(3))
    assert _metric(row, "whole", "ece") == pytest.approx(0.3)
    assert _metric(row, "whole", "accuracy") == pytest.approx(0.5)
    assert _metric(row, "whole", "saturation_fraction") == 0.0
    for stratum in ("gt_foreground", "foreground_union", "incorrect"):
        assert _metric(row, stratum, "voxel_count") == 1
        assert _metric(row, stratum, "nll") == pytest.approx(-math.log(0.1))
        assert _metric(row, stratum, "brier") == pytest.approx(1.46)
    assert _metric(row, "gt_boundary", "voxel_count") == 2
    assert _metric(row, "correct", "ece") == pytest.approx(0.2)
    assert _metric(row, "incorrect", "ece") == pytest.approx(0.8)
    assert logits.grad is None
    assert torch.equal(logits, before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert accumulator.summary()["calibration/case_count"] == 1
    json.dumps(accumulator.summary(), allow_nan=False)


def test_diagnostics_report_unclipped_nll_on_extreme_finite_logits() -> None:
    logits = torch.tensor([1000.0, -1000.0, -1000.0]).reshape(1, 3, 1, 1, 1)
    labels = torch.ones((1, 1, 1, 1, 1), dtype=torch.long)
    row = CalibrationDiagnostics(3).update(logits, labels)[0]
    assert _metric(row, "whole", "nll") == 2000.0
    assert _metric(row, "whole", "brier") == 2.0
    assert _metric(row, "whole", "entropy") == 0.0
    assert _metric(row, "whole", "saturation_fraction") == 1.0
    assert _metric(row, "whole", "ece") == 1.0


def test_ece_keeps_separate_bins_and_includes_confidence_one() -> None:
    probabilities = torch.tensor([[0.6, 0.4], [0.9, 0.1]])
    logits = probabilities.log().T.reshape(1, 2, 1, 1, 2)
    labels = torch.tensor([0, 1]).reshape(1, 1, 1, 1, 2)
    accumulator = CalibrationDiagnostics(2)
    row = accumulator.update(logits, labels)[0]
    # Different bins: (|0.6-1| + |0.9-0|) / 2 = 0.65;
    # |overall confidence-overall accuracy| = 0.25 is not ECE.
    assert _metric(row, "whole", "ece") == pytest.approx(0.65)
    perfect = torch.tensor([1000.0, -1000.0]).reshape(1, 2, 1, 1, 1)
    perfect_row = accumulator.update(perfect, torch.zeros((1, 1, 1, 1), dtype=torch.long))[0]
    assert _metric(perfect_row, "whole", "ece") == 0.0


def test_streaming_pools_bins_instead_of_averaging_case_ece() -> None:
    accumulator = CalibrationDiagnostics(2)
    logits = torch.tensor([0.8, 0.2]).log().reshape(1, 2, 1, 1, 1)
    first = accumulator.update(logits, torch.zeros((1, 1, 1, 1), dtype=torch.long))[0]
    second = accumulator.update(logits, torch.ones((1, 1, 1, 1), dtype=torch.long))[0]
    assert _metric(first, "whole", "ece") == pytest.approx(0.2)
    assert _metric(second, "whole", "ece") == pytest.approx(0.8)
    summary = accumulator.summary()
    assert summary["calibration/case_count"] == 2
    assert _metric(summary, "whole", "ece") == pytest.approx(0.3)
    batched = CalibrationDiagnostics(2)
    cases = batched.update(logits.repeat(2, 1, 1, 1, 1), torch.tensor([0, 1]).reshape(2, 1, 1, 1))
    assert len(cases) == 2
    assert summary == batched.summary()
    summary["calibration/case_count"] = 0
    assert accumulator.summary()["calibration/case_count"] == 2


def test_empty_strata_are_null_and_boundary_has_no_fov_padding() -> None:
    accumulator = CalibrationDiagnostics(2)
    for stratum in CALIBRATION_STRATA:
        assert _metric(accumulator.summary(), stratum, "nll") is None
    logits = torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1, 1)
    row = accumulator.update(logits, torch.zeros((1, 1, 1, 1), dtype=torch.long))[0]
    for stratum in ("gt_foreground", "foreground_union", "gt_boundary", "incorrect"):
        assert _metric(row, stratum, "voxel_count") == 0
        assert _metric(row, stratum, "ece") is None
        assert _metric(row, stratum, "nll") is None
    json.dumps(row, allow_nan=False)


def test_saturation_threshold_is_inclusive() -> None:
    logits = torch.tensor([0.75, 0.25]).log().reshape(1, 2, 1, 1, 1)
    labels = torch.zeros((1, 1, 1, 1), dtype=torch.long)
    row = CalibrationDiagnostics(2, saturation_threshold=0.75).update(logits, labels)[0]
    assert _metric(row, "whole", "confidence") == 0.75
    assert _metric(row, "whole", "saturation_fraction") == 1.0
    config = calibration_diagnostics_config()
    assert config["num_bins"] == 15
    assert config["saturation_threshold"] == 0.99
    assert config["saturation_comparison"] == ">="


def test_ce_and_diagnostics_use_float32_under_cpu_autocast() -> None:
    logits, labels = _sample()
    logits = logits.to(torch.bfloat16).requires_grad_()
    objective = build_supervised_loss("dice_ce")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual_loss = objective(logits, labels)
        actual = CalibrationDiagnostics(3).update(logits, labels)
    expected_dice = DiceLoss(to_onehot_y=True, softmax=True)(logits, labels)
    expected_ce = F.cross_entropy(logits.float(), labels[:, 0])
    assert actual_loss.dtype == torch.float32
    assert torch.equal(actual_loss, expected_dice + expected_ce)
    expected = CalibrationDiagnostics(3).update(logits.float(), labels)
    assert actual == expected
    actual_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
