"""Checks for section-level fitted contour supervision."""

import torch

from thesis.new_constraints.sagittal_step_fit import (
    SagittalStepContourLoss,
    fit_decreasing_steps,
)
from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective


def _logits(mask: torch.Tensor, strength: float = 6.0) -> torch.Tensor:
    logits = torch.full((1, 3, *mask.shape), -strength)
    logits[:, 0] = torch.where(mask, -strength, strength)
    logits[:, 1] = torch.where(mask, strength, -strength)
    return logits


def _staircase() -> torch.Tensor:
    mask = torch.zeros((3, 10, 16), dtype=torch.bool)
    for x in (1, 2):
        for y in range(1, 9):
            top = 12 if y < 4 else 9 if y < 6 else 6
            mask[x, y, top - 3:top + 1] = True
    return mask


def test_fit_pools_upward_violation_and_backpropagates_through_block_means() -> None:
    values = torch.tensor([[3.0, 1.0, 2.0, 0.0]], requires_grad=True)
    fitted = fit_decreasing_steps(values, torch.ones_like(values, dtype=torch.bool))
    assert torch.allclose(fitted, torch.tensor([[3.0, 1.5, 1.5, 0.0]]))
    fitted[0, 1].backward()
    assert torch.allclose(values.grad, torch.tensor([[0.0, 0.5, 0.5, 0.0]]))


def test_fit_keeps_disconnected_contour_segments_separate() -> None:
    values = torch.tensor([[3.0, 1.0, 0.0, 0.0, 2.0, 1.0]])
    valid = torch.tensor([[True, True, False, False, True, True]])
    fitted = fit_decreasing_steps(values, valid)
    assert torch.equal(fitted, torch.tensor([[3.0, 1.0, 0.0, 0.0, 2.0, 1.0]]))


def test_projected_curve_can_hide_a_wrong_local_step() -> None:
    annotated = torch.tensor([[3.0, 2.0, 2.0, 1.0]])
    predicted = torch.tensor([[3.0, 1.0, 3.0, 1.0]])
    valid = torch.ones_like(annotated, dtype=torch.bool)
    annotated_fit = fit_decreasing_steps(annotated, valid)
    predicted_fit = fit_decreasing_steps(predicted, valid)
    assert torch.equal(predicted_fit, annotated_fit)
    predicted_residual = predicted - predicted_fit
    annotated_residual = annotated - annotated_fit
    assert (predicted_residual - annotated_residual).abs().sum() > 0


def test_wrong_plateau_and_transition_cost_more_than_correct_section() -> None:
    target = _staircase()
    shifted = torch.zeros_like(target)
    shifted[:, 1:] = target[:, :-1]
    objective = SagittalStepContourLoss()
    labels = target.long().unsqueeze(0)
    correct = objective(_logits(target), labels)
    wrong = objective(_logits(shifted), labels)
    assert wrong.loss > correct.loss
    assert wrong.details["height_loss"] > correct.details["height_loss"]
    assert wrong.details["transition_loss"] > correct.details["transition_loss"]
    assert int(correct.details["sections_fitted"][0]) == 2


def test_displaced_contour_sends_finite_gradient_to_prediction() -> None:
    target = _staircase()
    shifted = torch.zeros_like(target)
    shifted[..., 1:] = target[..., :-1]
    logits = _logits(shifted, strength=3.0).requires_grad_()
    result = SagittalStepContourLoss()(logits, target.long().unsqueeze(0))
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_identical_curves_with_real_upward_step_are_not_penalized_as_mismatch() -> None:
    target = _staircase()
    target[1, 5, 9:13] = True
    result = SagittalStepContourLoss()( _logits(target), target.long().unsqueeze(0))
    assert result.details["height_loss"].item() < 1e-3
    assert result.details["transition_loss"].item() < 1e-3


def test_sagittal_step_is_available_through_training_objective() -> None:
    target = _staircase()
    logits = _logits(target).requires_grad_()
    objective = NewConstraintObjective(NewConstraintConfig(
        equivariance_weight=0.0,
        sagittal_step_weight=0.5,
    ))
    output = objective(None, torch.zeros((1, 1, *target.shape)), logits, target.long().unsqueeze(0))
    assert "sagittal_step" in output["results"]
    assert torch.allclose(output["loss"], 0.5 * output["results"]["sagittal_step"].loss)
