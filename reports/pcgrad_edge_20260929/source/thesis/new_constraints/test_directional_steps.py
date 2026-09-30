"""Structural checks for the directional step-surface objective."""

import torch

from thesis.new_constraints.directional_steps import (
    DirectionalStepSurfaceLoss,
    directional_step_maps,
    terminal_distributions,
)


def _logits(foreground: torch.Tensor, strength: float = 8.0) -> torch.Tensor:
    logits = torch.full((1, 3, *foreground.shape), -strength)
    logits[:, 0] = torch.where(foreground, -strength, strength)
    logits[:, 1] = torch.where(foreground, strength, -strength)
    return logits


def test_terminal_distributions_locate_both_ends_of_a_ray() -> None:
    probability = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
    low, high = terminal_distributions(probability, axis=2)
    assert torch.equal(low, torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]]))
    assert torch.equal(high, torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]]))


def test_axis_steps_respond_at_every_voxel_between_boundary_and_ray_end() -> None:
    probability = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
    forward, backward = directional_step_maps(probability, axis=2)
    assert torch.equal(forward, torch.tensor([[[[0.0, 1.0, 1.0, 1.0]]]]))
    assert torch.equal(backward, torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]))


def test_boundary_displacement_costs_more_than_correct_steps() -> None:
    target = torch.zeros((5, 6, 12), dtype=torch.bool)
    for y in range(1, 5):
        target[1:4, y, 3:9-y] = True
    shifted = torch.zeros_like(target)
    shifted[:, :, 1:] = target[:, :, :-1]
    objective = DirectionalStepSurfaceLoss(axes=(0, 1, 2))
    labels = target.long().unsqueeze(0)
    correct = objective(_logits(target), labels)
    displaced = objective(_logits(shifted), labels)
    assert correct.loss.isfinite() and displaced.loss.isfinite()
    assert displaced.loss > correct.loss
    assert correct.details["longitudinal_order_loss"] < 1e-5


def test_gt_upward_exception_has_no_order_penalty_when_predicted_correctly() -> None:
    target = torch.zeros((3, 4, 10), dtype=torch.bool)
    target[1, 1, 2:5] = True
    target[1, 2, 2:8] = True  # Real upward step of three voxels.
    objective = DirectionalStepSurfaceLoss(axes=(2,), order_weight=0.25)
    result = objective(_logits(target), target.long().unsqueeze(0))
    assert result.details["longitudinal_order_loss"] < 1e-5


def test_spurious_outer_voxel_receives_removal_gradient() -> None:
    target = torch.zeros((3, 4, 10), dtype=torch.bool)
    target[1, 1:3, 2:7] = True
    predicted = target.clone()
    predicted[1, 1, 9] = True
    logits = _logits(predicted, strength=3.0).requires_grad_()
    objective = DirectionalStepSurfaceLoss(axes=(2,))
    result = objective(logits, target.long().unsqueeze(0))
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, 1, 1, 9] > 0
