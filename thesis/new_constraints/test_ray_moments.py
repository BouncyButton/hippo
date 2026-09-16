"""Tests for orthogonal ray-moment constraints."""

import torch

from .objective import NewConstraintConfig, NewConstraintObjective
from .ray_moments import RayMomentLoss, ray_moment_map


def test_zero_order_never_allocates_unused_coordinates(monkeypatch):
    values = torch.randn(2, 3, 4, 5, 6, requires_grad=True)

    def reject_coordinates(*args, **kwargs):
        raise AssertionError("Zeroth moments do not need a coordinate grid.")

    monkeypatch.setattr(torch, "linspace", reject_coordinates)
    for axis in (0, 1, 2):
        actual = ray_moment_map(values, axis, 0)
        expected = values.sum(dim=axis + 2) / float(values.shape[axis + 2])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _labels() -> torch.Tensor:
    labels = torch.zeros((1, 1, 12, 12, 12), dtype=torch.long)
    labels[:, :, 2:7, 3:9, 3:9] = 1
    labels[:, :, 7:10, 4:8, 4:8] = 2
    return labels


def _logits(labels: torch.Tensor, magnitude: float = 8.0) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(labels[:, 0], num_classes=3).movedim(-1, 1).float()
    return magnitude * (2.0 * one_hot - 1.0)


def test_matching_confident_segmentation_has_small_loss() -> None:
    labels = _labels()
    for orders in ((0,), (0, 1)):
        result = RayMomentLoss(orders=orders)(_logits(labels), labels)
        assert result.loss.ndim == 0
        assert float(result.loss) < 1e-8


def test_equal_mass_shift_is_invisible_to_m0_but_visible_to_m1() -> None:
    first = torch.zeros((1, 1, 7, 5, 5))
    second = torch.zeros_like(first)
    first[:, :, 1, 2, 2] = 1.0
    second[:, :, 5, 2, 2] = 1.0
    torch.testing.assert_close(ray_moment_map(first, 0, 0), ray_moment_map(second, 0, 0))
    assert float((ray_moment_map(first, 0, 1) - ray_moment_map(second, 0, 1)).abs().sum()) > 0


def test_foreground_only_is_blind_to_swap_but_semantic_groups_are_not() -> None:
    labels = _labels()
    swapped = labels.clone()
    swapped[labels == 1] = 2
    swapped[labels == 2] = 1
    logits = _logits(swapped)
    foreground = RayMomentLoss(
        orders=(0, 1), class_groups=((1, 2),), group_names=("foreground",)
    )(logits, labels)
    semantic = RayMomentLoss(
        orders=(0, 1), class_groups=((1,), (2,)), group_names=("anterior", "posterior")
    )(logits, labels)
    assert float(foreground.loss) < 1e-8
    assert float(semantic.loss) > 0.0


def test_gradient_is_finite_and_nonzero_for_boundary_error() -> None:
    labels = _labels()
    logits = _logits(labels).clone()
    logits[:, 0, 4, 3, 5] = 4.0
    logits[:, 1, 4, 3, 5] = -4.0
    logits.requires_grad_()
    loss = RayMomentLoss(orders=(0, 1))(logits, labels).loss
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_objective_applies_exact_m0_weight() -> None:
    labels = _labels()
    logits = _logits(labels).clone()
    logits[:, 0, 4, 3, 5] = 4.0
    logits[:, 1, 4, 3, 5] = -4.0
    direct = RayMomentLoss(orders=(0,))(logits, labels).loss
    objective = NewConstraintObjective(
        NewConstraintConfig(equivariance_weight=0.0, ray_moment_weight=0.4)
    )
    output = objective(torch.nn.Identity(), logits, logits, labels)
    assert set(output["results"]) == {"ray_moment"}
    torch.testing.assert_close(output["loss"], 0.4 * direct)


def test_details_keep_axis_group_and_order_dimensions() -> None:
    labels = _labels()
    result = RayMomentLoss(orders=(0, 1))(_logits(labels), labels)
    assert result.details["component_loss"].shape == (1, 3, 3, 2)
    assert result.details["axis_loss"].shape == (1, 3)
    assert result.details["group_loss"].shape == (1, 3)
    assert result.details["order_loss"].shape == (1, 2)
