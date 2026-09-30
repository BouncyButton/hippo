"""Tests for the multi-axis perimeter-profile diagnostic candidate."""

import torch

from .objective import NewConstraintConfig, NewConstraintObjective
from .perimeter_profile import PerimeterProfileLoss, perimeter_profile


def _labels() -> torch.Tensor:
    labels = torch.zeros((1, 1, 12, 12, 12), dtype=torch.long)
    labels[:, :, 2:7, 3:9, 3:9] = 1
    labels[:, :, 7:10, 4:8, 4:8] = 2
    return labels


def _logits(labels: torch.Tensor, magnitude: float = 8.0) -> torch.Tensor:
    classes = torch.nn.functional.one_hot(labels[:, 0], num_classes=3).movedim(-1, 1).float()
    return magnitude * (2.0 * classes - 1.0)


def test_matching_confident_segmentation_has_small_loss() -> None:
    labels = _labels()
    result = PerimeterProfileLoss()(_logits(labels), labels)
    assert result.loss.ndim == 0
    assert float(result.loss) < 1e-8


def test_gradient_is_finite_and_nonzero_for_boundary_error() -> None:
    labels = _labels()
    logits = _logits(labels).clone()
    logits[:, 0, 4, 3, 5] = 4.0
    logits[:, 1, 4, 3, 5] = -4.0
    logits.requires_grad_()
    loss = PerimeterProfileLoss()(logits, labels).loss
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_translation_changes_at_least_one_axis_profile() -> None:
    labels = _labels()
    one_hot = torch.nn.functional.one_hot(labels[:, 0], num_classes=3).movedim(-1, 1).float()
    shifted = torch.zeros_like(one_hot)
    shifted[:, :, 2:] = one_hot[:, :, :-2]
    differences = [
        float((perimeter_profile(one_hot[:, 1:], axis) - perimeter_profile(shifted[:, 1:], axis)).abs().sum())
        for axis in (0, 1, 2)
    ]
    assert differences[0] > 0.0
    assert sum(differences) > differences[0]


def test_objective_applies_exact_configured_weight() -> None:
    labels = _labels()
    logits = _logits(labels).clone()
    logits[:, 0, 4, 3, 5] = 4.0
    logits[:, 1, 4, 3, 5] = -4.0
    direct = PerimeterProfileLoss()(logits, labels).loss
    objective = NewConstraintObjective(NewConstraintConfig(
        equivariance_weight=0.0,
        perimeter_profile_weight=0.4,
    ))
    output = objective(torch.nn.Identity(), logits, logits, labels)
    assert set(output["results"]) == {"perimeter_profile"}
    torch.testing.assert_close(output["loss"], 0.4 * direct)
