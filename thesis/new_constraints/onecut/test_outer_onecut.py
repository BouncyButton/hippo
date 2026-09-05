"""Semantic, numerical, and integration tests for outer one-cut LogLTN."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from thesis.new_constraints import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.onecut.outer_onecut import (
    OuterOneCutLogLTNLoss,
    _onecut_components,
)
from thesis.new_constraints.train_swinunetr_constraints import resolve_constraint_config


class UnusedModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise AssertionError("One-cut must reuse the segmentation logits.")


def _labels(size: int = 16) -> torch.Tensor:
    labels = torch.zeros((1, 1, size, size, size), dtype=torch.long)
    labels[:, :, 4:8, 4:12, 4:12] = 1
    labels[:, :, 8:12, 4:12, 4:12] = 2
    return labels


def _class_logits(labels: torch.Tensor, magnitude: float = 6.0) -> torch.Tensor:
    logits = torch.full((labels.shape[0], 3, *labels.shape[-3:]), -magnitude)
    for class_id in (0, 1, 2):
        logits[:, class_id][labels[:, 0] == class_id] = magnitude
    return logits


def test_exact_onecut_prefers_a_tolerance_aligned_single_crossing() -> None:
    offsets = torch.tensor((-2.0, -1.0, 0.0, 1.0, 2.0))
    profiles = torch.tensor(
        (
            (8.0, 8.0, 8.0, -8.0, -8.0),  # one crossing in tolerance
            (8.0, 8.0, 8.0, 8.0, -8.0),  # outward-shifted crossing
            (8.0, 8.0, 8.0, 8.0, 8.0),  # missing crossing
            (8.0, -8.0, 8.0, -8.0, -8.0),  # multiple crossings
        )
    )
    losses, truths, masses = _onecut_components(
        profiles,
        offsets,
        tolerance_mm=0.6,
        margin=0.0,
        temperature=1.0,
    )

    assert losses[0] < losses[1:].min()
    assert truths[0] > truths[1:].max()
    assert masses[0] > masses[1:].max()
    assert 0.0 <= float(truths.min()) <= float(truths.max()) <= 1.0
    assert 0.0 <= float(masses.min()) <= float(masses.max()) <= 1.0


def test_onecut_gradient_orients_far_inside_and_outside_samples() -> None:
    offsets = torch.tensor((-2.0, -1.0, 0.0, 1.0, 2.0))
    values = torch.zeros((1, 5), requires_grad=True)
    loss, _, _ = _onecut_components(
        values,
        offsets,
        tolerance_mm=0.6,
        margin=0.0,
        temperature=1.0,
    )
    gradient = torch.autograd.grad(loss.mean(), values)[0]

    assert torch.isfinite(gradient).all()
    assert gradient[0, 0] < 0  # gradient descent raises the inside log-odds
    assert gradient[0, -1] > 0  # gradient descent lowers the outside log-odds
    assert gradient.abs().sum() > 0


def test_grouped_outer_onecut_is_invariant_to_anterior_posterior_swaps() -> None:
    labels = _labels()
    correct = _class_logits(labels)
    swapped = correct.clone()
    swapped[:, 1], swapped[:, 2] = correct[:, 2].clone(), correct[:, 1].clone()
    objective = OuterOneCutLogLTNLoss(max_surface_points=128)

    correct_result = objective(correct, labels)
    swapped_result = objective(swapped, labels)

    assert correct_result.details["valid"].item()
    assert torch.allclose(correct_result.loss, swapped_result.loss, atol=1e-7)
    assert torch.allclose(correct_result.truth, swapped_result.truth, atol=1e-7)


def test_ray_geometry_is_deterministic_and_cached() -> None:
    mask = (_labels()[0, 0].numpy() != 0)
    first = OuterOneCutLogLTNLoss(max_surface_points=17, geometry_seed=19)
    second = OuterOneCutLogLTNLoss(max_surface_points=17, geometry_seed=19)

    first_rays = first._rays(mask)
    cached_rays = first._rays(mask.copy())
    second_rays = second._rays(mask)

    assert first_rays is cached_rays
    assert first_rays.coordinates.shape == (17, len(first.offsets_mm), 3)
    np.testing.assert_array_equal(first_rays.coordinates, second_rays.coordinates)


def test_empty_foreground_is_skipped_with_connected_zero_loss() -> None:
    labels = torch.zeros((1, 1, 12, 12, 12), dtype=torch.long)
    logits = torch.randn((1, 3, 12, 12, 12), requires_grad=True)
    result = OuterOneCutLogLTNLoss(max_surface_points=32)(logits, labels)
    gradient = torch.autograd.grad(result.loss, logits)[0]

    assert result.loss.item() == 0.0
    assert result.truth.numel() == 0
    assert not result.details["valid"].item()
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_public_loss_is_patient_balanced() -> None:
    first_labels = _labels()
    second_labels = torch.zeros_like(first_labels)
    second_labels[:, :, 5:11, 5:11, 5:11] = 1
    labels = torch.cat((first_labels, second_labels), dim=0)
    logits = _class_logits(labels)
    objective = OuterOneCutLogLTNLoss(max_surface_points=64)

    batch_loss = objective(logits, labels).loss
    individual_losses = torch.stack(
        [objective(logits[index : index + 1], labels[index : index + 1]).loss for index in range(2)]
    )

    assert torch.allclose(batch_loss, individual_losses.mean(), atol=1e-7)


def test_unified_objective_reuses_logits_and_applies_exact_weight() -> None:
    labels = _labels()
    logits = _class_logits(labels).requires_grad_(True)
    config = NewConstraintConfig(
        equivariance_weight=0.0,
        onecut_weight=0.025,
        onecut_max_surface_points=32,
    )
    objective = NewConstraintObjective(config)
    output = objective(UnusedModel(), torch.empty((1, 1, 16, 16, 16)), logits, labels)
    raw = output["results"]["outer_onecut"].loss

    assert torch.allclose(output["loss"], 0.025 * raw)
    assert torch.isfinite(torch.autograd.grad(output["loss"], logits)[0]).all()


def test_unified_objective_rejects_mixed_auxiliary_constraints() -> None:
    with pytest.raises(ValueError, match="Only one auxiliary"):
        NewConstraintObjective(
            NewConstraintConfig(
                equivariance_weight=0.1,
                bands_weight=0.1,
                onecut_weight=0.1,
            )
        )


def test_training_constraint_selector_resolves_only_onecut() -> None:
    config = resolve_constraint_config(
        SimpleNamespace(
            constraint_set="onecut",
            equivariance_weight=None,
            bands_weight=None,
            bands_calibration_json=None,
            bands_focal_gamma=0.0,
            bands_inner_focal_gamma=None,
            bands_outer_focal_gamma=None,
            bands_loss_type="focal_bce",
            tversky_fp_weight=0.6,
            tversky_fn_weight=0.4,
            onecut_weight=0.025,
            onecut_calibration_json="calibration.json",
            onecut_spacing=(1.0, 1.0, 1.0),
            onecut_radius_mm=3.0,
            onecut_ray_step_mm=0.5,
            onecut_tolerance_mm=1.0,
            onecut_margin=0.0,
            onecut_temperature=1.0,
            onecut_max_surface_points=4096,
            foreground_class_ids=(1, 2),
            complement_class_ids=(0,),
            translation_size=2,
            equivariance_max_samples=1,
            band_steps=2,
            seed=7,
        )
    )

    assert config.onecut_weight == 0.025
    assert config.equivariance_weight == 0.0
    assert config.bands_weight == 0.0
    assert config.onecut_geometry_seed == 7


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("radius_mm", 0.0),
        ("ray_step_mm", 0.0),
        ("tolerance_mm", -1.0),
        ("temperature", 0.0),
        ("max_surface_points", 0),
    ),
)
def test_invalid_geometry_is_rejected(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        OuterOneCutLogLTNLoss(**{keyword: value})
