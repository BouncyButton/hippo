"""Tests for the literal protocol-derived A/P plane constraint."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from .existential import ExistentialAPPlaneLoss, hard_project_ap_plane
from .conditional_ce import OriginalLabelAPConditionalCELoss


@pytest.mark.parametrize('require_both', [False, True])
@pytest.mark.parametrize('margin', [0.0, 0.3])
def test_vectorized_conditional_ce_matches_voxel_reference_and_empty_gradients(require_both, margin):
    generator = torch.Generator().manual_seed(42)
    labels = torch.randint(0, 3, (4, 1, 6, 7, 8), generator=generator)
    labels[1] = 0
    labels[2] = 1
    logits = torch.randn(4, 3, 6, 7, 8, generator=generator, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    result = OriginalLabelAPConditionalCELoss(require_both=require_both, margin=margin)(logits, labels)
    losses = []
    for z, y in zip(reference_logits, labels[:, 0]):
        support = y != 0
        valid = bool(support.any()) and (not require_both or bool((y == 1).any() & (y == 2).any()))
        if valid:
            gap = z[1] - z[2]
            costs = torch.where(y == 1, torch.nn.functional.softplus(margin-gap),
                                torch.nn.functional.softplus(margin+gap))
            losses.append(costs[support].mean())
    expected = torch.stack(losses).mean()
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert torch.count_nonzero(logits.grad[1]) == 0
    empty_logits = logits.detach().clone().requires_grad_(True)
    empty = OriginalLabelAPConditionalCELoss(require_both=require_both)(empty_logits, torch.zeros_like(labels))
    empty.loss.backward()
    assert empty.loss.item() == 0 and torch.count_nonzero(empty_logits.grad) == 0
from .location import BestFitAPPlaneLocationLoss
from ..objective import NewConstraintConfig, NewConstraintObjective


class _UnusedModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise AssertionError("The objective must reuse the supplied logits.")


def _planar_example() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros((1, 1, 5, 7, 5), dtype=torch.long)
    labels[:, :, 1:4, 1:3, 1:4] = 2
    labels[:, :, 1:4, 3:6, 1:4] = 1
    logits = torch.full((1, 3, 5, 7, 5), -6.0)
    for class_id in (0, 1, 2):
        logits[:, class_id][labels[:, 0] == class_id] = 6.0
    return logits, labels


def test_literal_axis_one_rule_selects_first_anterior_slice() -> None:
    logits, labels = _planar_example()
    result = ExistentialAPPlaneLoss(axis=1, anterior_high=True)(logits, labels)

    assert result.details["valid"].tolist() == [True]
    assert result.details["selected_cut"].tolist() == [3]
    assert result.loss < 1e-4


def test_existential_loss_does_not_require_planar_ground_truth() -> None:
    logits, labels = _planar_example()
    labels[0, 0, 2, 2, 2] = 1  # released masks contain small mixed-slice remnants
    result = ExistentialAPPlaneLoss(axis=1, anterior_high=True)(logits, labels)

    assert result.details["valid"].item()
    assert torch.isfinite(result.loss)


def test_confident_nonplanar_assignment_has_corrective_logit_gradient() -> None:
    logits, labels = _planar_example()
    logits = logits.clone()
    # Make a posterior island deep in the anterior slab.
    logits[:, 1, 2, 4, 2] = -40.0
    logits[:, 2, 2, 4, 2] = 40.0
    logits.requires_grad_()
    result = ExistentialAPPlaneLoss(axis=1, anterior_high=True)(logits, labels)
    gradient = torch.autograd.grad(result.loss, logits)[0]

    assert torch.isfinite(gradient).all()
    assert gradient[0, 1, 2, 4, 2] < 0
    assert gradient[0, 2, 2, 4, 2] > 0


def test_hard_projection_preserves_union_and_enforces_one_plane() -> None:
    logits, labels = _planar_example()
    logits = logits.clone()
    logits[:, 1, 2, 1, 2] = 20.0  # anterior island on posterior side
    logits[:, 2, 2, 4, 2] = 20.0  # posterior island on anterior side
    before = logits.argmax(dim=1)
    projection = hard_project_ap_plane(logits, axis=1, anterior_high=True)
    after = projection.labels
    cut = int(projection.cuts[0])
    y = torch.arange(after.shape[2]).reshape(1, 1, -1, 1)

    assert torch.equal(after != 0, before != 0)
    assert not bool(((after == 1) & (y < cut)).any())
    assert not bool(((after == 2) & (y >= cut)).any())
    assert projection.violations_before.item() == int((after != before).sum())


def test_projection_supports_reversed_orientation_explicitly() -> None:
    logits, _ = _planar_example()
    projection = hard_project_ap_plane(logits, axis=1, anterior_high=False)
    cut = int(projection.cuts[0])
    y = torch.arange(projection.labels.shape[2]).reshape(1, 1, -1, 1)

    assert not bool(((projection.labels == 1) & (y >= cut)).any())
    assert not bool(((projection.labels == 2) & (y < cut)).any())


def test_unified_objective_applies_exact_plane_weight() -> None:
    logits, labels = _planar_example()
    logits.requires_grad_()
    objective = NewConstraintObjective(
        NewConstraintConfig(equivariance_weight=0.0, ap_plane_weight=0.07)
    )
    output = objective(_UnusedModel(), torch.empty((1, 1, 5, 7, 5)), logits, labels)
    raw = output["results"]["existential_ap_plane"].loss

    assert torch.allclose(output["loss"], 0.07 * raw)
    assert torch.isfinite(torch.autograd.grad(output["loss"], logits)[0]).all()


def test_location_loss_uses_the_unique_label_derived_cut() -> None:
    logits, labels = _planar_example()
    result = BestFitAPPlaneLocationLoss(axis=1, anterior_high=True)(logits, labels)

    assert result.details["target_cut"].tolist() == [3]
    assert result.details["selected_cut"].tolist() == [3]
    assert result.details["cut_abs_error"].tolist() == [0.0]
    assert result.details["exact_cut"].tolist() == [1.0]
    assert result.details["target_cut_count"].tolist() == [1.0]
    assert result.details["gt_plane_disagreement_fraction"].tolist() == [0.0]
    assert result.details["target_cost"].tolist() == [0.0]
    assert result.details["target_cost_gap"].tolist() == [9.0]
    assert result.details["raw_selected_cut"].tolist() == [3]
    assert result.details["raw_cut_abs_error"].tolist() == [0.0]
    assert result.details["raw_exact_cut"].tolist() == [1.0]
    assert result.details["foreground_union_dice"].tolist() == [1.0]
    assert result.details["ap_swap_voxels"].tolist() == [0.0]
    assert result.loss < 1e-4


def test_location_loss_keeps_mixed_slice_case_and_finds_best_fit() -> None:
    logits, labels = _planar_example()
    labels[0, 0, 2, 2, 2] = 1
    result = BestFitAPPlaneLocationLoss(axis=1, anterior_high=True)(logits, labels)

    assert result.details["valid"].tolist() == [True]
    assert result.details["target_cut"].tolist() == [3]
    assert result.details["target_cut_count"].tolist() == [1.0]
    assert result.details["gt_plane_disagreement_fraction"].item() > 0


def test_location_loss_corrects_a_coherent_but_displaced_cut() -> None:
    _, labels = _planar_example()
    logits = torch.full((1, 3, 5, 7, 5), -6.0)
    logits[:, 0] = 0.0
    foreground = labels[:, 0] != 0
    y = torch.arange(7).reshape(1, 1, 7, 1)
    predicted_anterior = foreground & (y >= 4)
    predicted_posterior = foreground & (y < 4)
    logits[:, 1][predicted_anterior] = 12.0
    logits[:, 2][predicted_posterior] = 12.0
    logits.requires_grad_()

    existential = ExistentialAPPlaneLoss(axis=1, anterior_high=True)(logits, labels)
    location = BestFitAPPlaneLocationLoss(axis=1, anterior_high=True)(logits, labels)
    gradient = torch.autograd.grad(location.loss, logits)[0]

    assert existential.details["selected_cut"].tolist() == [4]
    assert existential.loss < 1e-4
    assert location.details["target_cut"].tolist() == [3]
    assert location.details["selected_cut"].tolist() == [4]
    assert location.details["cut_abs_error"].tolist() == [1.0]
    assert location.details["raw_selected_cut"].tolist() == [4]
    assert location.details["raw_cut_abs_error"].tolist() == [1.0]
    assert location.details["raw_within_one_cut"].tolist() == [1.0]
    assert location.details["foreground_union_dice"].tolist() == [1.0]
    assert location.details["ap_swap_voxels"].tolist() == [9.0]
    assert location.details["displaced_slab_swap_voxels"].tolist() == [9.0]
    assert location.details["displaced_slab_swap_fraction"].tolist() == [1.0]
    assert location.loss > 1.0
    assert gradient[0, 1, 2, 3, 2] < 0
    assert gradient[0, 2, 2, 3, 2] > 0


def test_location_loss_is_conditional_ap_ce_against_projected_plane() -> None:
    logits, labels = _planar_example()
    labels[0, 0, 2, 2, 2] = 1
    logits = logits.clone().requires_grad_()
    result = BestFitAPPlaneLocationLoss(axis=1, anterior_high=True)(logits, labels)

    support = labels[:, 0] != 0
    y = torch.arange(labels.shape[3]).reshape(1, 1, -1, 1)
    target_anterior = y >= int(result.details["target_cut"].item())
    margin = logits[:, 1] - logits[:, 2]
    projected_ce = torch.where(
        target_anterior,
        F.softplus(-margin),
        F.softplus(margin),
    )[support].mean()

    assert torch.allclose(result.loss, projected_ce)


def test_matched_conditional_ce_is_identical_on_exact_planar_labels() -> None:
    logits, labels = _planar_example()
    logits = (logits + torch.randn_like(logits)).requires_grad_()
    location = BestFitAPPlaneLocationLoss(axis=1, anterior_high=True)(logits, labels)
    control = OriginalLabelAPConditionalCELoss(axis=1, anterior_high=True)(
        logits, labels
    )

    assert torch.allclose(location.loss, control.loss)
    location_gradient = torch.autograd.grad(location.loss, logits, retain_graph=True)[0]
    control_gradient = torch.autograd.grad(control.loss, logits)[0]
    assert torch.allclose(location_gradient, control_gradient)


def test_matched_conditional_ce_preserves_mixed_slice_voxel_label() -> None:
    logits, labels = _planar_example()
    labels[0, 0, 2, 2, 2] = 1
    logits = logits.clone().requires_grad_()
    location = BestFitAPPlaneLocationLoss(axis=1, anterior_high=True)(logits, labels)
    control = OriginalLabelAPConditionalCELoss(axis=1, anterior_high=True)(
        logits, labels
    )

    assert not torch.allclose(location.loss, control.loss)


def test_unified_objective_routes_location_mode_and_weight() -> None:
    logits, labels = _planar_example()
    logits.requires_grad_()
    objective = NewConstraintObjective(
        NewConstraintConfig(
            equivariance_weight=0.0,
            ap_plane_weight=0.03,
            ap_plane_mode="location",
        )
    )
    output = objective(_UnusedModel(), torch.empty((1, 1, 5, 7, 5)), logits, labels)
    raw = output["results"]["ap_plane_location"].loss

    assert torch.allclose(output["loss"], 0.03 * raw)
    assert torch.isfinite(torch.autograd.grad(output["loss"], logits)[0]).all()
