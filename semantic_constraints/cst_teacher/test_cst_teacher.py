from __future__ import annotations

import pytest
import torch

from .analyze_embeddings import discover_clusters
from .counterfactual_repair import normalized_violation, redistribute_ap, repair_prediction
from .data import (
    corrupt_mask_elements,
    element_metadata,
    extract_coronal_slabs,
    uniform_positions,
)
from .descriptors import DESCRIPTOR_NAMES, labels_to_probabilities, sampled_slice_profiles, soft_descriptors
from .losses import (
    anomaly_constraint,
    conditional_descriptor_constraint,
    descriptor_quantile_loss,
    slice_profile_loss,
)
from .model import CSTDescriptorTeacher, CSTMaskAnomalyTeacher, SetConv2d
from .profile_counterfactual import redistribute_by_slice
from .risk_probe import grouped_folds, repeated_nested_predictions, slice_dice_error


def test_setconv_is_permutation_equivariant_when_coordinates_follow_elements():
    torch.manual_seed(1)
    block = SetConv2d(3, 8, heads=2).eval()
    elements = torch.randn(2, 5, 3, 12, 10)
    metadata = torch.linspace(-1, 1, 5).view(1, 5, 1).expand(2, -1, -1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    expected = block(elements, metadata)[:, permutation]
    actual = block(elements[:, permutation], metadata[:, permutation])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_coronal_slabs_have_replication_padding_and_position_metadata():
    image = torch.arange(5.0).view(1, 1, 5, 1).expand(1, 3, 5, 2)
    positions = torch.tensor([0, 2, 4])
    slabs = extract_coronal_slabs(image, positions, slab_depth=3)
    assert slabs.shape == (3, 3, 3, 2)
    assert torch.equal(slabs[0, :, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(slabs[-1, :, 0, 0], torch.tensor([3.0, 4.0, 4.0]))
    assert torch.equal(element_metadata(positions, 5).squeeze(1), torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(uniform_positions(5, 3), positions)
    resized = extract_coronal_slabs(image, positions, slab_depth=3, output_size=(2, 2))
    assert resized.shape == (3, 3, 2, 2)


def test_soft_descriptors_and_profiles_are_finite_and_differentiable():
    logits = torch.full((1, 3, 8, 8, 8), -4.0, requires_grad=True)
    with torch.no_grad():
        logits[:, 1, 2:6, 4:7, 2:6] = 4
        logits[:, 2, 2:6, 1:4, 2:6] = 4
    probabilities = logits.softmax(dim=1)
    descriptors = soft_descriptors(probabilities)
    profiles = sampled_slice_profiles(probabilities, torch.tensor([1, 3, 5, 7]))
    assert descriptors.shape == (1, len(DESCRIPTOR_NAMES))
    assert profiles.shape == (1, 4, 2)
    assert torch.isfinite(descriptors).all() and torch.isfinite(profiles).all()
    assert descriptors[0, 2] > 0
    (descriptors.sum() + profiles.sum()).backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_soft_descriptors_are_amp_safe_on_cpu(dtype):
    logits = torch.randn(1, 3, 5, 5, 5, dtype=dtype, requires_grad=True)
    descriptors = soft_descriptors(logits.softmax(dim=1))
    assert descriptors.dtype == torch.float32 and torch.isfinite(descriptors).all()
    descriptors.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_descriptor_teacher_shapes_and_ordered_quantiles():
    teacher = CSTDescriptorTeacher(channels=(8, 16), heads=2)
    output = teacher(
        torch.randn(2, 6, 3, 16, 16),
        torch.randn(2, 6, 1),
        torch.ones(2, 6, dtype=torch.bool),
    )
    assert output.descriptor_quantiles.shape == (2, len(DESCRIPTOR_NAMES), 3)
    assert output.slice_profiles.shape == (2, 6, 2)
    assert output.case_embedding.shape == (2, 16)
    assert torch.all(output.descriptor_quantiles[..., 0] <= output.descriptor_quantiles[..., 1])
    assert torch.all(output.descriptor_quantiles[..., 1] <= output.descriptor_quantiles[..., 2])
    loss = descriptor_quantile_loss(output.descriptor_quantiles, torch.randn(2, len(DESCRIPTOR_NAMES)))
    loss.backward()
    assert torch.isfinite(loss)


def test_conditional_constraint_detaches_bounds_and_corrects_probabilities():
    logits = torch.randn(2, 3, 6, 6, 6, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    current = soft_descriptors(probabilities).detach()
    quantiles = torch.stack((current + 0.2, current + 0.3, current + 0.4), dim=-1).requires_grad_()
    result = conditional_descriptor_constraint(probabilities, quantiles)
    assert result.loss > 0
    result.loss.backward()
    assert quantiles.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_corruption_and_frozen_anomaly_constraint_have_expected_gradients():
    masks = torch.zeros(2, 5, 2, 12, 12)
    masks[:, :, 0, 3:7, 3:7] = 1
    corrupted, targets = corrupt_mask_elements(masks, corruption_rate=0.4)
    assert corrupted.shape == masks.shape and targets.shape == masks.shape[:2]
    assert targets.any(dim=1).all()
    assert torch.any(corrupted != masks)

    teacher = CSTMaskAnomalyTeacher(channels=(8, 16), heads=2).eval()
    teacher.requires_grad_(False)
    soft_masks = masks.clone().requires_grad_()
    loss = anomaly_constraint(
        teacher,
        torch.randn(2, 5, 3, 12, 12),
        soft_masks,
        torch.randn(2, 5, 1),
    )
    loss.backward()
    assert soft_masks.grad is not None and torch.isfinite(soft_masks.grad).all()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_labels_reject_out_of_range_values():
    labels = torch.zeros(1, 1, 3, 3, 3)
    labels[..., 0, 0, 0] = 3
    with pytest.raises(ValueError):
        labels_to_probabilities(labels)


def test_profile_loss_supports_l1_and_rejects_unknown_kind():
    predicted = torch.tensor([[[0.0, 1.0], [0.5, 0.5]]])
    target = torch.zeros_like(predicted)
    assert torch.isclose(slice_profile_loss(predicted, target, loss_kind="l1"), torch.tensor(0.5))
    with pytest.raises(ValueError):
        slice_profile_loss(predicted, target, loss_kind="unknown")


def test_cluster_discovery_is_train_only_and_emits_soft_memberships():
    rng = torch.Generator().manual_seed(4)
    first = torch.randn(12, 4, generator=rng) * 0.1 - 2
    second = torch.randn(12, 4, generator=rng) * 0.1 + 2
    train_embeddings = torch.cat((first, second)).numpy()
    train_descriptors = torch.cat(
        (
            torch.randn(12, len(DESCRIPTOR_NAMES), generator=rng) * 0.05,
            torch.randn(12, len(DESCRIPTOR_NAMES), generator=rng) * 0.05 + 1,
        )
    ).numpy()
    validation_embeddings = torch.stack((first[0], second[0])).numpy()
    validation_descriptors = torch.stack(
        (torch.zeros(len(DESCRIPTOR_NAMES)), torch.ones(len(DESCRIPTOR_NAMES)))
    ).numpy()
    report, arrays = discover_clusters(
        train_embeddings,
        train_descriptors,
        validation_embeddings,
        validation_descriptors,
        max_clusters=2,
        minimum_cluster_size=5,
        bootstrap_samples=5,
        seed=3,
    )
    assert report["selection"]["selected_clusters"] == 2
    assert len(report["observational_rules"]) == 2
    assert arrays["validation_memberships"].shape == (2, 2)
    membership_sums = torch.from_numpy(arrays["validation_memberships"].sum(axis=1))
    assert torch.allclose(membership_sums, torch.ones_like(membership_sums))


def test_counterfactual_ap_repair_preserves_foreground_and_reduces_violation():
    logits = torch.randn(1, 3, 8, 8, 8)
    original = logits.softmax(dim=1)
    initial = soft_descriptors(original)[0, 0]
    lower = initial.detach() + 0.05
    upper = lower + 0.10
    shifted = redistribute_ap(original, torch.ones(1, 1, 1, 1, 1) * 0.25)
    assert torch.allclose(shifted[:, 1:3].sum(dim=1), original[:, 1:3].sum(dim=1), atol=1e-7)
    assert torch.allclose(shifted.sum(dim=1), torch.ones_like(shifted[:, 0]), atol=1e-6)

    repaired, diagnostics = repair_prediction(
        original,
        lower,
        upper,
        descriptor_index=0,
        mode="global_ap_bias",
        gamma=1.0,
        steps=30,
        learning_rate=0.1,
    )
    assert diagnostics["final_violation"] < diagnostics["initial_violation"]
    assert torch.allclose(repaired[:, 1:3].sum(dim=1), original[:, 1:3].sum(dim=1), atol=1e-6)
    final = soft_descriptors(repaired)[0, 0]
    assert normalized_violation(final, lower, upper) < normalized_violation(initial, lower, upper)


def test_slice_redistribution_is_normalized_and_has_local_y_effects():
    original = torch.randn(1, 3, 5, 6, 4).softmax(dim=1)
    foreground_delta = torch.zeros(1, 1, 1, 6, 1)
    ap_delta = torch.zeros_like(foreground_delta)
    foreground_delta[..., 2, :] = 0.5
    ap_delta[..., 4, :] = -0.5
    repaired = redistribute_by_slice(original, foreground_delta, ap_delta)
    assert repaired.shape == original.shape
    assert torch.allclose(repaired.sum(dim=1), torch.ones_like(repaired[:, 0]), atol=1e-6)
    assert torch.allclose(repaired[:, :, :, 0], original[:, :, :, 0], atol=1e-5)
    assert not torch.allclose(repaired[:, :, :, 2], original[:, :, :, 2])
    assert not torch.allclose(repaired[:, :, :, 4], original[:, :, :, 4])


def test_grouped_risk_probe_prevents_case_leakage_and_fits_signal():
    groups = torch.arange(20).repeat_interleave(3).numpy()
    x = torch.arange(20.0).repeat_interleave(3).view(-1, 1).numpy()
    y = (2.0 * x[:, 0] + 1.0)
    folds = grouped_folds(groups, folds=5, seed=2)
    assert set().union(*(set(fold.tolist()) for fold in folds)) == set(range(20))
    assert sum(len(fold) for fold in folds) == 20
    prediction, _ = repeated_nested_predictions(
        x,
        y,
        groups,
        outer_folds=5,
        inner_folds=4,
        repeats=2,
        alphas=[1e-4, 1e-2, 1.0],
        seed=7,
    )
    assert ((prediction - y) ** 2).mean() < 0.1


def test_slice_dice_error_is_zero_for_exact_prediction():
    labels = torch.zeros(1, 1, 4, 5, 4, dtype=torch.long)
    labels[:, :, 1:3, 1:3, 1:3] = 1
    labels[:, :, 1:3, 3:5, 1:3] = 2
    probabilities = labels_to_probabilities(labels)
    error = slice_dice_error(probabilities, labels, torch.arange(5))
    assert error.shape == (1, 5)
    assert torch.allclose(error, torch.zeros_like(error))
