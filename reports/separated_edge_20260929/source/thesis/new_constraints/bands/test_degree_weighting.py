"""Independent numerical checks for optional GT degree weighting."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from thesis.new_constraints.bands.outer_boundary import (
    OuterBoundaryBandLoss, build_boundary_bands, foreground_log_odds,
    inner_degree_weights,
)
from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.train_swinunetr_constraints import resolve_constraint_config


def geometry():
    labels = torch.zeros(2, 1, 13, 13, 13, dtype=torch.long)
    labels[0, :, 3:10, 3:10, 3:10] = 1
    labels[0, :, 7:10, 3:10, 3:10] = 2
    labels[0, :, 2, 6, 6] = 1
    labels[0, :, 1, 6, 6] = 1
    labels[1, :, 4:8, 4:8, 4:8] = 2
    labels[1, :, 1, 1, 1] = 1
    return labels


def reference_weights(labels, alpha, normalization):
    """Count each occupied voxel's neighbours directly, without convolution."""
    foreground = labels != 0
    inner, outer = build_boundary_bands(foreground)
    degree = torch.zeros_like(labels)
    for b, c, x, y, z in foreground.nonzero().tolist():
        degree[b, c, x, y, z] = sum(
            int(foreground[b, c, x + dx, y + dy, z + dz])
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
        )
    surface = foreground & (degree < 6)
    weights = torch.ones_like(labels, dtype=torch.float32)
    for b in range(labels.shape[0]):
        mask = inner[b] if normalization == "inner" else surface[b]
        raw = 1 + alpha * (6 - degree[b][mask].float()) / 6
        weights[b][mask] = raw / raw.mean()
    return weights, inner, outer, surface


@pytest.mark.parametrize("normalization", ["inner", "surface"])
def test_weights_and_backprop_match_explicit_per_case_formula(normalization):
    labels = geometry()
    weights, inner, outer, _ = reference_weights(labels, 2, normalization)
    actual_weights = inner_degree_weights(labels != 0, inner, alpha=2, normalization=normalization)
    torch.testing.assert_close(actual_weights[inner], weights[inner])
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 13, 13, 13, requires_grad=True)
    result = OuterBoundaryBandLoss(degree_alpha=2, degree_normalization=normalization)(logits, labels)
    odds = foreground_log_odds(logits, (1, 2), (0,))
    expected = []
    for b in range(2):
        positive = (F.softplus(-odds[b])[inner[b, 0]] * weights[b, 0][inner[b, 0]]).sum()
        positive = positive / (inner[b].sum() + 1e-6)
        negative = F.softplus(odds[b])[outer[b, 0]].sum() / (outer[b].sum() + 1e-6)
        expected.append((positive + negative) / 2)
    expected = torch.stack(expected).mean()
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(torch.autograd.grad(result.loss, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])
    uniform = OuterBoundaryBandLoss()(logits, labels)
    assert torch.equal(result.details['outer_loss'], uniform.details['outer_loss'])


def test_normalizations_have_distinct_surface_mass_but_equal_total_mass():
    labels = geometry()
    _, inner, _, surface = reference_weights(labels, 2, "surface")
    a = inner_degree_weights(labels != 0, inner, alpha=2, normalization="inner")
    b = inner_degree_weights(labels != 0, inner, alpha=2, normalization="surface")
    for index in range(2):
        for weights in (a, b):
            torch.testing.assert_close(weights[index][inner[index]].sum(), inner[index].sum().float())
        torch.testing.assert_close(b[index][surface[index]].mean(), torch.tensor(1.0))
        assert a[index][surface[index]].mean() > 1
    assert (b[inner & ~surface] == 1).all()
    assert (a[inner & ~surface] < 1).all()


@pytest.mark.parametrize("normalization", ["inner", "surface"])
@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_zero_alpha_preserves_legacy_loss_and_gradient_exactly(normalization, gamma):
    labels = geometry()
    logits = torch.randn(2, 3, 13, 13, 13, requires_grad=True)
    old = OuterBoundaryBandLoss(focal_gamma=gamma)(logits, labels).loss
    new = OuterBoundaryBandLoss(focal_gamma=gamma, degree_alpha=0, degree_normalization=normalization)(logits, labels).loss
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, logits)[0], torch.autograd.grad(new, logits)[0])


@pytest.mark.parametrize("normalization", ["inner", "surface"])
def test_empty_case_is_skipped_and_ap_swaps_do_not_change_weights(normalization):
    labels = geometry()
    logits = torch.randn(2, 3, 13, 13, 13, requires_grad=True)
    loss = OuterBoundaryBandLoss(degree_alpha=2, degree_normalization=normalization)
    labels[1].zero_()
    result = loss(logits, labels)
    torch.testing.assert_close(result.loss, loss(logits[:1], labels[:1]).loss)
    swapped = torch.where(labels == 1, 2, torch.where(labels == 2, 1, 0))
    torch.testing.assert_close(result.loss, loss(logits[:, [0, 2, 1]], swapped).loss)
    empty = loss(logits, torch.zeros_like(labels)).loss
    assert empty == 0 and torch.isfinite(torch.autograd.grad(empty, logits)[0]).all()


def test_crop_edge_is_rejected_only_when_degree_weighting_is_enabled():
    labels = geometry()
    labels[0, 0, 0, 6, 6] = 1
    logits = torch.randn(2, 3, 13, 13, 13)
    assert torch.isfinite(OuterBoundaryBandLoss()(logits, labels).loss)
    with pytest.raises(ValueError, match="background halo"):
        OuterBoundaryBandLoss(degree_alpha=2)(logits, labels)


@pytest.mark.parametrize("alpha", [-1.0, float("nan"), float("inf")])
def test_invalid_strength_is_rejected(alpha):
    with pytest.raises(ValueError, match="degree_alpha"):
        OuterBoundaryBandLoss(degree_alpha=alpha)


def test_configuration_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="degree_normalization"):
        OuterBoundaryBandLoss(degree_normalization="invalid")
    for kwargs in ({"bands_weight": 0}, {"bands_weight": 0.1, "bands_loss_type": "class_tversky"}):
        with pytest.raises(ValueError, match="Degree weighting"):
            NewConstraintObjective(NewConstraintConfig(equivariance_weight=0, bands_degree_alpha=2, **kwargs))


@pytest.mark.parametrize("normalization", ["inner", "surface"])
def test_training_configuration_routes_degree_options(normalization):
    args = SimpleNamespace(constraint_set="bands", bands_weight=0.1, bands_calibration_json="report.json",
                           bands_degree_alpha=2, bands_degree_normalization=normalization)
    config = resolve_constraint_config(args)
    objective = NewConstraintObjective(config)
    assert objective.bands.degree_alpha == 2
    assert objective.bands.degree_normalization == normalization
