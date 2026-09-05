import torch

from surface_normal_ordinal.objective import (
    onecut_log_truth_from_values,
    onecut_logltn_loss,
    ordinal_logltn_loss,
    sample_volume,
    semantic_field,
)


def test_grouped_log_odds_matches_softmax_foreground_probability() -> None:
    logits = torch.randn(3, 4, 5, 6)
    field = semantic_field(logits, "outer")
    probability = torch.softmax(logits, dim=0)[1:].sum(dim=0)
    assert torch.allclose(torch.sigmoid(field), probability, atol=1e-6)


def test_trilinear_sampling_uses_array_axis_coordinates() -> None:
    d, h, w = torch.meshgrid(
        torch.arange(4.0), torch.arange(5.0), torch.arange(6.0), indexing="ij"
    )
    volume = 100.0 * d + 10.0 * h + w
    coordinates = torch.tensor([[1.5, 2.0, 3.0]])
    assert torch.allclose(sample_volume(volume, coordinates), torch.tensor([173.0]))


def test_ordinal_loss_rewards_correct_order_and_is_common_bias_invariant() -> None:
    logits = torch.zeros(3, 5, 5, 5)
    logits[1, 1] = 3.0
    logits[1, 3] = -3.0
    inside = torch.tensor([[1.0, 2.0, 2.0]])
    outside = torch.tensor([[3.0, 2.0, 2.0]])
    correct = ordinal_logltn_loss(logits, inside, outside)[0]
    reversed_loss = ordinal_logltn_loss(logits, outside, inside)[0]
    biased = ordinal_logltn_loss(logits + 17.0, inside, outside)[0]
    assert correct < reversed_loss
    assert torch.allclose(correct, biased, atol=1e-6)


def test_outer_loss_is_invariant_to_anterior_posterior_swap() -> None:
    logits = torch.randn(3, 5, 5, 5)
    inside = torch.tensor([[1.0, 2.0, 2.0], [2.0, 1.0, 2.0]])
    outside = torch.tensor([[3.0, 2.0, 2.0], [2.0, 3.0, 2.0]])
    first = ordinal_logltn_loss(logits, inside, outside)[0]
    second = ordinal_logltn_loss(logits[[0, 2, 1]], inside, outside)[0]
    assert torch.allclose(first, second, atol=1e-6)


def test_gradient_raises_inside_field_and_lowers_outside_field() -> None:
    logits = torch.zeros(3, 5, 5, 5, requires_grad=True)
    inside = torch.tensor([[1.0, 2.0, 2.0]])
    outside = torch.tensor([[3.0, 2.0, 2.0]])
    loss = ordinal_logltn_loss(logits, inside, outside)[0]
    loss.backward()
    # Gradient descent decreases z0 inside and increases foreground there; the
    # signs reverse outside.
    assert logits.grad[0, 1, 2, 2] > 0
    assert logits.grad[1, 1, 2, 2] < 0
    assert logits.grad[0, 3, 2, 2] < 0
    assert logits.grad[1, 3, 2, 2] > 0


def test_onecut_prefers_a_single_nearby_transition() -> None:
    offsets = torch.arange(-3.0, 3.1, 1.0)
    correct = -5.0 * offsets
    shifted = -5.0 * (offsets - 2.0)
    missing = torch.full_like(offsets, 5.0)
    multiple = torch.tensor([5.0, 5.0, -5.0, 5.0, -5.0, -5.0, -5.0])
    values = torch.stack((correct, shifted, missing, multiple))
    log_truth, truth, allowed_mass, best_cut = onecut_log_truth_from_values(
        values, offsets, tolerance_mm=1.0
    )
    loss = -log_truth / values.shape[1]
    assert loss[0] < loss[1]
    assert loss[0] < loss[2]
    assert loss[0] < loss[3]
    assert truth[0] > truth[1]
    assert allowed_mass[0] > allowed_mass[1]
    assert torch.all(best_cut.abs() <= 1.0)


def test_onecut_log_truth_matches_exact_disjoint_assignment_sum() -> None:
    probabilities = torch.tensor([[0.8, 0.3, 0.1]])
    values = torch.logit(probabilities)
    offsets = torch.tensor([-1.0, 0.0, 1.0])
    log_truth = onecut_log_truth_from_values(
        values, offsets, tolerance_mm=1.0
    )[0]
    expected = 0.8 * (1.0 - 0.3) * (1.0 - 0.1) + 0.8 * 0.3 * (1.0 - 0.1)
    assert torch.allclose(torch.exp(log_truth), torch.tensor([expected]), atol=1e-6)


def test_onecut_loss_has_finite_full_ray_gradients() -> None:
    logits = torch.zeros(3, 7, 5, 5, requires_grad=True)
    points = torch.tensor([[[float(d), 2.0, 2.0] for d in range(7)]])
    offsets = torch.arange(-3.0, 3.1, 1.0)
    loss = onecut_logltn_loss(
        logits, points, offsets, tolerance_mm=1.0, interface="outer"
    )[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0
