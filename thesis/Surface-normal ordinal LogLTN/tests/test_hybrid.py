import torch

from surface_normal_ordinal.hybrid import (
    conditional_cut_location_loss_from_values,
    ray_anchor_bce_from_values,
)


def test_anchor_bce_pushes_inside_up_and_outside_down() -> None:
    offsets = torch.arange(-3.0, 3.01, 0.5)
    values = torch.zeros(2, len(offsets), requires_grad=True)
    loss, inside, outside = ray_anchor_bce_from_values(
        values, offsets, tolerance_mm=1.0
    )
    gradient = torch.autograd.grad(loss, values)[0]
    assert bool((gradient[:, inside] < 0).all())
    assert bool((gradient[:, outside] > 0).all())
    assert bool((gradient[:, ~(inside | outside)] == 0).all())


def test_conditional_location_prefers_a_cut_inside_tolerance() -> None:
    offsets = torch.arange(-3.0, 3.01, 0.5)
    centred = torch.where(offsets <= 0.0, 5.0, -5.0).unsqueeze(0)
    displaced = torch.where(offsets <= 2.0, 5.0, -5.0).unsqueeze(0)
    centred_loss, centred_mass = conditional_cut_location_loss_from_values(
        centred, offsets, tolerance_mm=1.0
    )
    displaced_loss, displaced_mass = conditional_cut_location_loss_from_values(
        displaced, offsets, tolerance_mm=1.0
    )
    assert centred_loss < displaced_loss
    assert centred_mass.item() > displaced_mass.item()


def test_hybrid_components_are_finite_at_uncertain_logits() -> None:
    offsets = torch.arange(-3.0, 3.01, 0.5)
    values = torch.zeros(3, len(offsets), requires_grad=True)
    anchor, _, _ = ray_anchor_bce_from_values(values, offsets)
    location, mass = conditional_cut_location_loss_from_values(values, offsets)
    gradient = torch.autograd.grad(anchor + location, values)[0]
    assert torch.isfinite(anchor)
    assert torch.isfinite(location)
    assert torch.isfinite(mass).all()
    assert torch.isfinite(gradient).all()
