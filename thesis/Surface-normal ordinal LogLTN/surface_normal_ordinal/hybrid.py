"""Tolerance-aware components for combining boundary anchors and one-cut logic."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .objective import sample_volume, semantic_field


def _validate_ray_inputs(
    values: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    tolerance_mm: float,
    margin: float,
    temperature: float,
) -> None:
    if values.ndim != 2:
        raise ValueError("values must have shape [N,S].")
    if offsets_mm.ndim != 1 or offsets_mm.numel() != values.shape[1]:
        raise ValueError("offsets_mm must match the ray sample dimension.")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("At least one ray with two samples is required.")
    if not bool(torch.all(offsets_mm[1:] > offsets_mm[:-1])):
        raise ValueError("offsets_mm must be strictly increasing.")
    if (
        tolerance_mm < 0
        or margin < 0
        or temperature <= 0
        or not all(
            math.isfinite(value)
            for value in (tolerance_mm, margin, temperature)
        )
    ):
        raise ValueError(
            "tolerance and margin must be finite and non-negative; "
            "temperature must be finite and positive."
        )


def onecut_assignment_log_scores(
    values: torch.Tensor,
    *,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return log truth for every foreground-prefix/background-suffix cut."""

    foreground = F.logsigmoid((values - margin) / temperature)
    background = F.logsigmoid((-values - margin) / temperature)
    foreground_prefix = torch.cumsum(foreground, dim=1)[:, :-1]
    background_prefix = torch.cumsum(background, dim=1)
    background_suffix = background_prefix[:, -1:] - background_prefix[:, :-1]
    return foreground_prefix + background_suffix


def conditional_cut_location_loss_from_values(
    values: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    tolerance_mm: float = 1.0,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize cut location outside the tolerance, conditional on any cut.

    Unlike the full one-cut loss, this term divides the probability mass of
    allowed cuts by the mass of every possible cut.  It therefore concentrates
    on *where* the transition occurs, leaving absolute foreground/background
    anchoring to a separate BCE term.
    """

    _validate_ray_inputs(
        values,
        offsets_mm,
        tolerance_mm=tolerance_mm,
        margin=margin,
        temperature=temperature,
    )
    cut_midpoints = 0.5 * (offsets_mm[:-1] + offsets_mm[1:])
    allowed = cut_midpoints.abs() <= tolerance_mm + 1e-7
    if not bool(allowed.any()):
        raise ValueError("No candidate cut lies inside the tolerance interval.")
    scores = onecut_assignment_log_scores(
        values, margin=margin, temperature=temperature
    )
    log_allowed = torch.logsumexp(scores[:, allowed], dim=1)
    log_any = torch.logsumexp(scores, dim=1)
    log_conditional_mass = torch.minimum(
        log_allowed - log_any,
        torch.zeros_like(log_allowed),
    )
    return -log_conditional_mass.mean(), log_conditional_mass.exp()


def ray_anchor_bce_from_values(
    values: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    tolerance_mm: float = 1.0,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced BCE anchors strictly outside the ambiguous tolerance tube."""

    _validate_ray_inputs(
        values,
        offsets_mm,
        tolerance_mm=tolerance_mm,
        margin=margin,
        temperature=temperature,
    )
    inside = offsets_mm < -tolerance_mm - 1e-7
    outside = offsets_mm > tolerance_mm + 1e-7
    if not bool(inside.any()) or not bool(outside.any()):
        raise ValueError("Rays must contain anchors on both sides of the tolerance.")
    inside_loss = F.softplus(-(values[:, inside] - margin) / temperature).mean()
    outside_loss = F.softplus((values[:, outside] + margin) / temperature).mean()
    return 0.5 * (inside_loss + outside_loss), inside, outside


def tolerance_aware_components(
    logits: torch.Tensor,
    ray_coordinates: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    tolerance_mm: float = 1.0,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return non-overlapping anchor and conditional cut-location objectives."""

    if logits.ndim != 4:
        raise ValueError("logits must have shape [3,D,H,W].")
    if ray_coordinates.ndim != 3 or ray_coordinates.shape[-1] != 3:
        raise ValueError("ray_coordinates must have shape [N,S,3].")
    field = semantic_field(logits, "outer")
    values = sample_volume(field, ray_coordinates)
    anchor_loss, inside, outside = ray_anchor_bce_from_values(
        values,
        offsets_mm,
        tolerance_mm=tolerance_mm,
        margin=margin,
        temperature=temperature,
    )
    location_loss, allowed_mass = conditional_cut_location_loss_from_values(
        values,
        offsets_mm,
        tolerance_mm=tolerance_mm,
        margin=margin,
        temperature=temperature,
    )
    diagnostics = {
        "allowed_cut_mass": allowed_mass,
        "inside_anchor_count": inside.sum(),
        "outside_anchor_count": outside.sum(),
    }
    return anchor_loss, location_loss, diagnostics
