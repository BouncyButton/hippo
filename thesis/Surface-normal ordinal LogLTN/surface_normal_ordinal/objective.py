"""Differentiable semantic fields and the ordinal LogLTN formula."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def semantic_field(logits: torch.Tensor, interface: str = "outer") -> torch.Tensor:
    """Return the exact grouped outer log-odds or the A/P contrast."""

    if logits.ndim not in (4, 5) or logits.shape[-4] != 3:
        raise ValueError("logits must have shape [3,D,H,W] or [B,3,D,H,W].")
    channel_dim = logits.ndim - 4
    values = logits.float()
    if interface == "outer":
        foreground = torch.logsumexp(
            torch.index_select(values, channel_dim, torch.tensor([1, 2], device=values.device)),
            dim=channel_dim,
        )
        background = torch.select(values, channel_dim, 0)
        return foreground - background
    if interface == "ap":
        return torch.select(values, channel_dim, 1) - torch.select(values, channel_dim, 2)
    raise ValueError("interface must be 'outer' or 'ap'.")


def sample_volume(volume: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample ``[D,H,W]`` at ``[...,3]`` array-axis coordinates."""

    if volume.ndim != 3 or coordinates.ndim < 2 or coordinates.shape[-1] != 3:
        raise ValueError("Expected volume [D,H,W] and coordinates [...,3].")
    original_shape = coordinates.shape[:-1]
    size = torch.tensor(volume.shape, device=coordinates.device, dtype=coordinates.dtype)
    grid = 2.0 * coordinates / (size - 1.0) - 1.0
    # grid_sample expects x/y/z = W/H/D, the reverse of array-axis D/H/W.
    grid = grid.flip(-1).reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        volume.reshape(1, 1, *volume.shape),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(original_shape)


def ordinal_truth(
    inside_values: torch.Tensor,
    outside_values: torch.Tensor,
    *,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Truth of ``r(inside) > r(outside) + margin``."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return torch.sigmoid((inside_values - outside_values - margin) / temperature)


def ordinal_logltn_loss(
    logits: torch.Tensor,
    inside_coordinates: torch.Tensor,
    outside_coordinates: torch.Tensor,
    *,
    interface: str = "outer",
    margin: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return patient-normalized guarded-universal LogLTN loss and diagnostics."""

    if logits.ndim != 4:
        raise ValueError("The audit loss expects one case with logits [3,D,H,W].")
    if inside_coordinates.shape != outside_coordinates.shape:
        raise ValueError("inside and outside coordinates must have identical shape.")
    if inside_coordinates.ndim != 2 or inside_coordinates.shape[1] != 3:
        raise ValueError("pair coordinates must have shape [N,3].")
    if inside_coordinates.shape[0] == 0:
        raise ValueError("At least one guarded pair is required.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    field = semantic_field(logits, interface)
    inside = sample_volume(field, inside_coordinates)
    outside = sample_volume(field, outside_coordinates)
    scaled_margin = (inside - outside - margin) / temperature
    loss = F.softplus(-scaled_margin).mean()
    return loss, torch.sigmoid(scaled_margin), inside - outside


def onecut_log_truth_from_values(
    values: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    tolerance_mm: float = 1.0,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the existential exactly-one-cut formula on ordered ray values.

    A cut after sample ``k`` represents the complete assignment
    ``H(x_0)..H(x_k), not-H(x_{k+1})..not-H(x_S)``.  The allowed disjunction
    contains cuts whose physical midpoint lies within ``tolerance_mm`` of the
    ground-truth surface.  Returned values are the raw log truth, a
    length-normalized truth, allowed one-cut mass conditional on any one-cut
    assignment, and the best allowed cut midpoint.
    """

    if values.ndim != 2:
        raise ValueError("values must have shape [N,S].")
    if offsets_mm.ndim != 1 or offsets_mm.numel() != values.shape[1]:
        raise ValueError("offsets_mm must match the ray sample dimension.")
    if values.shape[1] < 2 or not bool(torch.all(offsets_mm[1:] > offsets_mm[:-1])):
        raise ValueError("At least two strictly ordered ray samples are required.")
    if (
        tolerance_mm < 0
        or temperature <= 0
        or margin < 0
        or not all(math.isfinite(value) for value in (tolerance_mm, temperature, margin))
    ):
        raise ValueError(
            "tolerance_mm and margin must be finite and non-negative; temperature must be finite and positive."
        )
    cut_midpoints = 0.5 * (offsets_mm[:-1] + offsets_mm[1:])
    allowed = cut_midpoints.abs() <= tolerance_mm + 1e-7
    if not bool(allowed.any()):
        raise ValueError("No candidate cut lies inside the requested tolerance.")

    foreground_log_truth = F.logsigmoid((values - margin) / temperature)
    background_log_truth = F.logsigmoid((-values - margin) / temperature)
    foreground_prefix = torch.cumsum(foreground_log_truth, dim=1)[:, :-1]
    background_prefix = torch.cumsum(background_log_truth, dim=1)
    background_suffix = background_prefix[:, -1:] - background_prefix[:, :-1]
    cut_log_truth = foreground_prefix + background_suffix
    allowed_scores = cut_log_truth[:, allowed]
    log_truth = torch.logsumexp(allowed_scores, dim=1)
    all_onecut_log_mass = torch.logsumexp(cut_log_truth, dim=1)
    conditional_allowed_mass = torch.exp(log_truth - all_onecut_log_mass).clamp(0.0, 1.0)
    normalized_truth = torch.exp(log_truth / values.shape[1]).clamp(0.0, 1.0)
    best_allowed = allowed_scores.argmax(dim=1)
    best_cut_midpoint = cut_midpoints[allowed][best_allowed]
    return log_truth, normalized_truth, conditional_allowed_mass, best_cut_midpoint


def onecut_logltn_loss(
    logits: torch.Tensor,
    ray_coordinates: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    interface: str = "outer",
    tolerance_mm: float = 1.0,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the literal-normalized, patient-balanced one-cut LogLTN loss."""

    if logits.ndim != 4:
        raise ValueError("The audit loss expects one case with logits [3,D,H,W].")
    if ray_coordinates.ndim != 3 or ray_coordinates.shape[-1] != 3:
        raise ValueError("ray_coordinates must have shape [N,S,3].")
    if ray_coordinates.shape[0] == 0:
        raise ValueError("At least one guarded ray is required.")
    field = semantic_field(logits, interface)
    values = sample_volume(field, ray_coordinates)
    log_truth, truth, allowed_mass, best_cut = onecut_log_truth_from_values(
        values,
        offsets_mm,
        tolerance_mm=tolerance_mm,
        margin=margin,
        temperature=temperature,
    )
    loss = (-log_truth / values.shape[1]).mean()
    return loss, truth, allowed_mass, best_cut
