"""Patient-specific sagittal contour fitting with differentiable step heights.

For every sagittal x section, the foreground's upper and lower z contours are
functions of posterior-to-anterior y. The prediction is converted to soft
first/last foreground positions, and both predicted and annotated curves are
projected onto decreasing step functions. Gradients flow through the fitted
block means back to the voxel logits; the discrete block partition is held
fixed for each forward pass, as with a piecewise-linear isotonic projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .constraint_result import ConstraintResult
from .directional_steps import terminal_distributions


def fit_decreasing_steps(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fit least-squares decreasing steps along the final dimension.

    Pool-adjacent-violators chooses blocks from detached curve values. Means
    inside those blocks remain ordinary tensor operations, giving the exact
    local derivative wherever the block partition does not change.
    """
    if values.shape != valid.shape or values.ndim < 2:
        raise ValueError("values and valid must have matching [*, positions] shapes")
    length = values.shape[-1]
    flat = values.reshape(-1, length)
    active = valid.bool().reshape(-1, length)
    detached = flat.detach().cpu().tolist()
    active_cpu = active.detach().cpu().tolist()
    group_ids = torch.zeros((flat.shape[0], length), dtype=torch.long)
    for row, (curve, mask) in enumerate(zip(detached, active_cpu)):
        blocks: list[tuple[float, int, list[int]]] = []
        next_group = 0
        for index, present in enumerate(mask):
            if not present:
                for _, _, indices in blocks:
                    group_ids[row, indices] = row * length + next_group
                    next_group += 1
                blocks = []
                continue
            blocks.append((float(curve[index]), 1, [index]))
            while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
                right_sum, right_count, right_indices = blocks.pop()
                left_sum, left_count, left_indices = blocks.pop()
                blocks.append((left_sum + right_sum, left_count + right_count, left_indices + right_indices))
        for _, _, indices in blocks:
            group_ids[row, indices] = row * length + next_group
            next_group += 1
    groups = group_ids.to(device=values.device).reshape(-1)
    weights = active.to(dtype=values.dtype).reshape(-1)
    total_groups = flat.numel()
    sums = torch.zeros(total_groups, device=values.device, dtype=values.dtype)
    counts = torch.zeros_like(sums)
    sums = sums.scatter_add(0, groups, flat.reshape(-1) * weights)
    counts = counts.scatter_add(0, groups, weights)
    fitted = sums[groups] / counts[groups].clamp_min(1)
    return torch.where(active, fitted.reshape_as(flat), 0).reshape_as(values)


def _section_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average within each section, then equally across valid sections."""
    per_section = (values * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    has_section = valid.any(dim=-1)
    return (per_section * has_section).sum(dim=-1) / has_section.sum(dim=-1).clamp_min(1)


class SagittalStepContourLoss(nn.Module):
    """Compare fitted superior/inferior curves in each sagittal section."""

    def __init__(
        self,
        *,
        transition_weight: float = 0.0,
        residual_weight: float = 0.0,
        presence_weight: float = 0.1,
        minimum_positions: int = 3,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        for name, weight in (
            ("transition_weight", transition_weight),
            ("residual_weight", residual_weight),
            ("presence_weight", presence_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if minimum_positions < 2 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("minimum_positions must be at least 2 and epsilon positive")
        self.transition_weight = float(transition_weight)
        self.residual_weight = float(residual_weight)
        self.presence_weight = float(presence_weight)
        self.minimum_positions = int(minimum_positions)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        if labels.ndim == 5 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if logits.ndim != 5 or logits.shape[1] != 3 or labels.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError("Expected logits [B,3,X,Y,Z] and labels [B,X,Y,Z]")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            foreground = torch.softmax(logits.float(), dim=1)[:, 1:].sum(dim=1)
            target = (labels == 1) | (labels == 2)
            lower_mass, upper_mass = terminal_distributions(foreground, axis=2)
            z_length = foreground.shape[-1]
            z = torch.arange(z_length, dtype=foreground.dtype, device=foreground.device)
            presence = lower_mass.sum(dim=-1)
            predicted_lower = (lower_mass * z).sum(dim=-1) / presence.clamp_min(self.epsilon)
            predicted_upper = (upper_mass * z).sum(dim=-1) / presence.clamp_min(self.epsilon)
            valid = target.any(dim=-1)
            section_valid = valid.sum(dim=-1) >= self.minimum_positions
            curve_valid = valid & section_valid.unsqueeze(-1)
            annotated_lower = torch.where(target, z, z_length).amin(dim=-1).float()
            annotated_upper = torch.where(target, z, -1).amax(dim=-1).float()
            lower_fit = fit_decreasing_steps(predicted_lower, curve_valid)
            upper_fit = fit_decreasing_steps(predicted_upper, curve_valid)
            lower_target_fit = fit_decreasing_steps(annotated_lower, curve_valid)
            upper_target_fit = fit_decreasing_steps(annotated_upper, curve_valid)
            height_error = 0.5 * (
                (lower_fit - lower_target_fit).abs()
                + (upper_fit - upper_target_fit).abs()
            ) / z_length
            height_loss = _section_mean(height_error, curve_valid)
            residual_error = 0.5 * (
                ((predicted_lower - lower_fit) - (annotated_lower - lower_target_fit)).abs()
                + ((predicted_upper - upper_fit) - (annotated_upper - upper_target_fit)).abs()
            ) / z_length
            residual_loss = _section_mean(residual_error, curve_valid)
            adjacent = curve_valid[..., 1:] & curve_valid[..., :-1]
            transition_error = 0.5 * (
                ((lower_fit[..., 1:] - lower_fit[..., :-1])
                 - (lower_target_fit[..., 1:] - lower_target_fit[..., :-1])).abs()
                + ((upper_fit[..., 1:] - upper_fit[..., :-1])
                   - (upper_target_fit[..., 1:] - upper_target_fit[..., :-1])).abs()
            ) / z_length
            transition_loss = _section_mean(transition_error, adjacent)
            positive_presence = (presence * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
            negative_presence = (presence * ~valid).sum(dim=(1, 2)) / (~valid).sum(dim=(1, 2)).clamp_min(1)
            presence_loss = 1.0 - positive_presence + negative_presence
            case_loss = (
                height_loss
                + self.residual_weight * residual_loss
                + self.transition_weight * transition_loss
                + self.presence_weight * presence_loss
            )
            loss = case_loss.mean()
        truth = torch.exp(-case_loss).clamp(0.0, 1.0)
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=torch.stack((height_loss, residual_loss, transition_loss, presence_loss), dim=1),
            details={
                "case_loss": case_loss,
                "height_loss": height_loss,
                "residual_loss": residual_loss,
                "transition_loss": transition_loss,
                "presence_loss": presence_loss,
                "sections_fitted": section_valid.sum(dim=1),
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= 0.90,
                "metrics": {"raw_loss": case_loss},
            },
        )
