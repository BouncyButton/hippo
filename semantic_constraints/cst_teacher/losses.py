"""Training objectives and SwinUNETR-facing CST constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .descriptors import sampled_slice_profiles, soft_descriptors


DEFAULT_DESCRIPTOR_SCALES = (0.10, 0.05, 0.10, 1.00)


@dataclass
class ConditionalConstraintResult:
    loss: torch.Tensor
    descriptor_values: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    normalized_violation: torch.Tensor


def descriptor_quantile_loss(
    predicted_quantiles: torch.Tensor,
    targets: torch.Tensor,
    *,
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
    scales: tuple[float, ...] = DEFAULT_DESCRIPTOR_SCALES,
) -> torch.Tensor:
    """Pinball loss for ordered ``[lower, median, upper]`` predictions."""

    if predicted_quantiles.ndim != 3 or predicted_quantiles.shape[-1] != len(quantiles):
        raise ValueError("predicted_quantiles must have shape [B, D, 3]")
    if targets.shape != predicted_quantiles.shape[:2]:
        raise ValueError("targets must have shape [B, D]")
    if len(scales) != targets.shape[1]:
        raise ValueError("one descriptor scale is required per target")
    scale = targets.new_tensor(scales).view(1, -1, 1)
    error = (targets.unsqueeze(-1) - predicted_quantiles) / scale
    q = targets.new_tensor(quantiles).view(1, 1, -1)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def slice_profile_loss(
    predicted_profiles: torch.Tensor,
    target_profiles: torch.Tensor,
    valid_elements: torch.Tensor | None = None,
    *,
    loss_kind: str = "smooth_l1",
) -> torch.Tensor:
    if predicted_profiles.shape != target_profiles.shape or predicted_profiles.ndim != 3:
        raise ValueError("slice profiles must have matching shape [B, N, 2]")
    if loss_kind == "smooth_l1":
        pointwise = F.smooth_l1_loss(predicted_profiles, target_profiles, reduction="none")
    elif loss_kind == "l1":
        pointwise = F.l1_loss(predicted_profiles, target_profiles, reduction="none")
    else:
        raise ValueError("loss_kind must be 'smooth_l1' or 'l1'")
    per_element = pointwise.mean(dim=2)
    if valid_elements is None:
        return per_element.mean()
    if valid_elements.shape != per_element.shape:
        raise ValueError("valid_elements must have shape [B, N]")
    weights = valid_elements.to(per_element.dtype)
    return (per_element * weights).sum() / weights.sum().clamp_min(1.0)


def anomaly_detection_loss(
    logits: torch.Tensor,
    anomaly_targets: torch.Tensor,
    valid_elements: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != anomaly_targets.shape or logits.ndim != 2:
        raise ValueError("anomaly logits and targets must have shape [B, N]")
    per_element = F.binary_cross_entropy_with_logits(logits, anomaly_targets.float(), reduction="none")
    if valid_elements is None:
        return per_element.mean()
    if valid_elements.shape != logits.shape:
        raise ValueError("valid_elements must have shape [B, N]")
    weights = valid_elements.to(per_element.dtype)
    return (per_element * weights).sum() / weights.sum().clamp_min(1.0)


def conditional_descriptor_constraint(
    probabilities: torch.Tensor,
    teacher_quantiles: torch.Tensor,
    *,
    reliability: torch.Tensor | None = None,
    minimum_scale: float = 0.02,
) -> ConditionalConstraintResult:
    """Penalize descriptors outside MRI-conditioned teacher intervals.

    Teacher outputs are detached by design: the constraint updates the
    segmentation probabilities, never the teacher or its predicted bounds.
    """

    if teacher_quantiles.ndim != 3 or teacher_quantiles.shape[-1] != 3:
        raise ValueError("teacher_quantiles must have shape [B, D, 3]")
    values = soft_descriptors(probabilities)
    if values.shape != teacher_quantiles.shape[:2]:
        raise ValueError("teacher descriptor count does not match the constraint descriptors")
    bounds = teacher_quantiles.detach()
    lower, upper = bounds[..., 0], bounds[..., 2]
    if not torch.isfinite(bounds).all() or torch.any(lower > upper):
        raise ValueError("teacher intervals must be finite and ordered")
    scale = ((upper - lower) / 2.0).clamp_min(minimum_scale)
    violation = F.relu(lower - values) + F.relu(values - upper)
    normalized = violation / scale
    per_case = normalized.square().mean(dim=1)
    if reliability is not None:
        if reliability.shape != per_case.shape:
            raise ValueError("reliability must have shape [B]")
        per_case = per_case * reliability.detach().clamp(0.0, 1.0)
    return ConditionalConstraintResult(
        loss=per_case.mean(),
        descriptor_values=values,
        lower_bounds=lower,
        upper_bounds=upper,
        normalized_violation=normalized,
    )


def contextual_profile_constraint(
    probabilities: torch.Tensor,
    positions: torch.Tensor,
    teacher_profiles: torch.Tensor,
    *,
    valid_elements: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match SwinUNETR slice areas to MRI-only CST profile predictions."""

    current = sampled_slice_profiles(probabilities, positions)
    return slice_profile_loss(current, teacher_profiles.detach(), valid_elements)


def anomaly_constraint(
    anomaly_teacher: nn.Module,
    image_slabs: torch.Tensor,
    probability_slices: torch.Tensor,
    metadata: torch.Tensor,
    *,
    valid_elements: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage a frozen anomaly teacher to classify soft masks as valid.

    ``no_grad`` is intentionally not used because gradients must pass through
    ``probability_slices``.  Call ``requires_grad_(False)`` on the teacher
    before using this function in segmentation training.
    """

    if any(parameter.requires_grad for parameter in anomaly_teacher.parameters()):
        raise ValueError("freeze anomaly_teacher parameters before using it as a constraint")
    logits, _ = anomaly_teacher(image_slabs, probability_slices, metadata, valid_elements)
    targets = torch.zeros_like(logits)
    return anomaly_detection_loss(logits, targets, valid_elements)
