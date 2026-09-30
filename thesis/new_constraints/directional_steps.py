"""Fit patient-specific outer step functions along the three voxel axes.

For each oriented ray, the reference mask defines a binary step at its first
foreground voxel. A reverse ray defines the opposite step. The model's soft
steps are differentiable cumulative foreground-hit probabilities. Matching the
whole steps couples errors along each ray; crossing their interval masks adds
a voxel-level comparison of the three axis views.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn

from .constraint_result import ConstraintResult, differentiable_zero


def _labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if logits.ndim != 5 or labels.ndim != 4 or labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Expected logits [B,3,X,Y,Z] and labels [B,X,Y,Z].")
    if logits.shape[1] != 3:
        raise ValueError("Expected background, anterior, and posterior logits.")
    return labels.long()


def terminal_distributions(probability: torch.Tensor, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lowest/highest FG crossing masses with the selected ray axis last.

    For independent Bernoulli occupancy along a ray, the mass at position k is
    p[k] times the probability that all positions before/after k are empty.
    Both masses sum to the probability that the ray contains any foreground.
    """
    if probability.ndim != 4 or axis not in (0, 1, 2):
        raise ValueError("Expected probability [B,X,Y,Z] and a spatial axis 0, 1, or 2.")
    p = probability.movedim(axis + 1, -1)
    empty = 1.0 - p
    ones = torch.ones_like(empty[..., :1])
    below_empty = torch.cat((ones, empty.cumprod(dim=-1)[..., :-1]), dim=-1)
    reversed_empty = empty.flip(-1)
    above_empty = torch.cat((ones, reversed_empty.cumprod(dim=-1)[..., :-1]), dim=-1).flip(-1)
    return p * below_empty, p * above_empty


def directional_step_maps(
    probability: torch.Tensor, axis: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft first-hit steps from both ends of each ray, in original voxel order."""
    if probability.ndim != 4 or axis not in (0, 1, 2):
        raise ValueError("Expected probability [B,X,Y,Z] and a spatial axis 0, 1, or 2.")
    ray = probability.movedim(axis + 1, -1)
    empty = 1.0 - ray
    forward = 1.0 - empty.cumprod(dim=-1)
    backward = 1.0 - empty.flip(-1).cumprod(dim=-1).flip(-1)
    return forward.movedim(-1, axis + 1), backward.movedim(-1, axis + 1)


def _mean_valid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, values.ndim))
    return (values * valid).sum(dim=dimensions) / valid.sum(dim=dimensions).clamp_min(1)


class DirectionalStepSurfaceLoss(nn.Module):
    """Match six patient-specific voxelwise steps and their crossed axis views.

    `axes=(0, 1, 2)` uses all six orthogonal outer directions. The optional
    long-axis order term acts only on z-ray endpoints indexed by y. It is off
    by default because the patient's actual contour is the primary target.
    The foreground union cannot supervise A/P class swaps; combine with a
    segmentation objective that supervises both classes.
    """

    def __init__(
        self,
        *,
        axes: Sequence[int] = (0, 1, 2),
        order_weight: float = 0.0,
        presence_weight: float = 0.25,
        cross_weight: float = 0.5,
        order_tolerance_voxels: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        self.axes = tuple(int(axis) for axis in axes)
        if not self.axes or len(set(self.axes)) != len(self.axes) or any(axis not in (0, 1, 2) for axis in self.axes):
            raise ValueError("axes must be a nonempty unique subset of 0, 1, 2")
        for name, value in (
            ("order_weight", order_weight),
            ("presence_weight", presence_weight),
            ("cross_weight", cross_weight),
            ("order_tolerance_voxels", order_tolerance_voxels),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.order_weight = float(order_weight)
        self.presence_weight = float(presence_weight)
        self.cross_weight = float(cross_weight)
        self.order_tolerance_voxels = float(order_tolerance_voxels)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        labels = _labels(labels, logits)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability = torch.softmax(logits.float(), dim=1)[:, 1:].sum(dim=1)
            target = (labels == 1) | (labels == 2)
            axis_losses = []
            predicted_intervals = []
            target_intervals = []
            order_loss = differentiable_zero(probability)
            for axis in self.axes:
                forward, backward = directional_step_maps(probability, axis)
                target_forward = target.cummax(dim=axis + 1).values.float()
                target_backward = target.flip(axis + 1).cummax(dim=axis + 1).values.flip(axis + 1).float()
                predicted_intervals.append(forward * backward)
                target_intervals.append(target_forward * target_backward)
                ray_error = 0.5 * (
                    (forward - target_forward).abs() + (backward - target_backward).abs()
                ).movedim(axis + 1, -1).mean(dim=-1)
                target_rays = target.movedim(axis + 1, -1)
                valid = target_rays.any(dim=-1)
                axis_losses.append(
                    _mean_valid(ray_error, valid)
                    + self.presence_weight * _mean_valid(ray_error, ~valid)
                )

                if axis == 2 and self.order_weight > 0:
                    low_mass, high_mass = terminal_distributions(probability, axis)
                    ray_length = low_mass.shape[-1]
                    coordinate = torch.arange(ray_length, device=logits.device, dtype=probability.dtype)
                    coordinate = coordinate.reshape((1,) * (low_mass.ndim - 1) + (ray_length,))
                    target_low = torch.where(target_rays, coordinate, ray_length).amin(dim=-1)
                    target_high = torch.where(target_rays, coordinate, -1).amax(dim=-1)
                    present = high_mass.sum(dim=-1)
                    expected_high = (high_mass * coordinate).sum(dim=-1) / present.clamp_min(self.epsilon)
                    expected_low = (low_mass * coordinate).sum(dim=-1) / present.clamp_min(self.epsilon)
                    valid_pair = valid[:, :, :-1] & valid[:, :, 1:]
                    high_allowance = (
                        target_high[:, :, 1:] - target_high[:, :, :-1]
                    ).clamp_min(0) + self.order_tolerance_voxels
                    low_allowance = (
                        target_low[:, :, 1:] - target_low[:, :, :-1]
                    ).clamp_min(0) + self.order_tolerance_voxels
                    high_rise = (expected_high[:, :, 1:] - expected_high[:, :, :-1] - high_allowance).relu()
                    low_rise = (expected_low[:, :, 1:] - expected_low[:, :, :-1] - low_allowance).relu()
                    order_loss = _mean_valid(
                        0.5 * (high_rise.square() + low_rise.square()) / ray_length,
                        valid_pair,
                    )

            axis_component = torch.stack(axis_losses, dim=1)
            predicted_stack = torch.stack(predicted_intervals, dim=1)
            target_stack = torch.stack(target_intervals, dim=1)
            predicted_any = 1.0 - (1.0 - predicted_stack).prod(dim=1)
            predicted_all = predicted_stack.prod(dim=1)
            target_any = 1.0 - (1.0 - target_stack).prod(dim=1)
            target_all = target_stack.prod(dim=1)
            cross_error = 0.5 * (
                (predicted_any - target_any).abs() + (predicted_all - target_all).abs()
            )
            cross_support = (target_any > 0.5) | (predicted_any.detach() > 0.5)
            cross_loss = _mean_valid(cross_error, cross_support)
            case_loss = (
                axis_component.mean(dim=1)
                + self.cross_weight * cross_loss
                + self.order_weight * order_loss
            )
            loss = case_loss.mean()
        truth = torch.exp(-case_loss).clamp(0.0, 1.0)
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=axis_component,
            details={
                "case_loss": case_loss,
                "axis_loss": axis_component,
                "cross_axis_loss": cross_loss,
                "cross_axis_disagreement": _mean_valid(
                    predicted_stack.var(dim=1, unbiased=False), cross_support
                ),
                "longitudinal_order_loss": order_loss,
                "axes": self.axes,
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= 0.90,
                "metrics": {"raw_loss": case_loss},
            },
        )
