"""Differentiable classwise perimeter profiles over orthogonal 2-D slices."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constraint_result import ConstraintResult


def _normalise_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if labels.ndim == logits.ndim and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4 or logits.ndim != 5:
        raise ValueError("Expected logits [B,C,D,H,W] and labels [B,D,H,W].")
    if labels.shape[0] != logits.shape[0] or labels.shape[1:] != logits.shape[2:]:
        raise ValueError("Labels and logits must have matching batch and spatial shapes.")
    return labels.long()


def soft_morphological_contour(slices: torch.Tensor) -> torch.Tensor:
    """Return El Jurdi et al.'s erosion/opening contour for [N,C,H,W]."""

    if slices.ndim != 4:
        raise ValueError("slices must have shape [N,C,H,W].")
    eroded = -F.max_pool2d(-slices, kernel_size=3, stride=1, padding=1)
    opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
    return F.relu(opened - eroded)


def perimeter_profile(values: torch.Tensor, axis: int) -> torch.Tensor:
    """Return per-slice contour sums [B,C,S] along one spatial axis."""

    if values.ndim != 5:
        raise ValueError("values must have shape [B,C,D,H,W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    batch, channels = values.shape[:2]
    slice_dimension = axis + 2
    remaining = [dimension for dimension in (2, 3, 4) if dimension != slice_dimension]
    arranged = values.permute(0, slice_dimension, 1, *remaining).contiguous()
    slices = arranged.reshape(-1, channels, values.shape[remaining[0]], values.shape[remaining[1]])
    profile = soft_morphological_contour(slices).sum(dim=(-2, -1))
    return profile.reshape(batch, values.shape[slice_dimension], channels).transpose(1, 2)


class PerimeterProfileLoss(nn.Module):
    """Match classwise contour length on every slice of selected spatial axes.

    This retains the released paper's squared perimeter-difference divided by
    slice area, but avoids collapsing a whole 3-D object into one scalar.
    """

    def __init__(
        self,
        *,
        class_ids: Sequence[int] = (1, 2),
        axes: Sequence[int] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.class_ids = tuple(int(value) for value in class_ids)
        self.axes = tuple(int(value) for value in axes)
        if not self.class_ids or len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("class_ids must be nonempty and unique.")
        if not self.axes or len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must be a nonempty unique subset of 0, 1, 2.")
        if any(axis not in (0, 1, 2) for axis in self.axes):
            raise ValueError("axes must contain only 0, 1, and 2.")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        labels = _normalise_labels(labels, logits)
        if min(self.class_ids) < 0 or max(self.class_ids) >= logits.shape[1]:
            raise ValueError("class_ids are incompatible with logits.")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = torch.softmax(logits.float(), dim=1)[:, self.class_ids]
            target = F.one_hot(labels, num_classes=logits.shape[1]).movedim(-1, 1).float()
            target = target[:, self.class_ids]
            axis_losses = []
            for axis in self.axes:
                predicted_profile = perimeter_profile(probabilities, axis)
                target_profile = perimeter_profile(target, axis)
                in_plane = [size for index, size in enumerate(logits.shape[2:]) if index != axis]
                slice_area = float(in_plane[0] * in_plane[1])
                axis_losses.append((predicted_profile - target_profile).square().mean(dim=(1, 2)) / slice_area)
            per_axis = torch.stack(axis_losses, dim=1)
            case_loss = per_axis.mean(dim=1)
            loss = case_loss.mean()
        truth = torch.exp(-case_loss).clamp(0.0, 1.0)
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=per_axis,
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= 0.90,
                "case_loss": case_loss,
                "axis_loss": per_axis,
                "metrics": {"raw_loss": case_loss},
            },
        )
