"""Differentiable orthogonal ray-moment constraints for 3-D segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constraint_result import ConstraintResult


DEFAULT_CLASS_GROUPS = ((1, 2), (1,), (2,))
DEFAULT_GROUP_NAMES = ("foreground", "anterior", "posterior")


def _normalise_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if labels.ndim == logits.ndim and labels.shape[1] == 1:
        labels = labels[:, 0]
    if logits.ndim != 5 or labels.ndim != 4:
        raise ValueError("Expected logits [B,C,D,H,W] and labels [B,D,H,W].")
    if labels.shape[0] != logits.shape[0] or labels.shape[1:] != logits.shape[2:]:
        raise ValueError("Labels and logits must have matching batch and spatial shapes.")
    return labels.long()


def ray_moment_map(values: torch.Tensor, axis: int, order: int) -> torch.Tensor:
    """Return normalized per-ray moments after reducing one spatial axis.

    ``values`` has shape ``[B,C,D,H,W]``. Order zero is class thickness divided
    by ray length. Higher moments use coordinates in ``[-1, 1]`` and retain the
    same ray-length normalization.
    """

    if values.ndim != 5:
        raise ValueError("values must have shape [B,C,D,H,W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2.")
    tensor_axis = axis + 2
    ray_length = values.shape[tensor_axis]
    if order == 0:
        return values.sum(dim=tensor_axis) / float(ray_length)
    coordinate_shape = [1] * values.ndim
    coordinate_shape[tensor_axis] = ray_length
    coordinate = torch.linspace(
        -1.0,
        1.0,
        ray_length,
        device=values.device,
        dtype=values.dtype,
    ).reshape(coordinate_shape)
    weighted = values * coordinate.pow(order)
    return weighted.sum(dim=tensor_axis) / float(ray_length)


class RayMomentLoss(nn.Module):
    """Match class thickness and optional depth moments in three orthogonal views.

    Class groups are averaged equally. The default includes the foreground union
    plus each semantic hippocampal class, preventing a foreground-only loss from
    being blind to anterior/posterior swaps.
    """

    def __init__(
        self,
        *,
        orders: Sequence[int] = (0, 1),
        axes: Sequence[int] = (0, 1, 2),
        class_groups: Sequence[Sequence[int]] = DEFAULT_CLASS_GROUPS,
        group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
        moment_weights: Sequence[float] | None = None,
        smooth_l1_beta: float = 0.02,
    ) -> None:
        super().__init__()
        self.orders = tuple(int(order) for order in orders)
        self.axes = tuple(int(axis) for axis in axes)
        self.class_groups = tuple(tuple(int(class_id) for class_id in group) for group in class_groups)
        self.group_names = tuple(str(name) for name in group_names)
        if not self.orders or len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must be nonempty and unique.")
        if any(order not in (0, 1, 2) for order in self.orders):
            raise ValueError("orders must contain only 0, 1, and 2.")
        if not self.axes or len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must be nonempty and unique.")
        if any(axis not in (0, 1, 2) for axis in self.axes):
            raise ValueError("axes must contain only 0, 1, and 2.")
        if not self.class_groups or any(not group for group in self.class_groups):
            raise ValueError("class_groups must contain nonempty groups.")
        if len(self.group_names) != len(self.class_groups):
            raise ValueError("group_names must match class_groups.")
        if len(set(self.group_names)) != len(self.group_names):
            raise ValueError("group_names must be unique.")
        weights = (1.0,) * len(self.orders) if moment_weights is None else tuple(float(v) for v in moment_weights)
        if len(weights) != len(self.orders) or any(value <= 0 for value in weights):
            raise ValueError("moment_weights must contain one positive value per order.")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive.")
        self.register_buffer("moment_weights", torch.tensor(weights, dtype=torch.float32))
        self.smooth_l1_beta = float(smooth_l1_beta)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        labels = _normalise_labels(labels, logits)
        maximum_class = max(class_id for group in self.class_groups for class_id in group)
        minimum_class = min(class_id for group in self.class_groups for class_id in group)
        if minimum_class < 0 or maximum_class >= logits.shape[1]:
            raise ValueError("class_groups are incompatible with the logit channels.")

        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = torch.softmax(logits.float(), dim=1)
            one_hot = F.one_hot(labels, num_classes=logits.shape[1]).movedim(-1, 1).float()
            predicted_groups = torch.stack(
                [probabilities[:, group].sum(dim=1) for group in self.class_groups],
                dim=1,
            )
            target_groups = torch.stack(
                [one_hot[:, group].sum(dim=1) for group in self.class_groups],
                dim=1,
            )

            axis_components = []
            for axis in self.axes:
                order_components = []
                for order in self.orders:
                    predicted = ray_moment_map(predicted_groups, axis, order)
                    target = ray_moment_map(target_groups, axis, order)
                    component = F.smooth_l1_loss(
                        predicted,
                        target,
                        reduction="none",
                        beta=self.smooth_l1_beta,
                    ).flatten(start_dim=2).mean(dim=2)
                    order_components.append(component)
                axis_components.append(torch.stack(order_components, dim=2))

            # [B, axis, group, order]
            component_loss = torch.stack(axis_components, dim=1)
            weights = self.moment_weights.to(component_loss).reshape(1, 1, 1, -1)
            weighted = component_loss * weights
            case_loss = weighted.sum(dim=3).div(weights.sum()).mean(dim=(1, 2))
            loss = case_loss.mean()

        truth = torch.exp(-case_loss).clamp(0.0, 1.0)
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=component_loss,
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= 0.90,
                "case_loss": case_loss,
                "component_loss": component_loss,
                "axis_loss": weighted.sum(dim=3).div(weights.sum()).mean(dim=2),
                "group_loss": weighted.sum(dim=3).div(weights.sum()).mean(dim=1),
                "order_loss": component_loss.mean(dim=(1, 2)),
                "axes": self.axes,
                "orders": self.orders,
                "group_names": self.group_names,
                "metrics": {"raw_loss": case_loss},
            },
        )
