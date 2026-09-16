"""Differentiable consistency under small, exact 3-D translations."""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from ..constraint_result import ConstraintResult

Shift3D = tuple[int, int, int]


def _axis_slices(length: int, shift: int) -> tuple[slice, slice]:
    if abs(shift) >= length:
        raise ValueError(f"Shift {shift} is not smaller than axis length {length}.")
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    return slice(-shift, length), slice(0, length + shift)


def translate_3d(tensor: torch.Tensor, shift: Shift3D) -> torch.Tensor:
    """Translate the last three dimensions with zero padding."""

    if tensor.ndim < 3:
        raise ValueError("Expected a tensor with at least three spatial dimensions.")
    if len(shift) != 3:
        raise ValueError("shift must contain (dx, dy, dz).")

    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for length, amount in zip(tensor.shape[-3:], shift):
        source, destination = _axis_slices(length, amount)
        source_slices.append(source)
        destination_slices.append(destination)

    translated = torch.zeros_like(tensor)
    prefix = (slice(None),) * (tensor.ndim - 3)
    translated[prefix + tuple(destination_slices)] = tensor[
        prefix + tuple(source_slices)
    ]
    return translated


def restore_translation(tensor: torch.Tensor, shift: Shift3D) -> torch.Tensor:
    """Map a translated tensor back to the original coordinate system."""

    return translate_3d(tensor, tuple(-amount for amount in shift))


def translation_valid_mask(
    spatial_shape: Sequence[int],
    shift: Shift3D,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return voxels retained by a translate-and-restore round trip."""

    if len(spatial_shape) != 3:
        raise ValueError("spatial_shape must contain exactly three axes.")
    mask = torch.ones((1, 1, *spatial_shape), device=device, dtype=dtype)
    return restore_translation(translate_3d(mask, shift), shift)


def _classwise_soft_dice(
    first: torch.Tensor,
    second: torch.Tensor,
    valid_mask: torch.Tensor,
    class_ids: Sequence[int],
    epsilon: float,
    *,
    squared_denominator: bool,
) -> torch.Tensor:
    selected_first = first[:, class_ids] * valid_mask
    selected_second = second[:, class_ids] * valid_mask
    spatial_dimensions = tuple(range(2, first.ndim))
    intersection = (selected_first * selected_second).sum(spatial_dimensions)
    if squared_denominator:
        denominator = selected_first.square().sum(
            spatial_dimensions
        ) + selected_second.square().sum(spatial_dimensions)
        return (2.0 * intersection + epsilon) / (denominator + epsilon)
    denominator = selected_first.sum(spatial_dimensions) + selected_second.sum(
        spatial_dimensions
    )
    return 2.0 * intersection / denominator.clamp_min(epsilon)


class TranslationEquivarianceLoss(nn.Module):
    """Penalize disagreement after translation and inverse mapping."""

    def __init__(
        self,
        shifts: Sequence[Shift3D] = (
            (2, 0, 0),
            (-2, 0, 0),
            (0, 2, 0),
            (0, -2, 0),
            (0, 0, 2),
            (0, 0, -2),
        ),
        class_ids: Sequence[int] = (1, 2),
        confidence_agreement_threshold: float = 0.90,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not shifts:
            raise ValueError("At least one translation must be provided.")
        if any(len(shift) != 3 for shift in shifts):
            raise ValueError("Every translation must contain (dx, dy, dz).")
        if not class_ids:
            raise ValueError("At least one class must be selected.")
        if any(class_id < 0 for class_id in class_ids):
            raise ValueError("class_ids must be non-negative.")
        if not 0.0 <= confidence_agreement_threshold <= 1.0:
            raise ValueError("confidence_agreement_threshold must be between zero and one.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        self.shifts = tuple(tuple(int(value) for value in shift) for shift in shifts)
        self.class_ids = tuple(int(class_id) for class_id in class_ids)
        self.confidence_agreement_threshold = float(confidence_agreement_threshold)
        self.epsilon = float(epsilon)

    def _sample_shift(self, generator: torch.Generator | None) -> Shift3D:
        index = torch.randint(
            len(self.shifts),
            size=(),
            generator=generator,
        ).item()
        return self.shifts[index]

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        *,
        shift: Shift3D | None = None,
        generator: torch.Generator | None = None,
        transformed_logits: torch.Tensor | None = None,
    ) -> ConstraintResult:
        """Evaluate a sampled or explicitly supplied translation.

        A caller that already evaluated the shifted supervised view may provide
        its logits. This prevents a third forward pass in the compute-matched
        augmentation-plus-equivariance arm.
        """

        if base_logits is None:
            base_logits = model(images)
        selected_shift = shift if shift is not None else self._sample_shift(generator)
        if len(selected_shift) != 3:
            raise ValueError("shift must contain (dx, dy, dz).")
        if max(self.class_ids) >= base_logits.shape[1]:
            raise ValueError(
                "The selected class_ids are incompatible with the model output."
            )

        if transformed_logits is None:
            transformed_logits = model(translate_3d(images, selected_shift))
        elif transformed_logits.shape != base_logits.shape:
            raise ValueError("transformed_logits must match base_logits shape.")
        base_probabilities = torch.softmax(base_logits, dim=1)
        restored_probabilities = restore_translation(
            torch.softmax(transformed_logits, dim=1),
            selected_shift,
        )
        valid_mask = translation_valid_mask(
            base_logits.shape[-3:],
            selected_shift,
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        optimization_classwise_dice = _classwise_soft_dice(
            base_probabilities,
            restored_probabilities,
            valid_mask,
            self.class_ids,
            self.epsilon,
            squared_denominator=True,
        )
        confidence_weighted_classwise_dice = _classwise_soft_dice(
            base_probabilities,
            restored_probabilities,
            valid_mask,
            self.class_ids,
            self.epsilon,
            squared_denominator=False,
        )
        truth = optimization_classwise_dice.mean(dim=1).clamp(0.0, 1.0)
        confidence_weighted_agreement = confidence_weighted_classwise_dice.mean(
            dim=1
        ).clamp(0.0, 1.0)
        loss = 1.0 - truth.mean()

        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=confidence_weighted_classwise_dice,
            details={
                "shift": selected_shift,
                "confidence_weighted_agreement": confidence_weighted_agreement,
                "confidence_adherent": (
                    confidence_weighted_agreement >= self.confidence_agreement_threshold
                ),
                "optimization_classwise_dice": optimization_classwise_dice,
                "valid_fraction": valid_mask.mean(),
            },
        )
