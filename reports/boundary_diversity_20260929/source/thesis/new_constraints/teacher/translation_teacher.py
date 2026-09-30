"""Distil inverse-mapped translation predictions into the unshifted student."""

import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraint_result import ConstraintResult
from ..equivariance import (
    Shift3D,
    restore_translation,
    translate_3d,
    translation_valid_mask,
)


DEFAULT_TEACHER_SHIFTS: tuple[Shift3D, ...] = tuple(
    tuple(amount if axis == index else 0 for axis in range(3))
    for index in range(3)
    for amount in (1, -1, 2, -2)
)

TEACHER_METRICS = ("raw_loss", "kl_unscaled", "valid_fraction", "mean_valid_views")


@dataclass(frozen=True)
class AlignedTeacher:
    """Detached probabilities and coverage in the student's coordinate system."""

    probabilities: torch.Tensor  # [B, C, X, Y, Z]; zero outside valid_mask
    valid_mask: torch.Tensor  # [B, 1, X, Y, Z]
    view_counts: torch.Tensor  # [B, 1, X, Y, Z]
    shifts: tuple[Shift3D, ...]
    temperature: float
    support: str = "union"


def _validated_shifts(shifts: Sequence[Shift3D]) -> tuple[Shift3D, ...]:
    if not shifts:
        raise ValueError("At least one translation must be provided.")
    if any(
        len(shift) != 3
        or any(isinstance(amount, bool) or not isinstance(amount, Integral) for amount in shift)
        for shift in shifts
    ):
        raise ValueError("Every shift must contain exactly three integers.")
    result = tuple(tuple(int(amount) for amount in shift) for shift in shifts)
    if len(set(result)) != len(result):
        raise ValueError("Translations must be distinct.")
    return result


@contextmanager
def _teacher_evaluation(model: nn.Module, images: torch.Tensor):
    """Keep extra teacher forwards out of the student's mode, buffers, and RNG."""

    modules = tuple(model.modules())
    training_flags = tuple(module.training for module in modules)
    tensors = (images, *model.parameters(), *model.buffers())
    cuda_devices = sorted(
        {tensor.device.index for tensor in tensors if tensor.device.type == "cuda"}
    )
    uses_mps = any(tensor.device.type == "mps" for tensor in tensors)
    mps_rng_state = torch.mps.get_rng_state() if uses_mps else None
    # Standard evaluation layers do not update buffers. Keep a snapshot for
    # custom layers too, and avoid copy_ on unchanged buffers (autograd versions).
    buffers = tuple((buffer, buffer.detach().clone()) for buffer in model.buffers())
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                if not torch.equal(buffer, original):
                    buffer.copy_(original)
        for module, training in zip(modules, training_flags):
            module.training = training
        if mps_rng_state is not None:
            torch.mps.set_rng_state(mps_rng_state)


class TranslationTeacherKLLoss(nn.Module):
    """Patient-balanced KL(q || p), with a detached translated self-teacher.

    This is consistency regularization, not an independent source of truth.
    A confidently wrong, translation-invariant prediction is a fixed point.
    """

    def __init__(
        self,
        shifts: Sequence[Shift3D] = DEFAULT_TEACHER_SHIFTS,
        num_views: int = 2,
        temperature: float = 1.0,
        confidence_agreement_threshold: float = 0.90,
        class_ids: Sequence[int] | None = None,
        support: str = "union",
    ) -> None:
        super().__init__()
        self.shifts = _validated_shifts(shifts)
        if (0, 0, 0) in self.shifts:
            raise ValueError("The sampling set must contain nonzero translations.")
        if isinstance(num_views, bool) or not isinstance(num_views, Integral):
            raise ValueError("num_views must be an integer.")
        if not 1 <= num_views <= len(self.shifts):
            raise ValueError("num_views must be between one and the number of shifts.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not math.isfinite(confidence_agreement_threshold) or not 0 <= confidence_agreement_threshold <= 1:
            raise ValueError("confidence_agreement_threshold must be between zero and one.")
        if class_ids is not None and (
            not class_ids
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in class_ids)
            or len(set(class_ids)) != len(class_ids)
        ):
            raise ValueError("class_ids must be distinct non-negative integers.")
        self.num_views = int(num_views)
        self.temperature = float(temperature)
        self.confidence_agreement_threshold = float(confidence_agreement_threshold)
        self.class_ids = tuple(class_ids) if class_ids is not None else None
        if support not in {"union", "common"}:
            raise ValueError("support must be union or common.")
        self.support = support

    def select_shifts(
        self,
        *,
        shifts: Sequence[Shift3D] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Shift3D, ...]:
        """Use explicit views or sample K distinct views from an independent RNG."""

        if shifts is not None:
            return _validated_shifts(shifts)
        if generator is None or generator.device.type != "cpu":
            raise ValueError("Sampled teacher shifts require an independent CPU generator.")
        indices = torch.randperm(len(self.shifts), generator=generator)[:self.num_views]
        return tuple(self.shifts[index] for index in indices.tolist())

    @torch.no_grad()
    def build_from_aligned(
        self,
        aligned_probabilities: Sequence[torch.Tensor],
        valid_masks: Sequence[torch.Tensor],
        *,
        shifts: Sequence[Shift3D] | None = None,
    ) -> AlignedTeacher:
        """Average already aligned, temperature-softened probabilities by coverage.

        Each mask is binary and broadcasts to [B, 1, X, Y, Z]. Invalid
        probability entries are ignored. Use this helper for frozen-logit audits.
        """

        if not aligned_probabilities or len(aligned_probabilities) != len(valid_masks):
            raise ValueError("Provide one valid mask per nonempty probability view.")
        first = aligned_probabilities[0]
        if first.ndim != 5 or first.shape[0] < 1 or first.shape[1] < 2:
            raise ValueError("Teacher probabilities must have shape [B, C>=2, X, Y, Z].")
        selected_shifts = _validated_shifts(shifts) if shifts is not None else ()
        if selected_shifts and len(selected_shifts) != len(aligned_probabilities):
            raise ValueError("Provide one shift per probability view.")
        probability_sum = torch.zeros_like(first, dtype=torch.float32)
        counts = torch.zeros_like(probability_sum[:, :1])
        for probabilities, valid in zip(aligned_probabilities, valid_masks):
            if probabilities.shape != first.shape or probabilities.device != first.device:
                raise ValueError("All teacher probability views must share shape and device.")
            if valid.device != first.device:
                raise ValueError("Valid masks must be on the probability device.")
            try:
                valid = torch.broadcast_to(valid, counts.shape).float()
            except RuntimeError as error:
                raise ValueError("A valid mask must broadcast to [B, 1, X, Y, Z].") from error
            if not torch.all((valid == 0) | (valid == 1)):
                raise ValueError("Valid masks must contain only zero or one.")
            masked = torch.where(valid.bool(), probabilities.float(), 0.0)
            if not torch.isfinite(masked).all() or (masked < 0).any():
                raise ValueError("Valid teacher probabilities must be finite and non-negative.")
            if not torch.allclose(masked.sum(dim=1, keepdim=True), valid, atol=1e-5, rtol=1e-5):
                raise ValueError("Valid teacher probabilities must sum to one over classes.")
            probability_sum.add_(masked)
            counts.add_(valid)
        valid_mask = counts > 0
        if self.support == "common":
            if not selected_shifts or not set(selected_shifts).issubset(self.shifts):
                raise ValueError("Common support requires explicit shifts from the configured set.")
            # The support depends on the entire sampling set, never the draw.
            common = torch.ones_like(counts, dtype=torch.bool)
            for shift in self.shifts:
                common &= translation_valid_mask(first.shape[-3:], shift, device=first.device).bool()
            if ((counts != len(selected_shifts)) & common).any():
                raise ValueError("Every sampled view must cover the entire common support.")
            valid_mask = common
            probability_sum = torch.where(common, probability_sum, 0.0)
            counts = torch.where(common, counts, 0.0)
        if (valid_mask.sum(dim=(1, 2, 3, 4)) == 0).any():
            raise ValueError("Every case must have at least one valid teacher voxel.")
        return AlignedTeacher(
            probabilities=(probability_sum / counts.clamp_min(1.0)).detach(),
            valid_mask=valid_mask.detach(),
            view_counts=counts.detach(),
            shifts=selected_shifts,
            temperature=self.temperature,
            support=self.support,
        )

    @torch.no_grad()
    def build_from_cached(
        self,
        transformed_logits: Sequence[torch.Tensor],
        shifts: Sequence[Shift3D],
    ) -> AlignedTeacher:
        """Inverse-map cached shifted logits and average their valid probabilities."""

        selected_shifts = _validated_shifts(shifts)
        if len(transformed_logits) != len(selected_shifts):
            raise ValueError("Provide one cached logit tensor per shift.")
        aligned = []
        masks = []
        for logits, shift in zip(transformed_logits, selected_shifts):
            if logits.ndim != 5:
                raise ValueError("Cached logits must have shape [B, C, X, Y, Z].")
            probabilities = F.softmax(logits.float() / self.temperature, dim=1)
            aligned.append(restore_translation(probabilities, shift))
            masks.append(translation_valid_mask(logits.shape[-3:], shift, device=logits.device))
        return self.build_from_aligned(aligned, masks, shifts=selected_shifts)

    def build_teacher(
        self,
        model: nn.Module,
        images: torch.Tensor,
        *,
        shifts: Sequence[Shift3D] | None = None,
        generator: torch.Generator | None = None,
    ) -> AlignedTeacher:
        """Evaluate the same model without gradients, stochastic layers, or RNG drift."""

        selected_shifts = self.select_shifts(shifts=shifts, generator=generator)
        with _teacher_evaluation(model, images):
            logits = [model(translate_3d(images, shift)) for shift in selected_shifts]
            return self.build_from_cached(logits, selected_shifts)

    def loss_from_teacher(
        self,
        base_logits: torch.Tensor,
        teacher: AlignedTeacher,
    ) -> ConstraintResult:
        """Evaluate a frozen teacher; only the supplied student logits receive gradients."""

        if base_logits.shape != teacher.probabilities.shape or base_logits.device != teacher.probabilities.device:
            raise ValueError("Student logits and teacher probabilities must share shape and device.")
        if teacher.temperature != self.temperature:
            raise ValueError("The cached teacher must use the same temperature as the loss.")
        if teacher.support != self.support:
            raise ValueError("The cached teacher must use the same support as the loss.")
        log_probabilities = F.log_softmax(base_logits.float() / self.temperature, dim=1)
        target = teacher.probabilities.detach()
        valid = teacher.valid_mask.detach().squeeze(1)
        voxel_kl = F.kl_div(log_probabilities, target, reduction="none").sum(dim=1)
        dimensions = tuple(range(1, voxel_kl.ndim))
        denominator = valid.sum(dim=dimensions).to(torch.float32)
        # Equal case weighting, explicit mean over voxels with at least one view.
        case_kl_unscaled = (voxel_kl * valid).sum(dim=dimensions) / denominator
        case_kl = case_kl_unscaled.clamp_min(0.0) * self.temperature**2
        truth = torch.exp(-case_kl)

        # Compatibility diagnostic: the existing linear foreground soft Dice.
        # Unlike KL truth it is confidence-sensitive, so report it separately.
        class_ids = self.class_ids or tuple(range(1, base_logits.shape[1]))
        if max(class_ids) >= base_logits.shape[1]:
            raise ValueError("class_ids are incompatible with the model output.")
        with torch.no_grad():
            student = log_probabilities.exp()[:, class_ids] * valid.unsqueeze(1)
            selected_teacher = target[:, class_ids] * valid.unsqueeze(1)
            spatial_dimensions = (2, 3, 4)
            agreement_denominator = (student + selected_teacher).sum(spatial_dimensions)
            classwise_agreement = (
                2.0 * (student * selected_teacher).sum(spatial_dimensions)
                / agreement_denominator.clamp_min(1e-6)
            )
            agreement = classwise_agreement.mean(dim=1).clamp(0.0, 1.0)
        metrics = {
            "raw_loss": case_kl.detach(),
            "kl_unscaled": case_kl_unscaled.detach(),
            "valid_fraction": valid.float().mean(dim=dimensions).detach(),
            "mean_valid_views": teacher.view_counts.float().mean(dim=(1, 2, 3, 4)).detach(),
        }
        return ConstraintResult(
            loss=case_kl.mean(),
            truth=truth,
            value=case_kl,
            details={
                "shifts": teacher.shifts,
                "temperature": self.temperature,
                "support": self.support,
                "kl_unscaled": case_kl_unscaled.detach(),
                "valid_voxels": denominator.detach(),
                "valid_fraction": valid.float().mean().detach(),
                "mean_valid_views": teacher.view_counts.float().mean().detach(),
                "confidence_weighted_agreement": agreement,
                "confidence_adherent": agreement >= self.confidence_agreement_threshold,
                "metrics": metrics,
            },
        )

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        *,
        shifts: Sequence[Shift3D] | None = None,
        generator: torch.Generator | None = None,
    ) -> ConstraintResult:
        if base_logits is None:
            base_logits = model(images)
        teacher = self.build_teacher(model, images, shifts=shifts, generator=generator)
        return self.loss_from_teacher(base_logits, teacher)
