"""Balanced supervision immediately inside and outside the GT outer contour."""

import math
from collections.abc import Sequence
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraint_result import ConstraintResult, differentiable_zero


def _normalise_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if labels.ndim == logits.ndim and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != logits.ndim - 1:
        raise ValueError("labels must have shape [B, 1, X, Y, Z] or [B, X, Y, Z].")
    if labels.shape[0] != logits.shape[0] or labels.shape[-3:] != logits.shape[-3:]:
        raise ValueError("labels and logits must have matching batch and spatial shapes.")
    return labels.long()


@lru_cache(maxsize=8)
def _cross_kernel(device: torch.device) -> torch.Tensor:
    """Reuse the immutable morphology kernel on each device."""
    kernel = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = 1.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    return kernel


def _dilate_6(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    current = mask.bool()
    kernel = _cross_kernel(mask.device)
    for _ in range(iterations):
        neighbours = F.conv3d(current.float(), kernel, padding=1)
        current = neighbours > 0.0
    return current


def _erode_6(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    current = mask.bool()
    kernel = _cross_kernel(mask.device)
    for _ in range(iterations):
        neighbours = F.conv3d(current.float(), kernel, padding=1)
        current = neighbours == 7.0
    return current


@torch.no_grad()
def build_boundary_bands(
    foreground: torch.Tensor,
    *,
    steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint inner/outer 6-connected morphological bands.

    ``foreground`` must have shape ``[B, 1, X, Y, Z]``. The inner band is
    foreground removed by erosion; the outer band is background added by
    dilation.
    """

    if foreground.ndim != 5 or foreground.shape[1] != 1:
        raise ValueError("foreground must have shape [B, 1, X, Y, Z].")
    if steps < 1:
        raise ValueError("steps must be positive.")
    foreground = foreground.bool()
    eroded = _erode_6(foreground, steps)
    dilated = _dilate_6(foreground, steps)
    inner = foreground & ~eroded
    outer = dilated & ~foreground
    return inner, outer


def foreground_log_odds(
    logits: torch.Tensor,
    foreground_class_ids: Sequence[int],
    complement_class_ids: Sequence[int],
) -> torch.Tensor:
    """Return exact grouped foreground-vs-complement log odds."""

    if logits.ndim != 5:
        raise ValueError("logits must have shape [B, C, X, Y, Z].")
    foreground_ids = tuple(int(value) for value in foreground_class_ids)
    complement_ids = tuple(int(value) for value in complement_class_ids)
    if not foreground_ids or not complement_ids:
        raise ValueError("foreground and complement class IDs must both be nonempty.")
    if set(foreground_ids) & set(complement_ids):
        raise ValueError("foreground and complement class IDs must be disjoint.")
    all_ids = foreground_ids + complement_ids
    if len(set(foreground_ids)) != len(foreground_ids) or len(
        set(complement_ids)
    ) != len(complement_ids):
        raise ValueError("class ID groups cannot contain duplicates.")
    if min(all_ids) < 0 or max(all_ids) >= logits.shape[1]:
        raise ValueError("class IDs are incompatible with the model output.")
    if set(all_ids) != set(range(logits.shape[1])):
        raise ValueError(
            "foreground and complement class IDs must partition every output class."
        )
    logits32 = logits.float()
    if len(foreground_ids) == 2 and len(complement_ids) == 1:
        # The canonical three-class task has two foreground logits and one
        # background logit. Avoid generic gathers and singleton reductions.
        return torch.logaddexp(
            logits32[:, foreground_ids[0]], logits32[:, foreground_ids[1]],
        ) - logits32[:, complement_ids[0]]
    return torch.logsumexp(logits32[:, foreground_ids], dim=1) - torch.logsumexp(
        logits32[:, complement_ids], dim=1
    )


def _edge_touching(foreground: torch.Tensor) -> torch.Tensor:
    faces = (
        foreground[:, :, 0].flatten(1),
        foreground[:, :, -1].flatten(1),
        foreground[:, :, :, 0].flatten(1),
        foreground[:, :, :, -1].flatten(1),
        foreground[:, :, :, :, 0].flatten(1),
        foreground[:, :, :, :, -1].flatten(1),
    )
    return torch.cat(faces, dim=1).any(1)


def validate_degree_weighting(alpha: float, normalization: str) -> None:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("degree_alpha must be finite and non-negative.")
    if normalization not in {"inner", "surface"}:
        raise ValueError("degree_normalization must be 'inner' (A) or 'surface' (B).")


@torch.no_grad()
def inner_degree_weights(
    foreground: torch.Tensor,
    inner: torch.Tensor,
    *,
    alpha: float,
    normalization: str,
) -> torch.Tensor:
    """Return case-normalized weights [B, 1, X, Y, Z] for the inner band.

    A (inner) redistributes mass over the entire inner band. B (surface)
    redistributes only within its degree<6 surface and leaves deeper weights
    at one. Foreground must include a background halo: cropped foreground
    faces do not define anatomical exposure. No predicted mask enters weights.
    """
    validate_degree_weighting(alpha, normalization)
    if foreground.ndim != 5 or foreground.shape[1] != 1 or inner.shape != foreground.shape:
        raise ValueError("foreground and inner must have matching [B, 1, X, Y, Z] shapes.")
    if alpha == 0:
        return torch.ones_like(foreground, dtype=torch.float32)
    if bool(_edge_touching(foreground).any()):
        raise ValueError("Degree weighting requires a background halo; foreground touches the crop edge.")
    foreground_float = foreground.float()
    degree = F.conv3d(foreground_float, _cross_kernel(foreground.device), padding=1) - foreground_float
    surface = foreground.bool() & (degree < 6)
    raw = 1.0 + alpha * (6.0 - degree) / 6.0
    support = inner.bool() if normalization == "inner" else surface
    dimensions = (1, 2, 3, 4)
    count = support.sum(dimensions, keepdim=True)
    mean = (raw * support).sum(dimensions, keepdim=True) / count.clamp_min(1)
    mean = torch.where(count > 0, mean, torch.ones_like(mean))
    weights = raw / mean
    return torch.where(support, weights, torch.ones_like(weights))


class OuterBoundaryBandLoss(nn.Module):
    """Balanced focal BCE on two morphological steps around the GT contour.

    ``focal_gamma=0`` preserves the original BCE objective exactly. Positive
    gamma applies the standard focal factor ``(1 - p_t) ** gamma`` to each
    voxel before the existing side-balanced and patient-balanced reductions.
    Optional side-specific exponents support mechanistic ablations while the
    shared ``focal_gamma`` remains the backwards-compatible default.
    Degree weighting is disabled at ``degree_alpha=0``. Positive alpha uses
    whole-foreground face degree and either inner (A) or surface (B) normalization.
    """

    def __init__(
        self,
        *,
        foreground_class_ids: Sequence[int] = (1, 2),
        complement_class_ids: Sequence[int] = (0,),
        steps: int = 2,
        focal_gamma: float = 0.0,
        inner_focal_gamma: float | None = None,
        outer_focal_gamma: float | None = None,
        degree_alpha: float = 0.0,
        degree_normalization: str = "inner",
        adherence_threshold: float = 0.90,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        validate_degree_weighting(degree_alpha, degree_normalization)
        self.degree_alpha = float(degree_alpha)
        self.degree_normalization = degree_normalization
        self.foreground_class_ids = tuple(int(v) for v in foreground_class_ids)
        self.complement_class_ids = tuple(int(v) for v in complement_class_ids)
        if not self.foreground_class_ids or not self.complement_class_ids:
            raise ValueError("foreground and complement class IDs must be nonempty.")
        if set(self.foreground_class_ids) & set(self.complement_class_ids):
            raise ValueError("foreground and complement class IDs must be disjoint.")
        if steps != 2:
            raise ValueError("The canonical outer-boundary loss requires exactly 2 steps.")
        effective_inner_gamma = (
            focal_gamma if inner_focal_gamma is None else inner_focal_gamma
        )
        effective_outer_gamma = (
            focal_gamma if outer_focal_gamma is None else outer_focal_gamma
        )
        for name, gamma in (
            ("focal_gamma", focal_gamma),
            ("inner_focal_gamma", effective_inner_gamma),
            ("outer_focal_gamma", effective_outer_gamma),
        ):
            if not math.isfinite(gamma) or gamma < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0.0 <= adherence_threshold <= 1.0:
            raise ValueError("adherence_threshold must be between zero and one.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        self.steps = int(steps)
        self.focal_gamma = float(focal_gamma)
        self.inner_focal_gamma = float(effective_inner_gamma)
        self.outer_focal_gamma = float(effective_outer_gamma)
        self.adherence_threshold = float(adherence_threshold)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        labels = _normalise_labels(labels, logits)
        foreground = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.foreground_class_ids:
            foreground |= labels == class_id
        foreground = foreground.unsqueeze(1)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            inner, outer = build_boundary_bands(foreground, steps=self.steps)
            log_odds = foreground_log_odds(
                logits,
                self.foreground_class_ids,
                self.complement_class_ids,
            )
            target = foreground[:, 0].float()
            voxel_bce = F.binary_cross_entropy_with_logits(
                log_odds,
                target,
                reduction="none",
            )
            if self.inner_focal_gamma == 0.0 and self.outer_focal_gamma == 0.0:
                inner_voxel_loss = outer_voxel_loss = voxel_bce
            else:
                probability = torch.sigmoid(log_odds)
                truth_probability = torch.where(
                    target.bool(), probability, 1.0 - probability
                )
                focal_base = (1.0 - truth_probability).clamp_min(
                    torch.finfo(truth_probability.dtype).eps
                )
                inner_voxel_loss = (
                    voxel_bce
                    if self.inner_focal_gamma == 0.0
                    else focal_base.pow(self.inner_focal_gamma) * voxel_bce
                )
                outer_voxel_loss = (
                    voxel_bce
                    if self.outer_focal_gamma == 0.0
                    else focal_base.pow(self.outer_focal_gamma) * voxel_bce
                )
            spatial_dimensions = tuple(range(1, voxel_bce.ndim))
            inner_float = inner[:, 0].float()
            outer_float = outer[:, 0].float()
            inner_count = inner_float.sum(spatial_dimensions)
            outer_count = outer_float.sum(spatial_dimensions)
            if self.degree_alpha > 0:
                weights = inner_degree_weights(
                    foreground, inner, alpha=self.degree_alpha,
                    normalization=self.degree_normalization,
                )[:, 0]
                inner_voxel_loss = inner_voxel_loss * weights
            inner_loss = (inner_voxel_loss * inner_float).sum(
                spatial_dimensions
            ) / (inner_count + self.epsilon)
            outer_loss = (outer_voxel_loss * outer_float).sum(
                spatial_dimensions
            ) / (outer_count + self.epsilon)
            valid = (inner_count > 0) & (outer_count > 0)
            case_loss = 0.5 * (inner_loss + outer_loss)
            # Resolve the valid rows once. Repeated CUDA boolean indexing forces
            # repeated dynamic-size synchronization for the same tiny batch.
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            valid_case_loss = case_loss.index_select(0, valid_indices)
            valid_inner_loss = inner_loss.index_select(0, valid_indices)
            valid_outer_loss = outer_loss.index_select(0, valid_indices)
            if valid_indices.numel():
                loss = valid_case_loss.mean()
            else:
                loss = differentiable_zero(logits.float())

        truth = torch.exp(-valid_case_loss).clamp(0.0, 1.0)
        value = torch.stack(
            (torch.exp(-valid_inner_loss), torch.exp(-valid_outer_loss)),
            dim=1,
        )
        edge_touching = _edge_touching(foreground)
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=value,
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid,
                "inner_loss": inner_loss,
                "outer_loss": outer_loss,
                "case_loss": case_loss,
                "inner_voxels": inner_count,
                "outer_voxels": outer_count,
                "edge_touching": edge_touching,
                "metrics": {
                    "raw_loss": valid_case_loss,
                    "inner_loss": valid_inner_loss,
                    "outer_loss": valid_outer_loss,
                    "inner_voxels": inner_count,
                    "outer_voxels": outer_count,
                    "valid_patient": valid.float(),
                    "skipped_patient": (~valid).float(),
                    "edge_touching": edge_touching.float(),
                },
            },
        )
