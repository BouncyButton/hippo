"""Literal MSD hippocampus annotation-plane constraint.

The Task04 protocol defines the last hippocampal-head slice using the uncal
apex.  In the released RAS arrays, stored spatial axis 1 increases toward the
anterior side.  Consequently a valid A/P assignment admits one integer cut
``c`` such that foreground voxels with ``y >= c`` are class 1 (anterior) and
foreground voxels with ``y < c`` are class 2 (posterior).

This module deliberately uses the stored annotation axis.  It does not fit the
hippocampal long axis, PCA direction, centroid order, or an oblique surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraint_result import ConstraintResult, differentiable_zero


@dataclass(frozen=True)
class APPlaneProjection:
    """Hard projected labels and the selected first-anterior slice per case."""

    labels: torch.Tensor
    cuts: torch.Tensor
    violations_before: torch.Tensor
    valid: torch.Tensor


def _normalise_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if tuple(labels.shape) != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("labels must have shape [B,D,H,W] or [B,1,D,H,W].")
    labels = labels.detach().to(device=logits.device)
    if not bool(((labels == 0) | (labels == 1) | (labels == 2)).all()):
        raise ValueError("labels must contain only background=0, anterior=1, posterior=2.")
    return labels


def _validate_logits(logits: torch.Tensor, axis: int) -> None:
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError("logits must have shape [B,3,D,H,W].")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values.")
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("axis must be one of the three spatial tensor axes.")


def _legal_cuts(
    support_by_slice: torch.Tensor,
    *,
    require_both: bool,
) -> torch.Tensor:
    occupied = torch.nonzero(support_by_slice > 0, as_tuple=False).flatten()
    if occupied.numel() == 0:
        return occupied
    low = int(occupied[0])
    high = int(occupied[-1])
    if require_both:
        if high <= low:
            return occupied[:0]
        # c is the first anterior slice.  These candidates leave at least one
        # occupied coordinate on each side of the plane.
        return torch.arange(low + 1, high + 1, device=support_by_slice.device)
    return torch.arange(low, high + 2, device=support_by_slice.device)


def _case_plane_losses(
    ap_margin: torch.Tensor,
    support: torch.Tensor,
    *,
    axis: int,
    anterior_high: bool,
    margin: float,
    require_both: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legal cuts and their mean foreground margin losses for one case."""

    moved_margin = ap_margin.movedim(axis, 0).reshape(ap_margin.shape[axis], -1)
    moved_support = support.movedim(axis, 0).reshape(support.shape[axis], -1).float()
    support_by_slice = moved_support.sum(dim=1)
    cuts = _legal_cuts(support_by_slice, require_both=require_both)
    if cuts.numel() == 0:
        return cuts, moved_margin.reshape(-1)[:0]

    # ap_margin = z_anterior - z_posterior.  Softplus margin violations retain
    # an order-one corrective logit gradient for confidently wrong voxels.
    anterior_cost = F.softplus(margin - moved_margin) * moved_support
    posterior_cost = F.softplus(margin + moved_margin) * moved_support
    anterior_suffix = anterior_cost.flip(0).cumsum(dim=0).flip(0).sum(dim=1)
    posterior_prefix = posterior_cost.cumsum(dim=0).sum(dim=1)
    total_support = moved_support.sum().clamp_min(1.0)

    if anterior_high:
        losses = (posterior_prefix[cuts - 1] + anterior_suffix[cuts]) / total_support
    else:
        losses = (anterior_cost.cumsum(dim=0).sum(dim=1)[cuts - 1]
                  + posterior_cost.flip(0).cumsum(dim=0).flip(0).sum(dim=1)[cuts])
        losses = losses / total_support
    return cuts, losses


class ExistentialAPPlaneLoss(nn.Module):
    """Penalize the best legal axis-aligned A/P assignment plane.

    Ground truth supplies only the foreground support.  It does not supply the
    cut or select a subset of candidates.  The ordinary voxel-supervised loss
    therefore remains responsible for locating the annotated cut, while this
    auxiliary removes assignments that cannot be explained by any single cut.
    """

    def __init__(
        self,
        *,
        axis: int = 1,
        anterior_high: bool = True,
        margin: float = 0.0,
        require_both: bool = True,
        adherence_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("axis must be one of the three spatial tensor axes.")
        if type(anterior_high) is not bool or type(require_both) is not bool:
            raise ValueError("anterior_high and require_both must be booleans.")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("margin must be finite and non-negative.")
        if not 0 <= adherence_threshold <= 1:
            raise ValueError("adherence_threshold must lie in [0,1].")
        self.axis = axis
        self.anterior_high = anterior_high
        self.margin = float(margin)
        self.require_both = require_both
        self.adherence_threshold = float(adherence_threshold)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        _validate_logits(logits, self.axis)
        labels = _normalise_labels(labels, logits)
        ap_margin = (logits[:, 1] - logits[:, 2]).float()
        support = labels != 0
        zero = differentiable_zero(ap_margin)
        losses: list[torch.Tensor] = []
        cuts: list[int] = []
        valid: list[bool] = []
        candidate_counts: list[int] = []
        for case_index in range(logits.shape[0]):
            candidates, candidate_losses = _case_plane_losses(
                ap_margin[case_index],
                support[case_index],
                axis=self.axis,
                anterior_high=self.anterior_high,
                margin=self.margin,
                require_both=self.require_both,
            )
            is_valid = bool(candidates.numel())
            valid.append(is_valid)
            candidate_counts.append(int(candidates.numel()))
            if is_valid:
                best = candidate_losses.argmin()
                losses.append(candidate_losses[best])
                cuts.append(int(candidates[best]))
            else:
                losses.append(zero)
                cuts.append(-1)

        case_loss = torch.stack(losses)
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=logits.device)
        cut_tensor = torch.tensor(cuts, dtype=torch.long, device=logits.device)
        count_tensor = torch.tensor(candidate_counts, dtype=torch.float32, device=logits.device)
        loss = case_loss[valid_tensor].mean() if bool(valid_tensor.any()) else zero
        truth_all = torch.exp(-case_loss).clamp(0.0, 1.0)
        truth = truth_all[valid_tensor]
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=cut_tensor[valid_tensor].float(),
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid_tensor,
                "case_loss": case_loss,
                "selected_cut": cut_tensor,
                "candidate_count": count_tensor,
                "metrics": {
                    "raw_loss": case_loss[valid_tensor],
                    "selected_cut": cut_tensor[valid_tensor].float(),
                    "candidate_count": count_tensor[valid_tensor],
                    "valid_patient": valid_tensor.float(),
                    "skipped_patient": (~valid_tensor).float(),
                },
            },
        )


@torch.no_grad()
def hard_project_ap_plane(
    logits: torch.Tensor,
    *,
    axis: int = 1,
    anterior_high: bool = True,
    require_both: bool = True,
) -> APPlaneProjection:
    """Preserve the predicted foreground union and enforce one A/P plane.

    The selected cut minimizes the number of changed A/P voxels.  This is a
    deterministic inference decoder; it never reads ground-truth labels.
    """

    _validate_logits(logits, axis)
    prediction = logits.argmax(dim=1)
    projected = torch.zeros_like(prediction)
    cuts: list[int] = []
    violations: list[int] = []
    valid: list[bool] = []
    low_class, high_class = ((2, 1) if anterior_high else (1, 2))

    for case_index in range(prediction.shape[0]):
        case = prediction[case_index]
        moved = case.movedim(axis, 0)
        support = moved != 0
        support_by_slice = support.reshape(moved.shape[0], -1).sum(dim=1)
        candidates = _legal_cuts(support_by_slice, require_both=require_both)
        if candidates.numel() == 0:
            projected[case_index] = case
            cuts.append(-1)
            violations.append(0)
            valid.append(False)
            continue

        low_count = (moved == low_class).reshape(moved.shape[0], -1).sum(dim=1)
        high_count = (moved == high_class).reshape(moved.shape[0], -1).sum(dim=1)
        wrong_low_prefix = high_count.cumsum(dim=0)[candidates - 1]
        wrong_high_suffix = low_count.flip(0).cumsum(dim=0).flip(0)[candidates]
        errors = wrong_low_prefix + wrong_high_suffix
        best = errors.argmin()
        cut = int(candidates[best])

        coordinate_shape = [1, 1, 1]
        coordinate_shape[axis] = case.shape[axis]
        coordinate = torch.arange(case.shape[axis], device=case.device).reshape(coordinate_shape)
        case_projected = torch.zeros_like(case)
        case_projected[(case != 0) & (coordinate < cut)] = low_class
        case_projected[(case != 0) & (coordinate >= cut)] = high_class
        projected[case_index] = case_projected
        cuts.append(cut)
        violations.append(int(errors[best]))
        valid.append(True)

    return APPlaneProjection(
        labels=projected,
        cuts=torch.tensor(cuts, dtype=torch.long, device=logits.device),
        violations_before=torch.tensor(violations, dtype=torch.long, device=logits.device),
        valid=torch.tensor(valid, dtype=torch.bool, device=logits.device),
    )
