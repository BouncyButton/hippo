"""Matched non-structural control for patient-specific plane supervision."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraint_result import ConstraintResult, differentiable_zero
from .existential import _normalise_labels, _validate_logits


AP_CONDITIONAL_CE_METRICS = (
    "raw_loss",
    "valid_patient",
    "skipped_patient",
)


class OriginalLabelAPConditionalCELoss(nn.Module):
    """Foreground-only A/P logistic loss using the original dense labels.

    At zero margin this is ordinary binary cross-entropy after conditioning on
    the voxel being foreground: background logit 0 receives no direct gradient.
    It is the mechanism-matched control for ``BestFitAPPlaneLocationLoss``.
    The two losses are identical on exactly planar labels; on mixed slices this
    control retains the released voxel labels while the location loss uses the
    best plane's projected side assignments.
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
        # Axis and direction are retained in the interface and provenance so
        # calibration is exactly matched to the plane experiment.  Original
        # voxel labels themselves determine the class target in this control.
        self.axis = axis
        self.anterior_high = anterior_high
        self.margin = float(margin)
        self.require_both = require_both
        self.adherence_threshold = float(adherence_threshold)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        _validate_logits(logits, self.axis)
        labels = _normalise_labels(labels, logits)
        ap_margin = (logits[:, 1] - logits[:, 2]).float()
        zero = differentiable_zero(ap_margin)
        losses: list[torch.Tensor] = []
        valid: list[bool] = []
        for case_index in range(logits.shape[0]):
            case_labels = labels[case_index]
            support = case_labels != 0
            has_both = bool((case_labels == 1).any()) and bool((case_labels == 2).any())
            is_valid = bool(support.any()) and (has_both or not self.require_both)
            valid.append(is_valid)
            if not is_valid:
                losses.append(zero)
                continue
            margin = ap_margin[case_index]
            voxel_losses = torch.where(
                case_labels == 1,
                F.softplus(self.margin - margin),
                F.softplus(self.margin + margin),
            )
            losses.append(voxel_losses[support].mean())

        case_loss = torch.stack(losses)
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=logits.device)
        loss = case_loss[valid_tensor].mean() if bool(valid_tensor.any()) else zero
        truth_all = torch.exp(-case_loss).clamp(0.0, 1.0)
        truth = truth_all[valid_tensor]
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=case_loss[valid_tensor],
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid_tensor,
                "case_loss": case_loss,
                "metrics": {
                    "raw_loss": case_loss[valid_tensor],
                    "valid_patient": valid_tensor.float(),
                    "skipped_patient": (~valid_tensor).float(),
                },
            },
        )
