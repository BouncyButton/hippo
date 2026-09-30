"""Matched non-structural control for patient-specific plane supervision."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraint_result import ConstraintResult
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
        support = labels != 0
        counts = support.flatten(1).sum(1)
        valid_tensor = counts > 0
        if self.require_both:
            valid_tensor = valid_tensor & (labels == 1).flatten(1).any(1) & (labels == 2).flatten(1).any(1)
        # Fixed-shape reductions keep validity decisions on the GPU. Python bool
        # tests and foreground boolean indexing previously synchronized every case.
        voxel_losses = torch.where(
            labels == 1,
            F.softplus(self.margin - ap_margin),
            F.softplus(self.margin + ap_margin),
        )
        case_loss = torch.where(support, voxel_losses, 0.0).flatten(1).sum(1) / counts.clamp_min(1)
        case_loss = case_loss * valid_tensor
        loss = case_loss.sum() / valid_tensor.sum().clamp_min(1)
        truth_all = torch.exp(-case_loss).clamp(0.0, 1.0)
        truth = truth_all[valid_tensor]
        valid_losses = case_loss[valid_tensor]
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=valid_losses,
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid_tensor,
                "case_loss": case_loss,
                "metrics": {
                    "raw_loss": valid_losses,
                    "valid_patient": valid_tensor.float(),
                    "skipped_patient": (~valid_tensor).float(),
                },
            },
        )
