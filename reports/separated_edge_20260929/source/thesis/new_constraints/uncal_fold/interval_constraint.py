"""Experimental landmark-interval LogLTN constraint with detached grounding.

This module consumes anatomical evidence; it does not discover an uncal apex.
The coordinate field must increase anteriorly, in the SAME millimetre frame as
the interval. Transform the field and foreground support with each image.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..constraint_result import ConstraintResult


class UncalIntervalLogLTNLoss(nn.Module):
    """Constrain A/P only on trusted support outside a landmark interval.

    Clauses: H & Near & AboveInterval -> A; H & Near & BelowInterval -> P.
    Antecedents are detached Boolean groundings. Consequents are conditional
    softmax probabilities for class 1 (A) and 2 (P). Product conjunction,
    Reichenbach implication and geometric-mean universal aggregation reduce
    exactly to mean conditional CE on active voxels, computed in log space.
    External reliability scales each case after aggregation; it is not learned
    through this loss. Active cases get equal weight. Empty cases contribute no
    evidence and have NaN diagnostic truth (rather than vacuous truth=1).
    """

    def __init__(self, *, local_radius_mm: float = 6.0, margin_mm: float = 0.0):
        super().__init__()
        if not math.isfinite(local_radius_mm) or local_radius_mm <= 0:
            raise ValueError("local_radius_mm must be positive and finite")
        if not math.isfinite(margin_mm) or margin_mm < 0:
            raise ValueError("margin_mm must be nonnegative and finite")
        self.local_radius_mm = float(local_radius_mm)
        self.margin_mm = float(margin_mm)

    def forward(self, logits: torch.Tensor, *, support: torch.Tensor,
                anterior_coordinate_mm: torch.Tensor, interval_mm: torch.Tensor,
                reliability: torch.Tensor | None = None) -> ConstraintResult:
        if logits.ndim != 5 or logits.shape[1] != 3 or not logits.is_floating_point():
            raise ValueError("logits must be floating [B,3,D,H,W], classes background/A/P")
        shape = (logits.shape[0], *logits.shape[2:])
        if tuple(support.shape) != shape or support.dtype != torch.bool:
            raise ValueError("support must be Boolean [B,D,H,W]")
        if tuple(anterior_coordinate_mm.shape) != shape:
            raise ValueError("coordinate field must have shape [B,D,H,W]")
        if interval_mm.shape != (logits.shape[0], 2):
            raise ValueError("interval_mm must have shape [B,2]")
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        coords = anterior_coordinate_mm.detach().to(device=logits.device, dtype=dtype)
        interval = interval_mm.detach().to(device=logits.device, dtype=dtype)
        if not torch.isfinite(coords).all() or not torch.isfinite(interval).all():
            raise ValueError("coordinate field and intervals must be finite")
        if (interval[:, 0] > interval[:, 1]).any():
            raise ValueError("interval lower bound must not exceed upper bound")
        if reliability is None:
            confidence = torch.ones(logits.shape[0], device=logits.device, dtype=dtype)
        else:
            confidence = reliability.detach().to(device=logits.device, dtype=dtype)
        if confidence.shape != (logits.shape[0],) or not torch.isfinite(confidence).all():
            raise ValueError("reliability must be a finite [B] tensor")
        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("reliability must lie in [0,1]")
        lower, upper = (interval[:, i, None, None, None] for i in (0, 1))
        centre = (lower + upper) / 2
        local = (coords - centre).abs() <= self.local_radius_mm
        foreground = support.detach().to(logits.device)
        anterior = foreground & local & (coords > upper + self.margin_mm)
        posterior = foreground & local & (coords < lower - self.margin_mm)
        active = anterior | posterior
        counts = active.flatten(1).sum(1)
        valid = (counts > 0) & (confidence > 0)
        # Cast BEFORE subtraction/log-softmax to avoid FP16 overflow.
        log_probs = F.log_softmax(logits[:, 1:3].to(dtype), dim=1)
        nll = torch.where(anterior, -log_probs[:, 0],
                          torch.where(posterior, -log_probs[:, 1], 0.0))
        unweighted = nll.flatten(1).sum(1) / counts.clamp_min(1)
        case_loss = unweighted * confidence * valid
        loss = case_loss.sum() / valid.sum().clamp_min(1)
        truth = torch.where(valid, torch.exp(-unweighted), torch.full_like(unweighted, float("nan")))
        return ConstraintResult(loss=loss, truth=truth, value=unweighted, details={
            "confidence_weighted_agreement": truth * confidence,
            "confidence_adherent": valid & (truth >= .95),
            "valid": valid, "active_voxels": counts,
            "active_foreground_fraction": counts / foreground.flatten(1).sum(1).clamp_min(1),
            "anterior_voxels": anterior.flatten(1).sum(1),
            "posterior_voxels": posterior.flatten(1).sum(1),
            "case_loss": case_loss, "reliability": confidence,
            "interval_width_mm": interval[:, 1] - interval[:, 0],
        })
