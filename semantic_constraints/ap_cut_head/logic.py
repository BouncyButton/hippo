"""Differentiable cut-conditioned A/P ordering; not an independent cut locator."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def ap_cut_consistency_loss(
    segmentation_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    foreground_support: torch.Tensor,
    *,
    detach_cut: bool = True,
) -> torch.Tensor:
    """Penalize foreground A/P probabilities inconsistent with a soft cut.

    A cut index c separates Y=c and Y=c+1. For a foreground voxel at Y=y,
    P(anterior | foreground, cut) = P(c < y). The cut distribution should be
    anchored by its own supervised loss; detaching it here prevents the logic
    term from moving the cut to rationalize existing segmentation errors.
    """
    if segmentation_logits.ndim != 5 or segmentation_logits.shape[1] != 3:
        raise ValueError("expected [B,3,X,Y,Z] segmentation logits")
    if cut_logits.shape != (segmentation_logits.shape[0], segmentation_logits.shape[3] - 1):
        raise ValueError("expected [B,Y-1] cut logits")
    if foreground_support.shape != segmentation_logits.shape[:1] + segmentation_logits.shape[2:]:
        raise ValueError("expected [B,X,Y,Z] foreground support")
    cut_probability = cut_logits.softmax(dim=1)
    if detach_cut:
        cut_probability = cut_probability.detach()
    # At Y=0 no cut is below the voxel; at Y=y>0 sum cuts 0,...,y-1.
    anterior_given_foreground = F.pad(cut_probability.cumsum(dim=1), (1, 0))[:, None, None, :, None]
    ap_log_probability = F.log_softmax(segmentation_logits[:, 1:3], dim=1)
    log_conditioned = ap_log_probability - torch.logsumexp(ap_log_probability, dim=1, keepdim=True)
    per_voxel = -(
        anterior_given_foreground * log_conditioned[:, 0:1]
        + (1 - anterior_given_foreground) * log_conditioned[:, 1:2]
    ).squeeze(1)
    support = foreground_support.float().clamp(0, 1)
    return (per_voxel * support).sum() / support.sum().clamp_min(1)
