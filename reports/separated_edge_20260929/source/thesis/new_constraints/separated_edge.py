"""Independently normalized signed edge rules; no change to band geometry."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .bands.outer_boundary import build_boundary_bands
from .edge_consistency import BoundaryEdgeConsistencyLoss

RULES = ('inner', 'outer', 'cross')


def rule_statistics(logits: torch.Tensor, labels: torch.Tensor):
    """Return differentiable per-case sums and counts for disjoint face sets."""
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError('Expected [B,3,D,H,W] logits.')
    if labels.ndim == 4:
        labels = labels[:, None]
    if labels.shape != logits[:, :1].shape:
        raise ValueError('Labels must match spatial shape and batch size.')
    foreground = labels > 0
    with torch.no_grad():
        inner, outer = build_boundary_bands(foreground, steps=2)
    probability = logits.float().softmax(1)[:, 1:].sum(1, keepdim=True)
    residual = probability - foreground.float()
    sums = {k: residual.flatten(1).sum(1) * 0 for k in RULES}
    counts = {k: torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device) for k in RULES}
    support = inner | outer
    for axis in (2, 3, 4):
        a, b = [slice(None)] * 5, [slice(None)] * 5
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        error = (residual[a] - residual[b]).square()
        masks = {'inner': inner[a] & inner[b], 'outer': outer[a] & outer[b],
                 'cross': support[a] & support[b] & (foreground[a] != foreground[b])}
        for k, mask in masks.items():
            sums[k] = sums[k] + (error * mask).flatten(1).sum(1)
            counts[k] = counts[k] + mask.flatten(1).sum(1)
    return sums, counts


def rule_losses(logits: torch.Tensor, labels: torch.Tensor):
    sums, counts = rule_statistics(logits, labels)
    losses = {}
    for k in RULES:
        valid = counts[k] > 0
        losses[k] = ((sums[k] / counts[k].clamp_min(1)) * valid).sum() / valid.sum().clamp_min(1)
    return losses, counts


class SeparatedEdgeLoss(nn.Module):
    """Equal rule means, each averaging valid patients; empty rules contribute zero.

    The experiment requires all three rules in every training/calibration case.
    Empty-rule handling keeps the general loss finite without inventing edges.
    """
    def forward(self, logits, labels):
        terms, _ = rule_losses(logits, labels)
        return sum(terms.values()) / 3


def edge_loss(arm):
    if arm == 'pooled':
        return BoundaryEdgeConsistencyLoss()
    if arm == 'separated':
        return SeparatedEdgeLoss()
    raise ValueError(f'Unknown arm: {arm}')


def matched_budget(ratios, target=.1, cap=.5, reference_weight=None):
    """Match median budgets; a supplied historical coefficient is immutable.

    Fail if matching that coefficient would exceed the candidate's p95 cap.
    Never silently weaken the original control to accommodate a new rule.
    """
    if set(ratios) != {'pooled', 'separated'}:
        raise ValueError('Both experimental arms are required.')
    stats = {}
    for arm, values in ratios.items():
        a = np.asarray(values, dtype=float)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a <= 0).any():
            raise ValueError('Calibration requires finite positive gradient ratios.')
        stats[arm] = {'median': float(np.median(a)), 'p95': float(np.quantile(a, .95))}
    if reference_weight is None:
        budget = min([target] + [cap * s['median'] / s['p95'] for s in stats.values()])
    else:
        if not np.isfinite(reference_weight) or reference_weight <= 0:
            raise ValueError('Historical coefficient must be finite and positive.')
        budget = reference_weight * stats['pooled']['median']
        if any(budget * s['p95'] / s['median'] > cap + 1e-6 for s in stats.values()):
            raise ValueError('Cannot match historical median budget within the p95 cap; stop for review.')
    return {'target': target, 'cap': cap, 'matched_median_budget': budget,
            'weights': {k: budget / s['median'] for k, s in stats.items()}, 'ratios': stats}
