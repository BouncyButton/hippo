"""Detached error weighting and local support extension for union bands."""
import torch
import torch.nn.functional as F
from thesis.new_constraints.bands.outer_boundary import (
    build_boundary_bands, foreground_log_odds, _dilate_6,
)


@torch.no_grad()
def masks(logits, labels, variant):
    if variant not in ('weighted', 'adaptive'):
        raise ValueError(variant)
    if logits.ndim != 5 or logits.shape[1] != 4 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected aligned four-class MNI logits and labels')
    foreground = labels > 0
    errors = foreground ^ (logits.detach().argmax(1, keepdim=True) > 0)
    inner, outer = build_boundary_bands(foreground, steps=2)
    support = inner | outer
    if variant == 'adaptive':
        support = support | _dilate_6(errors, 1)
    return support & foreground, support & ~foreground, 1. + 2. * errors.float(), errors


def loss(logits, labels, variant):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        inner, outer, weights, _ = masks(logits.float(), labels, variant)
        odds = foreground_log_odds(logits, (1, 2, 3), (0,))
        bce = F.binary_cross_entropy_with_logits(odds, (labels[:, 0] > 0).float(), reduction='none')
        iw, ow = (inner * weights)[:, 0], (outer * weights)[:, 0]
        axes = (1, 2, 3)
        ni, no = iw.sum(axes), ow.sum(axes)
        value = .5 * ((bce * iw).sum(axes) / (ni + 1e-6) + (bce * ow).sum(axes) / (no + 1e-6))
        indices = ((ni > 0) & (no > 0)).nonzero(as_tuple=False).flatten()
        return value.index_select(0, indices).mean() if indices.numel() else logits.float().sum() * 0.


def install(variant):
    from scripts.mni_constraints import objectives
    original = getattr(objectives.term, '_original_union_term', objectives.term)
    def term(name, logits, labels, target_sdf=None, depth_weights=None):
        if name == 'bands':
            return loss(logits, labels, variant)
        return original(name, logits, labels, target_sdf, depth_weights)
    term._original_union_term = original
    objectives.term = term
