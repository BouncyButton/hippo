"""Categorical supervision on the original outer-contour band support.

Only the voxel target changes: the correct subfield replaces the foreground
union on the inner band. Outer/background targets and side/case averaging stay
unchanged. No internal-interface support or extra class weighting is added.
"""
import torch
import torch.nn.functional as F

from thesis.new_constraints.bands.outer_boundary import build_boundary_bands


def loss(logits, labels):
    if logits.ndim != 5 or logits.shape[1] != 4 or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected aligned four-class MNI logits and labels')
    with torch.no_grad():
        inner, outer = build_boundary_bands(labels > 0, steps=2)
    voxel_loss = F.cross_entropy(logits.float(), labels[:, 0].long(), reduction='none')
    axes = (1, 2, 3)
    inner_float, outer_float = inner[:, 0].float(), outer[:, 0].float()
    inner_count, outer_count = inner_float.sum(axes), outer_float.sum(axes)
    inner_loss = (voxel_loss * inner_float).sum(axes) / (inner_count + 1e-6)
    outer_loss = (voxel_loss * outer_float).sum(axes) / (outer_count + 1e-6)
    valid = (inner_count > 0) & (outer_count > 0)
    indices = valid.nonzero(as_tuple=False).flatten()
    case_loss = .5 * (inner_loss + outer_loss)
    if indices.numel():
        return case_loss.index_select(0, indices).mean()
    return logits.float().sum() * 0.


def install():
    """Bind this term into the existing actual-base training/calibration paths."""
    from scripts.mni_constraints import objectives
    if getattr(objectives.term, '_class_aware_bands', False):
        return
    original = objectives.term

    def term(name, logits, labels, target_sdf=None, depth_weights=None):
        if name == 'bands':
            return loss(logits, labels)
        return original(name, logits, labels, target_sdf, depth_weights)

    term._class_aware_bands = True
    objectives.term = term
