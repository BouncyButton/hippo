"""Binary salt adaptations of the repository's SDF and signed-face losses."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch

METHODS = ("baseline", "bands", "sdf", "bands_sdf", "bands_edge", "bands_sdf_edge")


def signed_distance_map(mask: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """Negative inside, positive outside; half-voxel distances in native voxels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not np.isfinite(clip) or clip <= 0:
        raise ValueError("Expected a 3D mask and finite positive distance clip")
    if not mask.any():
        return np.ones(mask.shape, dtype=np.float32)
    if mask.all():
        return -np.ones(mask.shape, dtype=np.float32)
    field = np.where(mask, -(ndimage.distance_transform_edt(mask) - .5),
                     ndimage.distance_transform_edt(~mask) - .5)
    return (np.clip(field, -clip, clip) / clip).astype(np.float32)


def signed_distance_loss(logits: torch.Tensor, labels: torch.Tensor,
                         target_sdf: torch.Tensor) -> torch.Tensor:
    """Mean (p-y)*phi; perfect probabilities have zero loss, unchanged gradients."""
    if logits.ndim != 5 or logits.shape[1] != 2 or labels.shape != logits[:, :1].shape:
        raise ValueError("Expected binary logits and matching single-channel labels")
    if target_sdf is None or target_sdf.shape != labels.shape:
        raise ValueError("A matching precomputed salt SDF is required")
    probability = logits.float().softmax(1)[:, 1:2]
    return ((probability - labels.float()) * target_sdf.detach().float()).mean()


def signed_face_edge_loss(logits: torch.Tensor, labels: torch.Tensor,
                          support: torch.Tensor) -> torch.Tensor:
    """Mean squared signed face-contrast error; both endpoints must be supported."""
    if logits.ndim != 5 or logits.shape[1] != 2 or labels.shape != logits[:, :1].shape:
        raise ValueError("Expected binary logits and matching single-channel labels")
    if support.shape != labels.shape:
        raise ValueError("Edge support must match the labels")
    probability = logits.float().softmax(1)[:, 1:2]
    residual = probability - labels.float()
    total = residual.flatten(1).sum(1) * 0
    count = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    for axis in (2, 3, 4):
        left, right = [slice(None)] * 5, [slice(None)] * 5
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = support[left].bool() & support[right].bool()
        total = total + ((residual[left] - residual[right]).square() * valid).flatten(1).sum(1)
        count = count + valid.flatten(1).sum(1)
    valid_cases = count > 0
    return ((total / count.clamp_min(1)) * valid_cases).sum() / valid_cases.sum().clamp_min(1)
