"""Full-volume classwise boundary supervision with clipped target distances."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch


def signed_distance_maps(labels: torch.Tensor, clip: float = 8.0) -> torch.Tensor:
    """Return normalized anterior/posterior SDFs on the transformed label grid.

    Uses the existing spline refiner's half-voxel sign convention. Distances
    are preprocessing, never differentiable predictions. Missing/full classes
    have constant positive/negative clipped distances respectively.
    """
    if labels.ndim != 5 or labels.shape[1] != 1 or clip <= 0:
        raise ValueError("Expected [N,1,D,H,W] labels and a positive distance clip")
    values = labels.detach().cpu().numpy()[:, 0]
    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Signed-distance supervision requires labels 0, 1, 2")
    maps = np.empty((len(values), 2, *values.shape[1:]), dtype=np.float32)
    for batch, volume in enumerate(values):
        for channel, class_id in enumerate((1, 2)):
            mask = volume == class_id
            if not mask.any():
                field = np.full(mask.shape, clip, dtype=np.float32)
            elif mask.all():
                field = np.full(mask.shape, -clip, dtype=np.float32)
            else:
                field = np.where(mask, -(ndimage.distance_transform_edt(mask) - .5),
                                 ndimage.distance_transform_edt(~mask) - .5)
            maps[batch, channel] = np.clip(field, -clip, clip) / clip
    return torch.from_numpy(maps).to(labels.device)


def signed_distance_loss(logits, labels, target_sdf):
    """Mean (p-y)*phi over both foreground classes and every voxel.

    Subtracting the fixed y*phi sets perfect probabilities to zero without
    changing the probability-based boundary loss's parameter gradients.
    """
    if logits.ndim != 5 or logits.shape[1] != 3 or labels.shape != logits[:, :1].shape:
        raise ValueError("Expected three-class logits and matching single-channel labels")
    if target_sdf is None or target_sdf.shape != logits[:, 1:].shape:
        raise ValueError("Precomputed classwise target SDF is required")
    truth = torch.cat((labels == 1, labels == 2), dim=1).float()
    probability = logits.float().softmax(dim=1)[:, 1:]
    return ((probability - truth) * target_sdf.detach().float()).mean()
