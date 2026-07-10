"""Differentiable geometric, topological, and statistical mask primitives.

The primitives operate on soft class masks, usually probabilities selected from
segmentation logits. Expected shapes are ``(B, *spatial)`` or
``(B, 1, *spatial)`` for 2D and 3D spatial domains. Functions return one value
per sample unless otherwise noted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _as_mask(mask: Tensor) -> Tensor:
    if mask.ndim < 3:
        raise ValueError(
            "Mask primitives expect batched masks shaped (B, *spatial) or "
            "(B, 1, *spatial). For an unbatched mask, pass mask.unsqueeze(0)."
        )
    if not torch.is_floating_point(mask):
        mask = mask.float()
    if mask.ndim >= 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return mask


def _pair(mask1: Tensor, mask2: Tensor) -> tuple[Tensor, Tensor]:
    mask1 = _as_mask(mask1)
    mask2 = _as_mask(mask2)
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes must match, got {tuple(mask1.shape)} and {tuple(mask2.shape)}.")
    return mask1, mask2


def _spatial_ndim(mask: Tensor) -> int:
    return mask.ndim - 1


def _spatial_dims(mask: Tensor) -> tuple[int, ...]:
    return tuple(range(1, mask.ndim))


def _spacing(mask: Tensor, spacing: float | Sequence[float] | Tensor | None) -> Tensor:
    spatial_ndim = _spatial_ndim(mask)
    if spacing is None:
        values = torch.ones(spatial_ndim, device=mask.device, dtype=mask.dtype)
    elif isinstance(spacing, Tensor):
        values = spacing.to(device=mask.device, dtype=mask.dtype)
    elif isinstance(spacing, Sequence):
        values = torch.tensor(list(spacing), device=mask.device, dtype=mask.dtype)
    else:
        values = torch.full((spatial_ndim,), float(spacing), device=mask.device, dtype=mask.dtype)
    if values.numel() != spatial_ndim:
        raise ValueError(f"Expected {spatial_ndim} spacing values, got {values.numel()}.")
    return values


def _voxel_volume(mask: Tensor, spacing: float | Sequence[float] | Tensor | None) -> Tensor:
    return _spacing(mask, spacing).prod()


def _coordinate_grid(mask: Tensor, spacing: float | Sequence[float] | Tensor | None) -> Tensor:
    spacings = _spacing(mask, spacing)
    axes = [
        torch.arange(size, device=mask.device, dtype=mask.dtype) * spacings[idx]
        for idx, size in enumerate(mask.shape[1:])
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def _smooth_abs(x: Tensor, eps: float) -> Tensor:
    return torch.sqrt(x.square() + eps) - math.sqrt(eps)


def _forward_diff(mask: Tensor, axis: int) -> Tensor:
    diff = torch.zeros_like(mask)
    current = [slice(None)] * mask.ndim
    previous = [slice(None)] * mask.ndim
    current[axis] = slice(1, None)
    previous[axis] = slice(None, -1)
    diff[tuple(current)] = mask[tuple(current)] - mask[tuple(previous)]
    return diff


def _soft_boundary(mask: Tensor, eps: float = 1e-6) -> Tensor:
    grad_sq = torch.zeros_like(mask)
    for axis in _spatial_dims(mask):
        grad_sq = grad_sq + _forward_diff(mask, axis).square()
    return torch.sqrt(grad_sq + eps) - math.sqrt(eps)


def _pool(mask: Tensor, radius: int, mode: str) -> Tensor:
    spatial_ndim = _spatial_ndim(mask)
    if spatial_ndim not in (2, 3):
        raise ValueError("Neighborhood primitives currently support 2D and 3D masks.")
    if radius < 1:
        return mask

    kernel_size = 2 * radius + 1
    x = mask.unsqueeze(1)
    pool_fn = F.max_pool2d if spatial_ndim == 2 else F.max_pool3d
    avg_fn = F.avg_pool2d if spatial_ndim == 2 else F.avg_pool3d
    fn = pool_fn if mode == "max" else avg_fn
    return fn(x, kernel_size=kernel_size, stride=1, padding=radius).squeeze(1)


def volume(mask: Tensor, spacing: float | Sequence[float] | Tensor | None = None) -> Tensor:
    """Soft physical volume of each mask, returned as ``(B,)``."""
    mask = _as_mask(mask)
    return mask.sum(dim=_spatial_dims(mask)) * _voxel_volume(mask, spacing)


def centroid(
    mask: Tensor,
    spacing: float | Sequence[float] | Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Soft center of mass in physical coordinates, returned as ``(B, D)``."""
    mask = _as_mask(mask)
    grid = _coordinate_grid(mask, spacing)
    weighted = mask.unsqueeze(1) * grid.unsqueeze(0)
    reduce_dims = tuple(range(2, weighted.ndim))
    denom = mask.sum(dim=_spatial_dims(mask)).clamp_min(eps).unsqueeze(1)
    return weighted.sum(dim=reduce_dims) / denom


def distance(
    mask1: Tensor,
    mask2: Tensor,
    spacing: float | Sequence[float] | Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Euclidean distance between soft centroids, returned as ``(B,)``."""
    mask1, mask2 = _pair(mask1, mask2)
    delta = centroid(mask1, spacing=spacing, eps=eps) - centroid(mask2, spacing=spacing, eps=eps)
    return torch.sqrt(delta.square().sum(dim=1) + eps)


def overlap(mask1: Tensor, mask2: Tensor, mode: str = "dice", eps: float = 1e-8) -> Tensor:
    """Soft overlap score between two masks.

    Modes:
        ``"intersection"`` returns the unnormalized soft intersection.
        ``"dice"`` returns soft Dice overlap.
        ``"iou"`` returns soft intersection over union.
    """
    mask1, mask2 = _pair(mask1, mask2)
    dims = _spatial_dims(mask1)
    intersection = (mask1 * mask2).sum(dim=dims)
    if mode == "intersection":
        return intersection
    if mode == "dice":
        return 2.0 * intersection / (mask1.sum(dim=dims) + mask2.sum(dim=dims) + eps)
    if mode == "iou":
        union = mask1.sum(dim=dims) + mask2.sum(dim=dims) - intersection
        return intersection / (union + eps)
    raise ValueError(f"Unknown overlap mode: {mode}.")


def contains(container: Tensor, containee: Tensor, eps: float = 1e-8) -> Tensor:
    """Fraction of ``containee`` mass lying inside ``container``, returned as ``(B,)``."""
    container, containee = _pair(container, containee)
    dims = _spatial_dims(container)
    inside = (container * containee).sum(dim=dims)
    return inside / (containee.sum(dim=dims) + eps)


def adjacent(mask1: Tensor, mask2: Tensor, radius: int = 1, eps: float = 1e-8) -> Tensor:
    """Soft boundary contact score between masks, returned as ``(B,)``.

    A value near 1 means high-probability boundary regions of each mask lie
    within ``radius`` voxels of the other mask.
    """
    mask1, mask2 = _pair(mask1, mask2)
    b1 = _soft_boundary(mask1)
    b2 = _soft_boundary(mask2)
    near_1 = _pool(mask1, radius=radius, mode="max")
    near_2 = _pool(mask2, radius=radius, mode="max")
    contact = (b1 * near_2 + b2 * near_1).sum(dim=_spatial_dims(mask1))
    boundary_mass = (b1 + b2).sum(dim=_spatial_dims(mask1))
    return contact / (boundary_mass + eps)


def boundary_length(
    mask: Tensor,
    spacing: float | Sequence[float] | Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Soft boundary measure via total variation, returned as ``(B,)``.

    For 2D masks this approximates perimeter. For 3D masks this approximates
    surface area. The name follows the DSL primitive list.
    """
    mask = _as_mask(mask)
    spacings = _spacing(mask, spacing)
    total = torch.zeros(mask.shape[0], device=mask.device, dtype=mask.dtype)
    voxel_volume = spacings.prod()
    for idx, axis in enumerate(_spatial_dims(mask)):
        face_measure = voxel_volume / spacings[idx]
        total = total + _smooth_abs(_forward_diff(mask, axis), eps).sum(dim=_spatial_dims(mask)) * face_measure
    return total


def compactness(
    mask: Tensor,
    spacing: float | Sequence[float] | Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Scale-normalized compactness penalty, returned as ``(B,)``.

    For 2D this is ``perimeter^2 / (4 pi area)``. For 3D this is
    ``surface^3 / (36 pi volume^2)``. Lower is more compact, with idealized
    circles or spheres near 1.
    """
    mask = _as_mask(mask)
    dim = _spatial_ndim(mask)
    size = volume(mask, spacing=spacing)
    boundary = boundary_length(mask, spacing=spacing)
    if dim == 2:
        return boundary.square() / (4.0 * math.pi * size + eps)
    if dim == 3:
        return boundary.pow(3) / (36.0 * math.pi * size.square() + eps)
    return boundary.pow(dim) / (size.pow(dim - 1) + eps)


def entropy(mask: Tensor, eps: float = 1e-8, reduction: str = "mean") -> Tensor:
    """Binary entropy of a soft mask, returned as ``(B,)``."""
    mask = _as_mask(mask).clamp(min=eps, max=1.0 - eps)
    entropy_map = -(mask * mask.log() + (1.0 - mask) * (1.0 - mask).log())
    if reduction == "mean":
        return entropy_map.mean(dim=_spatial_dims(mask))
    if reduction == "sum":
        return entropy_map.sum(dim=_spatial_dims(mask))
    raise ValueError(f"Unknown entropy reduction: {reduction}.")


def connectedness(mask: Tensor, steps: int = 32, temperature: float = 20.0, eps: float = 1e-8) -> Tensor:
    """Soft single-component score, returned as ``(B,)``.

    This is a differentiable surrogate, not an exact connected-component count.
    It seeds a component at the highest-probability region using softmax, then
    repeatedly performs masked geodesic dilation. The score is the fraction of
    total mask mass reachable from that seed.
    """
    mask = _as_mask(mask)
    if _spatial_ndim(mask) not in (2, 3):
        raise ValueError("connectedness currently supports 2D and 3D masks.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    flat = mask.flatten(start_dim=1)
    seed = torch.softmax(flat * temperature, dim=1).reshape_as(mask)
    reachable = seed * mask
    for _ in range(steps):
        reachable = _pool(reachable, radius=1, mode="max") * mask

    dims = _spatial_dims(mask)
    return reachable.sum(dim=dims) / (mask.sum(dim=dims) + eps)
