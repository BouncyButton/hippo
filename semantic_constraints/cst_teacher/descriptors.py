"""Differentiable case and slice descriptors for the CST teacher.

The spatial convention in this module is ``(X, Y, Z)``.  ``Y`` is the
anterior/posterior sampling axis used by the existing MSD experiments.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


DESCRIPTOR_NAMES = (
    "anterior_volume_fraction",
    "union_volume_fraction",
    "signed_centroid_y_gap",
    "union_elongation",
)


def labels_to_probabilities(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Convert ``[B, 1, X, Y, Z]`` or ``[B, X, Y, Z]`` labels to one-hot."""

    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4:
        raise ValueError("labels must have shape [B, 1, X, Y, Z] or [B, X, Y, Z]")
    if torch.is_floating_point(labels):
        labels = labels.round()
    labels = labels.long()
    if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"labels must lie in [0, {num_classes - 1}]")
    return F.one_hot(labels, num_classes=num_classes).movedim(-1, 1).float()


def _validate_probabilities(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 5 or probabilities.shape[1] < 3:
        raise ValueError("probabilities must have shape [B, >=3, X, Y, Z]")
    if not torch.is_floating_point(probabilities):
        raise ValueError("probabilities must be floating point")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")


def _normalized_grid(probabilities: torch.Tensor) -> torch.Tensor:
    axes = [
        torch.linspace(-0.5, 0.5, size, device=probabilities.device, dtype=probabilities.dtype)
        for size in probabilities.shape[2:]
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def soft_descriptors(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return four case descriptors from soft anterior/posterior masks.

    All geometric coordinates are normalized by the volume dimensions.  The
    returned tensor has shape ``[B, len(DESCRIPTOR_NAMES)]`` and remains
    differentiable with respect to ``probabilities``.
    """

    _validate_probabilities(probabilities)
    # ``eigvalsh`` is not reliably implemented for half precision.  Keeping
    # descriptor algebra in float32 also prevents small soft masks from losing
    # their covariance under AMP while preserving the gradient to the logits.
    work = probabilities.float() if probabilities.dtype in (torch.float16, torch.bfloat16) else probabilities
    anterior = work[:, 1]
    posterior = work[:, 2]
    union = anterior + posterior
    spatial_dims = (1, 2, 3)

    anterior_mass = anterior.sum(dim=spatial_dims)
    posterior_mass = posterior.sum(dim=spatial_dims)
    union_mass = union.sum(dim=spatial_dims).clamp_min(eps)
    voxel_count = float(union[0].numel())

    anterior_fraction = anterior_mass / union_mass
    union_fraction = union_mass / voxel_count

    y = torch.linspace(
        -0.5,
        0.5,
        probabilities.shape[3],
        device=probabilities.device,
        dtype=probabilities.dtype,
    ).view(1, 1, -1, 1)
    anterior_y = (anterior * y).sum(dim=spatial_dims) / anterior_mass.clamp_min(eps)
    posterior_y = (posterior * y).sum(dim=spatial_dims) / posterior_mass.clamp_min(eps)
    centroid_gap = anterior_y - posterior_y

    grid = _normalized_grid(work)
    mean = (union[:, None] * grid[None]).sum(dim=(2, 3, 4)) / union_mass[:, None]
    centered = grid[None] - mean[:, :, None, None, None]
    weighted = centered * union[:, None].sqrt()
    flat = weighted.flatten(start_dim=2)
    covariance = flat @ flat.transpose(1, 2) / union_mass[:, None, None]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(eps)
    elongation = torch.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0])

    return torch.stack(
        (anterior_fraction, union_fraction, centroid_gap, elongation),
        dim=1,
    )


def sampled_slice_profiles(
    probabilities: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return normalized A/P cross-sectional areas at common Y positions.

    The output is ``[B, N, 2]``.  Areas are divided by the largest union area
    in the sampled set for each case, retaining the shape of the longitudinal
    profile while avoiding a cohort-specific voxel scale.
    """

    _validate_probabilities(probabilities)
    positions = torch.as_tensor(positions, device=probabilities.device, dtype=torch.long)
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError("positions must be a non-empty one-dimensional sequence")
    if positions.min() < 0 or positions.max() >= probabilities.shape[3]:
        raise ValueError("slice position is outside the Y axis")

    foreground = probabilities[:, 1:3].index_select(3, positions)
    areas = foreground.sum(dim=(2, 4)).movedim(1, 2)
    scale = areas.sum(dim=2).amax(dim=1, keepdim=True).clamp_min(eps)
    return areas / scale[:, :, None]
