"""Fixed-coordinate ray integrals; cumulative profiles extend legacy moments.

These are supervision operators on masks, not a physical MRI/optical forward
model. No centering, alignment, uncertainty gating, or learned target is used.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def grouped_masks(probabilities: torch.Tensor, labels: torch.Tensor):
    """Equal-weight groups: foreground union, anterior, posterior."""
    if probabilities.ndim != 5 or probabilities.shape[1] != 3:
        raise ValueError("Expected three-class probabilities [B,3,D,H,W]")
    if labels.shape != probabilities.shape[:1] + probabilities.shape[2:]:
        raise ValueError("Labels must have shape [B,D,H,W]")
    target = F.one_hot(labels.long(), 3).movedim(-1, 1).to(probabilities)
    return tuple(torch.cat((v[:, 1:].sum(1, keepdim=True), v[:, 1:]), 1)
                 for v in (probabilities, target))


def profile_discrepancy(residual: torch.Tensor, ds: float, length: float,
                        mode: str) -> torch.Tensor:
    """L1 on ray mass or symmetric prefix/suffix mass; last dimension is depth.

Both terms divide by one fixed physical crop length. Averaging prefix and
suffix terms makes the depth loss invariant to which end is called entry.
"""
    if mode not in {"thickness", "depth"}:
        raise ValueError(mode)
    mass = residual.sum(-1) * (ds / length)
    if mode == "thickness":
        return mass.abs().mean()
    prefix = residual.cumsum(-1) * (ds / length)
    suffix = residual.flip(-1).cumsum(-1) * (ds / length)
    return (prefix.abs().mean() + suffix.abs().mean()) / 2


class RayGeometry:
    """Three exact axis views plus optional four diagonal parallel-beam views.

Coordinates are physical (D,H,W). Diagonal views cover a sphere containing
the complete crop including interpolation support, with zero padding outside.
Trilinear interpolation is applied identically to probabilities and one-hot GT.
"""

    def __init__(self, shape: Sequence[int], spacing=(1., 1., 1.), *,
                 oblique: bool = True, detector_step: float = 2.,
                 depth_step: float = 1., device="cpu", dtype=torch.float32):
        self.shape = tuple(int(x) for x in shape)
        self.spacing = tuple(float(x) for x in spacing)
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError("Expected three spatial dimensions >= 2")
        if len(self.spacing) != 3 or min(self.spacing) <= 0:
            raise ValueError("Expected three positive spacings")
        if not all(math.isfinite(x) and x > 0 for x in (*self.spacing, detector_step, depth_step)):
            raise ValueError("Sampling distances must be positive and finite")
        self.length = max(n * s for n, s in zip(self.shape, self.spacing))
        self.grids = []
        self.depth_step = depth_step
        if not oblique:
            return
        extent = torch.tensor([(n + 1) * s / 2 for n, s in zip(self.shape, self.spacing)],
                              dtype=dtype, device=device)
        radius = float(extent.norm())
        detector_n = math.ceil(radius / detector_step)
        depth_n = math.ceil(radius / depth_step)
        detectors = torch.arange(-detector_n, detector_n + 1, device=device, dtype=dtype) * detector_step
        depths = torch.arange(-depth_n, depth_n + 1, device=device, dtype=dtype) * depth_step
        u, v, t = torch.meshgrid(detectors, detectors, depths, indexing="ij")
        denom = torch.tensor([(n - 1) * s / 2 for n, s in zip(self.shape, self.spacing)],
                             dtype=dtype, device=device)
        for direction in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)):
            d = torch.tensor(direction, dtype=dtype, device=device)
            d = d / d.norm()
            anchor = torch.tensor((1., 0., 0.), dtype=dtype, device=device)
            a = torch.linalg.cross(d, anchor)
            a = a / a.norm()
            b = torch.linalg.cross(d, a)
            physical = u[..., None] * a + v[..., None] * b + t[..., None] * d
            # grid_sample coordinates use W,H,D despite tensor indexing D,H,W.
            self.grids.append((physical / denom).flip(-1).unsqueeze(0))

    def loss(self, prediction: torch.Tensor, target: torch.Tensor, mode: str,
             *, include_oblique=True) -> torch.Tensor:
        if prediction.shape != target.shape or tuple(prediction.shape[2:]) != self.shape:
            raise ValueError("Ray geometry and mask shapes differ")
        residual = prediction - target
        terms = [profile_discrepancy(residual.movedim(axis + 2, -1),
                                     self.spacing[axis], self.length, mode)
                 for axis in range(3)]
        if include_oblique:
            for grid in self.grids:
                sampled = F.grid_sample(residual, grid.expand(residual.shape[0], -1, -1, -1, -1),
                                        mode="bilinear", padding_mode="zeros", align_corners=True)
                terms.append(profile_discrepancy(sampled, self.depth_step, self.length, mode))
        return torch.stack(terms).mean()


def wrong_foreground_bce(logits, labels):
    """Plain wrong-only foreground BCE, the strongest previous local-ray control.

Detached argmax eligibility and fixed whole-volume denominator reproduce the
strength-zero loss from the September 15 reevaluation (without its diagnostics).
"""
    target = labels != 0
    gate = ((logits.detach().argmax(1) != 0) != target)
    foreground_logit = torch.logsumexp(logits[:, 1:], 1) - logits[:, 0]
    return (F.binary_cross_entropy_with_logits(foreground_logit, target.to(logits),
                                               reduction="none") * gate).mean()
