"""Predict a distribution over the 63 between-slice A/P cuts in a 64³ ROI."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class APCutHead(nn.Module):
    """Small 3D-to-1D head conditioned on MRI and frozen Swin probabilities.

    Axis 1 of the input volume is coronal Y. The output index c denotes the
    boundary between slices c and c+1, with anterior at Y > c.
    """

    def __init__(self, image_enabled: bool = True):
        super().__init__()
        self.image_enabled = image_enabled
        self.spatial = nn.Sequential(
            nn.Conv3d(4, 16, 3, stride=(2, 1, 2), padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=(2, 1, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.sequence = nn.Sequential(
            nn.Conv1d(72, 64, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, 2),
        )

    def forward(self, image: torch.Tensor, probabilities: torch.Tensor, swin_cut: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or probabilities.ndim != 5 or image.shape[1] != 1 or probabilities.shape[1] != 3:
            raise ValueError("expected image [B,1,X,Y,Z] and probabilities [B,3,X,Y,Z]")
        if image.shape[0] != probabilities.shape[0] or image.shape[2:] != probabilities.shape[2:]:
            raise ValueError("image and probabilities must have matching batch/spatial shapes")
        if swin_cut.shape != (image.shape[0],):
            raise ValueError("expected one Swin cut per case")
        image = image if self.image_enabled else torch.zeros_like(image)
        encoded = self.spatial(torch.cat((image, probabilities), dim=1))
        pooled = torch.cat((encoded.mean(dim=(2, 4)), encoded.amax(dim=(2, 4))), dim=1)
        probability_profile = torch.cat((probabilities.mean(dim=(2, 4)), probabilities.amax(dim=(2, 4))), dim=1)
        size_y = image.shape[3]
        position = torch.arange(size_y, device=image.device, dtype=image.dtype).view(1, 1, -1)
        normalized_position = (2 * position / (size_y - 1) - 1).expand(image.shape[0], -1, -1)
        cut_hint = torch.exp(-0.5 * ((position - swin_cut[:, None, None] - 0.5) / 2.0).square())
        sequence = torch.cat((pooled, probability_profile, normalized_position, cut_hint), dim=1)
        return self.sequence(sequence).squeeze(1)


def soft_cut_targets(cut: torch.Tensor, num_cuts: int, sigma: float = 0.75) -> torch.Tensor:
    """Gaussian-smoothed categorical targets; adjacent cuts receive partial credit."""
    if sigma <= 0 or num_cuts < 1:
        raise ValueError("sigma and num_cuts must be positive")
    position = torch.arange(num_cuts, device=cut.device, dtype=torch.float32)
    return F.softmax(-0.5 * ((position[None, :] - cut[:, None].float()) / sigma).square(), dim=1)
