"""Dense cut-band supervision from Swin's frozen final decoder features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CutVoxelHead(nn.Module):
    """Predict which foreground voxels touch the between-slice A/P interface."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 1, 1),
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        return self.network(decoder_features).squeeze(1)


def cut_band_target(foreground: torch.Tensor, cuts: torch.Tensor) -> torch.Tensor:
    """One-voxel-thick foreground bands on both sides of each cut."""
    if foreground.ndim != 4 or cuts.shape != foreground.shape[:1]:
        raise ValueError("expected foreground [B,X,Y,Z] and cuts [B]")
    y = torch.arange(foreground.shape[2], device=foreground.device).view(1, 1, -1, 1)
    return foreground.float() * ((y == cuts[:, None, None, None]) | (y == cuts[:, None, None, None] + 1)).float()


def cut_voxel_loss(logits: torch.Tensor, foreground: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Patient-balanced positive/negative BCE restricted to true foreground."""
    if logits.shape != foreground.shape or logits.shape != target.shape:
        raise ValueError("cut logits, foreground, and targets must have matching [B,X,Y,Z] shapes")
    foreground = foreground.float()
    positives = target.float() * foreground
    negatives = foreground - positives
    pos_loss = (F.softplus(-logits) * positives).sum(dim=(1, 2, 3)) / positives.sum(dim=(1, 2, 3)).clamp_min(1)
    neg_loss = (F.softplus(logits) * negatives).sum(dim=(1, 2, 3)) / negatives.sum(dim=(1, 2, 3)).clamp_min(1)
    return (0.5 * (pos_loss + neg_loss)).mean()


def candidate_cut_scores(logits: torch.Tensor, predicted_foreground: torch.Tensor) -> torch.Tensor:
    """Pool cut-voxel logits across each adjacent coronal slice pair."""
    if logits.ndim != 4 or logits.shape != predicted_foreground.shape:
        raise ValueError("expected matching [B,X,Y,Z] cut logits and foreground probabilities")
    weight = predicted_foreground.float().clamp(0, 1)
    profile = (logits * weight).sum(dim=(1, 3)) / weight.sum(dim=(1, 3)).clamp_min(1)
    return 0.5 * (profile[:, :-1] + profile[:, 1:])
