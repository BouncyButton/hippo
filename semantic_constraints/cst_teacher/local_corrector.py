"""Conservative 2.5-D local MRI/mask residual proposal for frozen Swin outputs."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


INPUT_CHANNELS = 23
CST_CHANNELS = slice(20, 22)
CENTER_PROB_CHANNELS = slice(11, 14)


def make_slice_contexts(
    image: np.ndarray,
    probabilities: np.ndarray,
    teacher_profiles: np.ndarray,
    positions: np.ndarray,
    *,
    use_cst: bool,
) -> np.ndarray:
    """Return (Y,23,X,Z): five MRI/probability planes plus CST profile."""
    image = np.asarray(image, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.int64)
    teacher_profiles = np.asarray(teacher_profiles, dtype=np.float32)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or probabilities.shape != (3, *image.shape):
        raise ValueError("expected aligned MRI (X,Y,Z) and Swin probabilities (3,X,Y,Z)")
    if teacher_profiles.shape != (len(positions), 2):
        raise ValueError("teacher profiles must match sampled positions and two classes")
    if positions[0] != 0 or positions[-1] != image.shape[1] - 1 or np.any(np.diff(positions) <= 0):
        raise ValueError("sampled positions must cover the full ordered coronal axis")
    if not np.isfinite(image).all() or not np.isfinite(probabilities).all() or not np.isfinite(teacher_profiles).all():
        raise ValueError("non-finite local-correction inputs")
    x_size, y_size, z_size = image.shape
    result = np.empty((y_size, INPUT_CHANNELS, x_size, z_size), dtype=np.float16)
    y = np.arange(y_size)
    for offset_index, offset in enumerate(range(-2, 3)):
        nearby = np.clip(y + offset, 0, y_size - 1)
        result[:, offset_index] = image[:, nearby, :].transpose(1, 0, 2)
        result[:, 5 + offset_index * 3: 8 + offset_index * 3] = probabilities[:, :, nearby, :].transpose(2, 0, 1, 3)
    for class_index in range(2):
        profile = np.interp(y, positions, teacher_profiles[:, class_index]).astype(np.float16)
        result[:, 20 + class_index] = profile[:, None, None] if use_cst else 0
    result[:, 22] = (2.0 * y / (y_size - 1) - 1.0)[:, None, None]
    return result


class LocalResidualCorrector(nn.Module):
    """Small image-and-probability CNN with bounded spatial logit residuals."""

    def __init__(self, channels: int = INPUT_CHANNELS, width: int = 24, max_residual: float = 2.0):
        super().__init__()
        self.max_residual = max_residual
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.GroupNorm(6, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.GroupNorm(6, width),
            nn.GELU(),
        )
        self.deep = nn.Sequential(
            nn.Conv2d(width, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(width + 32, width, 3, padding=1),
            nn.GroupNorm(6, width),
            nn.GELU(),
        )
        self.output = nn.Conv2d(width, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 4 or context.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"expected (batch,{INPUT_CHANNELS},X,Z) contexts")
        shallow = self.encoder(context)
        deep = F.interpolate(self.deep(shallow), size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.max_residual * torch.tanh(self.output(self.decoder(torch.cat((shallow, deep), dim=1))))
        baseline = context[:, CENTER_PROB_CHANNELS].clamp_min(1e-6)
        return baseline.log() + residual, residual


def correction_loss(logits: torch.Tensor, residual: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Emphasize Swin mistakes while penalizing unsupported large edits."""
    baseline_hard = baseline.argmax(dim=1)
    wrong = (baseline_hard != target).float()
    cross_entropy = F.cross_entropy(logits, target.long(), reduction="none")
    weighted_ce = (cross_entropy * (1.0 + 3.0 * wrong)).mean()
    probabilities = logits.softmax(dim=1)
    target_one_hot = F.one_hot(target.long(), 3).permute(0, 3, 1, 2).float()
    intersection = (probabilities[:, 1:3] * target_one_hot[:, 1:3]).sum(dim=(0, 2, 3))
    denominator = probabilities[:, 1:3].sum(dim=(0, 2, 3)) + target_one_hot[:, 1:3].sum(dim=(0, 2, 3))
    foreground_dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return weighted_ce + 0.10 * foreground_dice_loss + 0.002 * residual.square().mean()


def case_dice(labels: np.ndarray, prediction: np.ndarray) -> float:
    """Mean whole-case hard Dice of anterior and posterior classes."""
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if labels.shape != prediction.shape:
        raise ValueError("label and prediction shapes differ")
    scores = []
    for class_index in (1, 2):
        truth, guess = labels == class_index, prediction == class_index
        scores.append((2 * np.count_nonzero(truth & guess) + 1e-8) / (truth.sum() + guess.sum() + 1e-8))
    return float(np.mean(scores))


def apply_slice_policy(baseline: np.ndarray, proposal: np.ndarray, risk: np.ndarray, fraction: float) -> np.ndarray:
    """Accept proposal only on a fixed highest-risk fraction of coronal slices."""
    if baseline.shape != proposal.shape or baseline.ndim != 3 or risk.shape != (baseline.shape[1],):
        raise ValueError("case prediction and coronal risk shapes do not align")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    if fraction == 0:
        return baseline.copy()
    if fraction == 1:
        return proposal.copy()
    count = max(1, int(np.ceil(fraction * len(risk))))
    selected = np.argpartition(risk, -count)[-count:]
    result = baseline.copy()
    result[:, selected, :] = proposal[:, selected, :]
    return result
