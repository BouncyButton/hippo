"""Signed face-contrast supervision on existing segmentation probabilities."""
import torch
from torch import nn

from .bands.outer_boundary import build_boundary_bands


def signed_face_error(probability: torch.Tensor, foreground: torch.Tensor,
                      support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-case sum/count over positive-axis faces, with both ends in support.

    Inputs have shape [B,1,D,H,W]. No wraparound, degree weights, or learned head.
    """
    if probability.ndim != 5 or probability.shape[1] != 1:
        raise ValueError('Expected [B,1,D,H,W] probabilities.')
    if probability.shape != foreground.shape or support.shape != foreground.shape:
        raise ValueError('Probability, foreground and support shapes must match.')
    residual = probability - foreground.to(probability.dtype)
    total = residual.flatten(1).sum(1) * 0
    count = torch.zeros(probability.shape[0], device=probability.device, dtype=torch.long)
    for axis in (2, 3, 4):
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = support[left].bool() & support[right].bool()
        total = total + ((residual[left] - residual[right]).square() * valid).flatten(1).sum(1)
        count = count + valid.flatten(1).sum(1)
    return total, count


class BoundaryEdgeConsistencyLoss(nn.Module):
    """Uniform face mean per valid case in the existing two-step GT bands."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 3:
            raise ValueError('Expected three-class logits [B,3,D,H,W].')
        if labels.ndim == 4:
            labels = labels[:, None]
        if labels.shape != logits[:, :1].shape:
            raise ValueError('Labels must have shape [B,1,D,H,W].')
        foreground = labels > 0
        with torch.no_grad():
            inner, outer = build_boundary_bands(foreground, steps=2)
        probability = logits.float().softmax(1)[:, 1:].sum(1, keepdim=True)
        total, count = signed_face_error(probability, foreground, inner | outer)
        valid = count > 0
        return ((total / count.clamp_min(1)) * valid).sum() / valid.sum().clamp_min(1)
