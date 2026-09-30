"""Local, bounded foreground-logit refinement without a ray-presence head.

The support is defined by the input segmentation, not labels. It is frozen for
one forward pass and is never recomputed from the corrected output. Guarantees
concern this pass's input prediction; they are not anatomical guarantees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalBoundaryConfig:
    radius_mm: float = 2.0
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_logit_shift: float = 2.0
    feature_channels: int = 24
    projected_channels: int = 4
    hidden_channels: int = 24

    def __post_init__(self):
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError('radius_mm must be finite and positive')
        if len(self.spacing_mm) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.spacing_mm):
            raise ValueError('spacing_mm must contain three finite positive values')
        if not math.isfinite(self.max_logit_shift) or self.max_logit_shift <= 0:
            raise ValueError('max_logit_shift must be finite and positive')
        if any(v <= 0 for v in (self.feature_channels, self.projected_channels, self.hidden_channels)):
            raise ValueError('channel counts must be positive')


def _neighborhood(config: LocalBoundaryConfig) -> torch.Tensor:
    axes = [torch.arange(-math.floor(config.radius_mm/s), math.floor(config.radius_mm/s)+1)*s
            for s in config.spacing_mm]
    coordinates = torch.meshgrid(*axes, indexing='ij')
    squared_distance = sum(c.square() for c in coordinates)
    return (squared_distance <= config.radius_mm**2 + 1e-6).float()[None, None]


class LocalBoundaryCorrection(nn.Module):
    """Correct background logits within a prediction-defined physical band.

    Input shapes: logits [B,3,X,Y,Z], decoder features [B,F,X,Y,Z]. The band
    includes voxels with an opposite foreground/background label within radius
    (centre-to-centre Euclidean distance). Out-of-volume space is ignored. The
    A/P interface is not part of this union-boundary definition.
    """

    def __init__(self, config: LocalBoundaryConfig = LocalBoundaryConfig()):
        super().__init__()
        self.config = config
        self.register_buffer('neighborhood', _neighborhood(config), persistent=False)
        self.project = nn.Conv3d(config.feature_channels, config.projected_channels, 1)
        h = config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv3d(config.projected_channels + 3, h, 3, padding=1), nn.GELU(),
            nn.Conv3d(h, h, 3, padding=1), nn.GELU())
        self.correction = nn.Conv3d(h, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    @torch.no_grad()
    def support(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 3:
            raise ValueError('logits must have shape [B,3,X,Y,Z]')
        with torch.autocast(device_type=logits.device.type, enabled=False):
            foreground = (logits.detach().argmax(1, keepdim=True) > 0).float()
            kernel = self.neighborhood.to(device=logits.device, dtype=torch.float32)
            padding = tuple(v//2 for v in kernel.shape[-3:])
            nearby_foreground = F.conv3d(foreground, kernel, padding=padding)
            nearby_background = F.conv3d(1-foreground, kernel, padding=padding)
            # Padding does not create an artificial boundary at the crop edge.
            return torch.where(foreground.bool(), nearby_background > .5, nearby_foreground > .5)

    def forward(self, logits: torch.Tensor, features: torch.Tensor,
                *, detach_inputs: bool = True) -> dict[str, torch.Tensor]:
        band = self.support(logits)
        expected = (logits.shape[0], self.config.feature_channels, *logits.shape[2:])
        if tuple(features.shape) != expected or features.device != logits.device:
            raise ValueError('decoder features must match logits spatial shape/device and configured channels')
        with torch.autocast(device_type=logits.device.type, enabled=False):
            raw = logits.float()
            source = raw.detach() if detach_inputs else raw
            decoder = features.detach().float() if detach_inputs else features.float()
            foreground_probability = (source[:, 1:].logsumexp(1, keepdim=True)-source[:, :1]).sigmoid()
            margin = source[:, 1:].amax(1, keepdim=True)-source[:, :1]
            foreground = (source.argmax(1, keepdim=True) > 0).float()
            inputs = torch.cat((self.project(decoder), foreground_probability,
                                (margin/self.config.max_logit_shift).clamp(-4,4), foreground), 1)
            h = self.correction(self.stem(inputs))
            cap = self.config.max_logit_shift
            delta = band.to(h.dtype) * cap * torch.tanh(h/cap)
            refined = torch.cat((raw[:, :1]-delta, raw[:, 1:]), 1)
        return dict(raw=raw, refined=refined, support=band, delta=delta)


class FrozenBackboneBoundaryRefiner(nn.Module):
    """Train only the refiner; capture the input to backbone.out as features.

    This deliberately freezes the supplied backbone. Decoder capture is scoped
    to each forward call, including cleanup on failure; no persistent hook is
    installed. Use a separate backbone instance if another experiment needs it.
    """

    def __init__(self, backbone: nn.Module,
                 config: LocalBoundaryConfig = LocalBoundaryConfig()):
        super().__init__()
        if not hasattr(backbone, 'out'):
            raise ValueError('backbone must expose its final segmentation layer as .out')
        self.backbone = backbone.requires_grad_(False).eval()
        self.refiner = LocalBoundaryCorrection(config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        captured = []
        def capture(_module, args):
            x = args[0]
            captured.append(x.as_tensor() if hasattr(x, 'as_tensor') else x)
        hook = self.backbone.out.register_forward_pre_hook(capture)
        try:
            self.backbone.eval()
            with torch.no_grad():
                logits = self.backbone(image)
                logits = logits.as_tensor() if hasattr(logits, 'as_tensor') else logits
            if len(captured) != 1:
                raise RuntimeError('expected exactly one final decoder output')
            return self.refiner(logits, captured[0], detach_inputs=True)
        finally:
            hook.remove()
