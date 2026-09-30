"""Coordinate-aware 2.5-D Convolutional Set Transformer teachers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .descriptors import DESCRIPTOR_NAMES


@dataclass
class CSTTeacherOutput:
    """Outputs used for probing and constraint construction."""

    descriptor_quantiles: torch.Tensor
    slice_profiles: torch.Tensor
    case_embedding: torch.Tensor
    element_embeddings: torch.Tensor


class SetConv2d(nn.Module):
    """SetConv2D with coordinate-conditioned attention tokens.

    The shared convolution preserves the spatial map for each set element.
    Pooled element vectors interact through multi-head self-attention and are
    added back as context-dependent channel biases.  Element coordinates move
    with their slices, so permuting the set still permutes the outputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        metadata_dim: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_channels % heads:
            raise ValueError("out_channels must be divisible by heads")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.token_norm = nn.LayerNorm(out_channels)
        self.attention = nn.MultiheadAttention(
            out_channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.activation = nn.GELU()

    def forward(
        self,
        elements: torch.Tensor,
        metadata: torch.Tensor,
        valid_elements: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if elements.ndim != 5:
            raise ValueError("elements must have shape [B, N, C, H, W]")
        batch, set_size, channels, height, width = elements.shape
        if metadata.shape[:2] != (batch, set_size):
            raise ValueError("metadata must share the [B, N] element axes")
        if valid_elements is not None and valid_elements.shape != (batch, set_size):
            raise ValueError("valid_elements must have shape [B, N]")
        if valid_elements is not None and torch.any(~valid_elements.bool().any(dim=1)):
            raise ValueError("every case must contain at least one valid set element")

        maps = self.conv(elements.reshape(batch * set_size, channels, height, width))
        out_channels, out_height, out_width = maps.shape[1:]
        maps = maps.reshape(batch, set_size, out_channels, out_height, out_width)
        tokens = maps.mean(dim=(-2, -1)) + self.metadata_projection(metadata)
        tokens = self.token_norm(tokens)
        padding_mask = None if valid_elements is None else ~valid_elements.bool()
        context, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        output = self.activation(maps + context[:, :, :, None, None])
        if valid_elements is not None:
            output = output * valid_elements[:, :, None, None, None]
        return output


class CSTEncoder2d(nn.Module):
    """Compact CST encoder suitable for the 260-case MSD experiment."""

    def __init__(
        self,
        in_channels: int,
        *,
        metadata_dim: int = 1,
        channels: tuple[int, ...] = (16, 32, 64),
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        current = in_channels
        for output in channels:
            blocks.append(
                SetConv2d(
                    current,
                    output,
                    metadata_dim=metadata_dim,
                    heads=heads,
                    dropout=dropout,
                )
            )
            current = output
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = channels[-1]

    def forward(
        self,
        elements: torch.Tensor,
        metadata: torch.Tensor,
        valid_elements: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = elements
        for index, block in enumerate(self.blocks):
            features = block(features, metadata, valid_elements)
            if index + 1 < len(self.blocks):
                batch, set_size, channels, height, width = features.shape
                pooled = F.max_pool2d(
                    features.reshape(batch * set_size, channels, height, width),
                    kernel_size=2,
                    stride=2,
                )
                features = pooled.reshape(batch, set_size, channels, pooled.shape[-2], pooled.shape[-1])

        element_embeddings = features.mean(dim=(-2, -1))
        if valid_elements is None:
            case_embedding = element_embeddings.mean(dim=1)
        else:
            weights = valid_elements.to(element_embeddings.dtype).unsqueeze(-1)
            case_embedding = (element_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return element_embeddings, case_embedding


class ConditionalQuantileHead(nn.Module):
    """Ordered lower/median/upper quantiles for each descriptor."""

    def __init__(self, input_dim: int, descriptor_count: int) -> None:
        super().__init__()
        self.descriptor_count = descriptor_count
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, descriptor_count * 3),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        raw = self.projection(embedding).reshape(-1, self.descriptor_count, 3)
        median = raw[..., 1]
        lower = median - F.softplus(raw[..., 0])
        upper = median + F.softplus(raw[..., 2])
        return torch.stack((lower, median, upper), dim=-1)


class CSTDescriptorTeacher(nn.Module):
    """MRI-only teacher for case intervals and per-slice area profiles."""

    def __init__(
        self,
        *,
        slab_channels: int = 3,
        metadata_dim: int = 1,
        channels: tuple[int, ...] = (16, 32, 64),
        heads: int = 4,
        descriptor_count: int = len(DESCRIPTOR_NAMES),
    ) -> None:
        super().__init__()
        self.encoder = CSTEncoder2d(
            slab_channels,
            metadata_dim=metadata_dim,
            channels=channels,
            heads=heads,
        )
        self.quantile_head = ConditionalQuantileHead(self.encoder.output_dim, descriptor_count)
        self.profile_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim),
            nn.GELU(),
            nn.Linear(self.encoder.output_dim, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image_slabs: torch.Tensor,
        metadata: torch.Tensor,
        valid_elements: torch.Tensor | None = None,
    ) -> CSTTeacherOutput:
        element_embeddings, case_embedding = self.encoder(image_slabs, metadata, valid_elements)
        return CSTTeacherOutput(
            descriptor_quantiles=self.quantile_head(case_embedding),
            slice_profiles=self.profile_head(element_embeddings),
            case_embedding=case_embedding,
            element_embeddings=element_embeddings,
        )


class CSTMaskAnomalyTeacher(nn.Module):
    """Contextual element-level detector for anatomically corrupted masks."""

    def __init__(
        self,
        *,
        slab_channels: int = 3,
        mask_channels: int = 2,
        metadata_dim: int = 1,
        channels: tuple[int, ...] = (16, 32, 64),
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = CSTEncoder2d(
            slab_channels + mask_channels,
            metadata_dim=metadata_dim,
            channels=channels,
            heads=heads,
        )
        self.anomaly_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(
        self,
        image_slabs: torch.Tensor,
        mask_slices: torch.Tensor,
        metadata: torch.Tensor,
        valid_elements: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_slabs.shape[:2] != mask_slices.shape[:2] or image_slabs.shape[-2:] != mask_slices.shape[-2:]:
            raise ValueError("image_slabs and mask_slices must align by element and pixel")
        elements = torch.cat((image_slabs, mask_slices), dim=2)
        element_embeddings, case_embedding = self.encoder(elements, metadata, valid_elements)
        return self.anomaly_head(element_embeddings).squeeze(-1), case_embedding
