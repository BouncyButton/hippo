"""Shared result types for differentiable constraints."""

from dataclasses import dataclass, field
from typing import Any

import torch


def differentiable_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Return a graph-connected scalar zero without decollating tensor metadata.

    A batched MetaTensor retains its batch flag after flattening. Indexing that
    flattened object can therefore split metadata once per voxel. Its plain
    tensor view shares storage and preserves autograd, without that bookkeeping.
    The caller controls the dtype; the input must be nonempty.
    """
    plain = tensor.as_tensor() if hasattr(tensor, "as_tensor") else tensor
    return plain.reshape(-1)[0] * 0.0


@dataclass
class ConstraintResult:
    """Loss, fuzzy truth, raw metric, and optional diagnostic values."""

    loss: torch.Tensor
    truth: torch.Tensor
    value: torch.Tensor
    details: dict[str, Any] = field(default_factory=dict)
