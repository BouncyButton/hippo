"""Shared result types for differentiable constraints."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ConstraintResult:
    """Loss, fuzzy truth, raw metric, and optional diagnostic values."""

    loss: torch.Tensor
    truth: torch.Tensor
    value: torch.Tensor
    details: dict[str, Any] = field(default_factory=dict)
