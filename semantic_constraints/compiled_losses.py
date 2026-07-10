"""Compile selected semantic-constraint specs into differentiable fuzzy losses."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_constraints.primitives import (
    adjacent,
    boundary_length,
    compactness,
    connectedness,
    contains,
    distance,
    entropy,
    overlap,
    volume,
)


DEFAULT_SELECTED_CONSTRAINTS = Path(__file__).resolve().parent / "probe_outputs" / "selected_constraints.json"
CLASS_ARG_RE = re.compile(r"^class_(?P<class_id>\d+)$")


PRIMITIVE_FUNCTIONS = {
    "adjacent": adjacent,
    "boundary_length": boundary_length,
    "compactness": compactness,
    "connectedness": connectedness,
    "contains": contains,
    "distance": distance,
    "entropy": entropy,
    "overlap": overlap,
    "volume": volume,
}


SPACING_PRIMITIVES = {"boundary_length", "compactness", "distance", "volume"}


def load_constraint_specs(path: str | Path = DEFAULT_SELECTED_CONSTRAINTS) -> list[dict[str, Any]]:
    """Load selected constraint specs from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError(f"{path} does not contain a 'constraints' list.")
    return constraints


def ste_clamp01(x: torch.Tensor) -> torch.Tensor:
    """Clamp to [0, 1] in the forward pass with identity backward gradient."""
    clipped = x.clamp(0.0, 1.0)
    return x + (clipped - x).detach()


def parse_class_id(arg: str) -> int:
    match = CLASS_ARG_RE.match(arg)
    if not match:
        raise ValueError(f"Expected class argument like 'class_1', got {arg!r}.")
    return int(match.group("class_id"))


def class_mask(probs: torch.Tensor, arg: str) -> torch.Tensor:
    class_id = parse_class_id(arg)
    if class_id >= probs.shape[1]:
        raise ValueError(f"{arg} requested class index {class_id}, but probs has {probs.shape[1]} channels.")
    return probs[:, class_id]


def as_scalar_tensor(value: Any, like: torch.Tensor, default: float) -> torch.Tensor:
    if value is None:
        value = default
    return torch.as_tensor(float(value), device=like.device, dtype=like.dtype)


def normalized_violation(
    values: torch.Tensor,
    direction: str,
    alpha: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if direction == "<=":
        gap = values - alpha
    elif direction == ">=":
        gap = alpha - values
    else:
        raise ValueError(f"Unknown constraint direction: {direction!r}.")

    violation = F.relu(gap)
    return violation, violation / scale.clamp_min(eps)


def fuzzy_truth(normalized: torch.Tensor, fuzzy: dict[str, Any], eps: float) -> torch.Tensor:
    mode = fuzzy.get("mode", "soft_exp")
    margin = max(float(fuzzy.get("margin", 1.0)), eps)
    beta = float(fuzzy.get("beta", 4.0))
    z = normalized / margin

    if mode == "soft_exp":
        return torch.exp(-beta * z.square())
    if mode == "lukasiewicz":
        return (1.0 - z).clamp(0.0, 1.0)
    if mode == "lukasiewicz_ste":
        return ste_clamp01(1.0 - z)
    if mode == "hinge":
        return (1.0 - z).clamp(0.0, 1.0)
    raise ValueError(f"Unknown fuzzy mode: {mode!r}.")


class SemanticConstraintLoss(nn.Module):
    """Apply selected semantic constraints to segmentation probabilities.

    Args:
        constraints: List of selected constraint specs.
        spacing: Optional physical voxel spacing passed to geometric primitives.
        input_is_logits: If true, ``forward`` applies ``softmax(dim=1)`` first.
        eps: Numerical stability constant.

    Forward input shape is ``(B, C, *spatial)``. The return value is a dictionary
    with total loss and per-constraint diagnostics.
    """

    def __init__(
        self,
        constraints: list[dict[str, Any]],
        spacing: float | tuple[float, ...] | list[float] | None = None,
        input_is_logits: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.constraints = constraints
        self.spacing = tuple(spacing) if isinstance(spacing, list) else spacing
        self.input_is_logits = input_is_logits
        self.eps = eps

    @classmethod
    def from_json(
        cls,
        path: str | Path = DEFAULT_SELECTED_CONSTRAINTS,
        spacing: float | tuple[float, ...] | list[float] | None = None,
        input_is_logits: bool = False,
        eps: float = 1e-8,
    ) -> "SemanticConstraintLoss":
        return cls(
            constraints=load_constraint_specs(path),
            spacing=spacing,
            input_is_logits=input_is_logits,
            eps=eps,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        if inputs.ndim < 4:
            raise ValueError("Expected inputs shaped (B, C, *spatial).")

        probs = torch.softmax(inputs, dim=1) if self.input_is_logits else inputs
        if not self.constraints:
            zero = probs.sum() * 0.0
            return {
                "loss": zero,
                "weighted_loss": {},
                "unweighted_loss": {},
                "truth": {},
                "violation": {},
                "value": {},
            }

        weighted_losses = {}
        unweighted_losses = {}
        truths = {}
        violations = {}
        values = {}

        total_loss = probs.sum() * 0.0
        for spec in self.constraints:
            result = self.evaluate_one(probs, spec)
            name = spec["name"]
            weighted_losses[name] = result["weighted_loss"]
            unweighted_losses[name] = result["unweighted_loss"]
            truths[name] = result["truth"]
            violations[name] = result["violation"]
            values[name] = result["value"]
            total_loss = total_loss + result["weighted_loss"]

        return {
            "loss": total_loss,
            "weighted_loss": weighted_losses,
            "unweighted_loss": unweighted_losses,
            "truth": truths,
            "violation": violations,
            "value": values,
        }

    def evaluate_one(self, probs: torch.Tensor, spec: dict[str, Any]) -> dict[str, torch.Tensor]:
        primitive = spec["primitive"]
        if primitive not in PRIMITIVE_FUNCTIONS:
            raise ValueError(f"Unsupported primitive: {primitive!r}.")

        masks = [class_mask(probs, arg) for arg in spec.get("args", [])]
        values = self.apply_primitive(primitive, masks, spec)
        alpha = as_scalar_tensor(spec.get("alpha"), values, default=0.0)
        scale = as_scalar_tensor(spec.get("scale"), values, default=1.0)
        raw_violation, normalized = normalized_violation(
            values=values,
            direction=spec["direction"],
            alpha=alpha,
            scale=scale,
            eps=self.eps,
        )

        fuzzy = spec.get("fuzzy", {})
        truth = fuzzy_truth(normalized, fuzzy=fuzzy, eps=self.eps)
        mode = fuzzy.get("mode", "soft_exp")
        per_sample_loss = normalized if mode == "hinge" else 1.0 - truth
        unweighted_loss = per_sample_loss.mean()
        lam = as_scalar_tensor(spec.get("lambda"), values, default=1.0)
        weighted_loss = lam * unweighted_loss

        return {
            "weighted_loss": weighted_loss,
            "unweighted_loss": unweighted_loss,
            "truth": truth.mean(),
            "violation": raw_violation.mean(),
            "value": values.mean(),
        }

    def apply_primitive(self, primitive: str, masks: list[torch.Tensor], spec: dict[str, Any]) -> torch.Tensor:
        fn = PRIMITIVE_FUNCTIONS[primitive]
        kwargs = dict(spec.get("primitive_kwargs", {}))
        if primitive in SPACING_PRIMITIVES and "spacing" not in kwargs:
            kwargs["spacing"] = spec.get("spacing", self.spacing)

        expected_args = {
            "adjacent": 2,
            "boundary_length": 1,
            "compactness": 1,
            "connectedness": 1,
            "contains": 2,
            "distance": 2,
            "entropy": 1,
            "overlap": 2,
            "volume": 1,
        }[primitive]
        if len(masks) != expected_args:
            raise ValueError(f"{primitive} expects {expected_args} class masks, got {len(masks)}.")
        return fn(*masks, **kwargs)


def semantic_constraint_loss_from_json(
    path: str | Path = DEFAULT_SELECTED_CONSTRAINTS,
    spacing: float | tuple[float, ...] | list[float] | None = None,
    input_is_logits: bool = False,
) -> SemanticConstraintLoss:
    """Convenience constructor for training scripts."""
    return SemanticConstraintLoss.from_json(path=path, spacing=spacing, input_is_logits=input_is_logits)
