"""Generic differentiable primitives for semantic-constraint discovery."""

from .compiled_losses import SemanticConstraintLoss, semantic_constraint_loss_from_json
from .primitives import (
    adjacent,
    boundary_length,
    centroid,
    compactness,
    connectedness,
    contains,
    distance,
    entropy,
    overlap,
    volume,
)
from .registry import PRIMITIVES, PrimitiveSpec

__all__ = [
    "PRIMITIVES",
    "PrimitiveSpec",
    "SemanticConstraintLoss",
    "adjacent",
    "boundary_length",
    "centroid",
    "compactness",
    "connectedness",
    "contains",
    "distance",
    "entropy",
    "overlap",
    "volume",
    "semantic_constraint_loss_from_json",
]
