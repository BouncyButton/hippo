"""Primitive metadata for the semantic-constraint DSL/search layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    inputs: tuple[str, ...]
    output: str
    description: str
    differentiable: bool = True
    parameters: tuple[str, ...] = ()


PRIMITIVES: dict[str, PrimitiveSpec] = {
    "volume": PrimitiveSpec(
        name="volume",
        inputs=("mask",),
        output="scalar_per_sample",
        parameters=("spacing",),
        description="Soft physical size of a class mask.",
    ),
    "centroid": PrimitiveSpec(
        name="centroid",
        inputs=("mask",),
        output="vector_per_sample",
        parameters=("spacing",),
        description="Soft center of mass in physical coordinates.",
    ),
    "distance": PrimitiveSpec(
        name="distance",
        inputs=("mask", "mask"),
        output="scalar_per_sample",
        parameters=("spacing",),
        description="Euclidean distance between soft centroids.",
    ),
    "contains": PrimitiveSpec(
        name="contains",
        inputs=("container_mask", "containee_mask"),
        output="scalar_per_sample",
        description="Fraction of containee mass lying inside the container.",
    ),
    "overlap": PrimitiveSpec(
        name="overlap",
        inputs=("mask", "mask"),
        output="scalar_per_sample",
        parameters=("mode",),
        description="Soft intersection, Dice, or IoU overlap.",
    ),
    "adjacent": PrimitiveSpec(
        name="adjacent",
        inputs=("mask", "mask"),
        output="scalar_per_sample",
        parameters=("radius",),
        description="Soft boundary contact score within a voxel radius.",
    ),
    "connectedness": PrimitiveSpec(
        name="connectedness",
        inputs=("mask",),
        output="scalar_per_sample",
        parameters=("steps", "temperature"),
        description="Differentiable surrogate for one connected component.",
    ),
    "compactness": PrimitiveSpec(
        name="compactness",
        inputs=("mask",),
        output="scalar_per_sample",
        parameters=("spacing",),
        description="Scale-normalized perimeter/area or surface/volume compactness.",
    ),
    "boundary_length": PrimitiveSpec(
        name="boundary_length",
        inputs=("mask",),
        output="scalar_per_sample",
        parameters=("spacing",),
        description="Soft total-variation boundary measure.",
    ),
    "entropy": PrimitiveSpec(
        name="entropy",
        inputs=("mask",),
        output="scalar_per_sample",
        parameters=("reduction",),
        description="Binary uncertainty of a soft class mask.",
    ),
}
