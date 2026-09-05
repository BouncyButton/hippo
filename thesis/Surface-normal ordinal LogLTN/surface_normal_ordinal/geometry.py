"""Spacing-aware surface points, normals, and sampling coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage


class Interface(str, Enum):
    """Supported typed interfaces."""

    OUTER = "outer"
    ANTERIOR_POSTERIOR = "ap"


@dataclass(frozen=True)
class SurfaceSamples:
    """Surface locations and oriented physical normals in array-axis order."""

    points_voxel: np.ndarray
    normals_physical: np.ndarray
    spacing: tuple[float, float, float]
    interface: Interface

    def coordinates_at(self, offsets_mm: np.ndarray | list[float]) -> np.ndarray:
        """Return ``[N, S, 3]`` voxel coordinates at physical ray offsets."""

        offsets = np.asarray(offsets_mm, dtype=np.float64)
        if offsets.ndim != 1:
            raise ValueError("offsets_mm must be one-dimensional.")
        spacing = np.asarray(self.spacing, dtype=np.float64)
        return (
            self.points_voxel[:, None, :]
            + offsets[None, :, None]
            * self.normals_physical[:, None, :]
            / spacing[None, None, :]
        )


def validate_spacing(spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in spacing)
    if len(values) != 3 or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("spacing must contain three finite positive values.")
    return values


def signed_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Return a signed distance field that is negative inside and positive outside."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any() or mask.all():
        raise ValueError("mask must be a nonempty, non-full 3-D binary array.")
    spacing = validate_spacing(spacing)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return outside - inside


def _outer_face_centres(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return foreground/background face centres and outward face directions."""

    points: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    ndim = mask.ndim
    for axis in range(ndim):
        for direction in (-1, 1):
            foreground_slice = [slice(None)] * ndim
            neighbour_slice = [slice(None)] * ndim
            if direction > 0:
                foreground_slice[axis] = slice(0, -1)
                neighbour_slice[axis] = slice(1, None)
            else:
                foreground_slice[axis] = slice(1, None)
                neighbour_slice[axis] = slice(0, -1)
            faces = mask[tuple(foreground_slice)] & ~mask[tuple(neighbour_slice)]
            indices = np.argwhere(faces)
            if indices.size == 0:
                continue
            if direction < 0:
                indices[:, axis] += 1
            centres = indices.astype(np.float64)
            centres[:, axis] += 0.5 * direction
            face_direction = np.zeros((len(indices), ndim), dtype=np.float64)
            face_direction[:, axis] = direction
            points.append(centres)
            directions.append(face_direction)
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(points), np.concatenate(directions)


def _ap_face_centres(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return A/P face centres with directions oriented from class 1 to class 2."""

    points: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    ndim = labels.ndim
    for axis in range(ndim):
        left_slice = [slice(None)] * ndim
        right_slice = [slice(None)] * ndim
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        left = labels[tuple(left_slice)]
        right = labels[tuple(right_slice)]
        for left_label, right_label, direction in ((1, 2, 1), (2, 1, -1)):
            faces = (left == left_label) & (right == right_label)
            indices = np.argwhere(faces)
            if indices.size == 0:
                continue
            centres = indices.astype(np.float64)
            centres[:, axis] += 0.5
            oriented = np.zeros((len(indices), ndim), dtype=np.float64)
            oriented[:, axis] = direction
            points.append(centres)
            directions.append(oriented)
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(points), np.concatenate(directions)


def _sdf_normals(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    points: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    sdf = signed_distance(mask, spacing)
    gradients = np.stack(np.gradient(sdf, *spacing, edge_order=1), axis=0)
    sampled = np.stack(
        [
            ndimage.map_coordinates(
                gradients[axis], points.T, order=1, mode="nearest", prefilter=False
            )
            for axis in range(3)
        ],
        axis=1,
    )
    norm = np.linalg.norm(sampled, axis=1, keepdims=True)
    unreliable = (~np.isfinite(norm[:, 0])) | (norm[:, 0] < 1e-6)
    sampled = sampled / np.maximum(norm, 1e-6)
    sampled[unreliable] = fallback[unreliable]
    # Numerical gradients can point incorrectly around sharp corners.  The face
    # direction is the authoritative discrete inside-to-outside orientation.
    flip = np.sum(sampled * fallback, axis=1) < 0
    sampled[flip] *= -1.0
    return sampled


def _valid_for_radius(
    points: np.ndarray,
    normals: np.ndarray,
    spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
    radius_mm: float,
) -> np.ndarray:
    displacement = radius_mm * normals / np.asarray(spacing)[None, :]
    low = points - displacement
    high = points + displacement
    upper = np.asarray(shape, dtype=np.float64) - 1.0
    return np.all((low >= 0.0) & (low <= upper) & (high >= 0.0) & (high <= upper), axis=1)


def build_surface_samples(
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    interface: Interface | str = Interface.OUTER,
    radius_mm: float = 3.0,
    max_points: int | None = 4096,
    seed: int = 0,
) -> SurfaceSamples:
    """Construct deterministic, oriented surface samples for one label volume."""

    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels must have shape [D, H, W].")
    if radius_mm <= 0 or not np.isfinite(radius_mm):
        raise ValueError("radius_mm must be finite and positive.")
    spacing = validate_spacing(spacing)
    interface = Interface(interface)
    if interface is Interface.OUTER:
        mask = labels > 0
        points, face_directions = _outer_face_centres(mask)
        if len(points):
            normals = _sdf_normals(mask, spacing, points, face_directions)
        else:
            normals = face_directions
    else:
        points, normals = _ap_face_centres(labels)
    if len(points) == 0:
        raise ValueError(f"No {interface.value} interface faces were found.")
    valid = _valid_for_radius(points, normals, spacing, labels.shape, radius_mm)
    points = points[valid]
    normals = normals[valid]
    if len(points) == 0:
        raise ValueError("No surface samples remain after the in-bounds ray guard.")
    if max_points is not None:
        if max_points <= 0:
            raise ValueError("max_points must be positive or None.")
        if len(points) > max_points:
            selected = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
            selected.sort()
            points = points[selected]
            normals = normals[selected]
    return SurfaceSamples(points, normals, spacing, interface)
