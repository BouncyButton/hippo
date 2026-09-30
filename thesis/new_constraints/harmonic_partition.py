"""Offline image-weighted harmonic A/P partition on a voxel graph.

This SciPy reference solver is a research diagnostic, not an autograd loss.
Boundary anchors are explicit; no target labels enter the solve. Components
without both anchor types retain fallback values instead of acquiring invented
seeds. Foreground support is fixed by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class HarmonicResult:
    field: np.ndarray
    active_seeds: np.ndarray
    solved_support: np.ndarray
    diagnostics: dict


def make_cores(support: np.ndarray, *, mode: str, cut: int | None = None,
               spacing_y: float = 1.0) -> np.ndarray:
    """Return -1 for unknown, 0 anterior, 1 posterior; axis 1 is RAS A/P."""
    support = np.asarray(support, dtype=bool)
    if support.ndim != 3 or min(support.shape) == 0:
        raise ValueError("support must be a nonempty 3D array")
    if not math.isfinite(spacing_y) or spacing_y <= 0:
        raise ValueError("spacing_y must be finite and positive")
    if mode not in {"ends", "band"}:
        raise ValueError("mode must be ends or band")
    if mode == "band" and (cut is None or not 0 < cut < support.shape[1]):
        raise ValueError("band mode requires an interior coronal cut")
    seeds = np.full(support.shape, -1, dtype=np.int8)
    occupied = np.flatnonzero(support.any(axis=(0, 2)))
    if len(occupied) < 2:
        return seeds
    interior = ndimage.binary_erosion(support, structure=ndimage.generate_binary_structure(3, 1))
    y = np.arange(support.shape[1])[None, :, None]
    if mode == "ends":
        low, high = occupied[[0, -1]]
        posterior = y <= low + .2 * (high - low)
        anterior = y >= high - .2 * (high - low)
    else:
        distance = (y - (cut - .5)) * spacing_y
        posterior, anterior = distance <= -4.0, distance >= 4.0
    seeds[interior & posterior] = 1
    seeds[interior & anterior] = 0
    return seeds


def build_laplacian(image: np.ndarray, support: np.ndarray, *, beta: float,
                    spacing: tuple[float, float, float] = (1., 1., 1.)) -> tuple[sparse.csr_matrix, float]:
    """Build symmetric six-neighbor conductances, scaled by physical spacing."""
    image = np.asarray(image, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if image.ndim != 3 or image.shape != support.shape or min(image.shape) == 0:
        raise ValueError("image and support must have matching nonempty 3D shapes")
    if not np.isfinite(image).all() or not math.isfinite(beta) or beta < 0:
        raise ValueError("image must be finite; beta must be finite and nonnegative")
    if len(spacing) != 3 or any(not math.isfinite(h) or h <= 0 for h in spacing):
        raise ValueError("spacing must contain three finite positive values")
    intensities = image[support]
    scale = float(np.subtract(*np.percentile(intensities, [75, 25]))) if intensities.size else 1.
    if scale <= 1e-12:
        scale = float(intensities.std()) if intensities.size else 1.
    if scale <= 1e-12:
        scale = 1.
    n = int(support.sum())
    indices = np.full(image.shape, -1, dtype=np.int64)
    indices[support] = np.arange(n)
    rows, columns, weights = [], [], []
    for axis, h in enumerate(spacing):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        valid = support[a] & support[b]
        i, j = indices[a][valid], indices[b][valid]
        difference = (image[a][valid] - image[b][valid]) / scale
        w = (1e-4 + (1. - 1e-4) * np.exp(-beta * difference**2)) / h**2
        rows.extend((i, j))
        columns.extend((j, i))
        weights.extend((w, w))
    adjacency = sparse.csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n))
    return sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency, scale


def harmonic_partition(image: np.ndarray, support: np.ndarray, seeds: np.ndarray,
                       fallback: np.ndarray, *, beta: float = 0.,
                       spacing: tuple[float, float, float] = (1., 1., 1.)) -> HarmonicResult:
    """Solve Dirichlet systems and retain fallback on insufficiently seeded components."""
    support = np.asarray(support, dtype=bool)
    seeds = np.asarray(seeds)
    fallback = np.asarray(fallback, dtype=np.float64)
    laplacian, scale = build_laplacian(image, support, beta=beta, spacing=spacing)
    if seeds.shape != support.shape or fallback.shape != support.shape:
        raise ValueError("seeds and fallback must match support")
    if not np.isin(seeds, (-1, 0, 1)).all() or np.any((seeds != -1) & ~support):
        raise ValueError("seeds must be -1/0/1 and all anchors must lie in support")
    if not np.isfinite(fallback).all() or np.any((fallback < 0) | (fallback > 1)):
        raise ValueError("fallback must contain finite values in [0,1]")
    components, count = ndimage.label(support, ndimage.generate_binary_structure(3, 1))
    node_components, node_seeds = components[support], seeds[support]
    values = fallback[support].copy()
    active = np.full(seeds.shape, -1, dtype=np.int8)
    solved = np.zeros(support.shape, dtype=bool)
    active_nodes = np.full(len(values), -1, dtype=np.int8)
    solved_nodes = np.zeros(len(values), dtype=bool)
    residual, solved_components = 0., 0
    for component in range(1, count + 1):
        nodes = np.flatnonzero(node_components == component)
        marked = node_seeds[nodes]
        if not ((marked == 0).any() and (marked == 1).any()):
            continue
        known = nodes[marked >= 0]
        unknown = nodes[marked < 0]
        values[known] = node_seeds[known]
        if len(unknown):
            matrix = laplacian[unknown][:, unknown].tocsr()
            rhs = -laplacian[unknown][:, known] @ values[known]
            values[unknown] = spsolve(matrix, rhs)
            residual = max(residual, float(np.max(np.abs(matrix @ values[unknown] - rhs))))
        solved_nodes[nodes] = True
        active_nodes[known] = node_seeds[known]
        solved_components += 1
    if not np.isfinite(values).all() or np.any((values < -1e-8) | (values > 1+1e-8)) or residual > 1e-7:
        raise RuntimeError("Harmonic solution violated finiteness, maximum principle, or residual tolerance")
    field = fallback.copy()
    field[support] = np.clip(values, 0, 1)
    active[support], solved[support] = active_nodes, solved_nodes
    return HarmonicResult(field, active, solved, {
        "components": int(count), "solved_components": solved_components,
        "support_voxels": int(support.sum()), "solved_voxels": int(solved.sum()),
        "proposed_seed_voxels": int((seeds >= 0).sum()),
        "active_seed_voxels": int((active >= 0).sum()),
        "max_linear_residual": residual, "intensity_scale": scale,
    })


def field_labels(result: HarmonicResult, prediction: np.ndarray) -> np.ndarray:
    """Threshold solved nodes; retain original labels on ties and unsolved nodes."""
    prediction = np.asarray(prediction)
    if prediction.shape != result.field.shape or not np.isin(prediction, (0, 1, 2)).all():
        raise ValueError("prediction must be a matching 0/1/2 label array")
    if np.any(result.solved_support & (prediction == 0)):
        raise ValueError("Solved support must lie inside predicted foreground")
    out = prediction.copy()
    out[result.solved_support & (result.field < .5-1e-12)] = 1
    out[result.solved_support & (result.field > .5+1e-12)] = 2
    return out
