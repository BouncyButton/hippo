"""Interpretable 2-D shape descriptors for the uncal-fold discovery audit.

All descriptor values are derived from the *union* of the two hippocampal
labels.  The A/P labels are read only by ``best_fit_first_anterior_slice`` to
produce the evaluation target.  This separation is deliberate: the audit asks
whether outer shape contains information about the annotated division.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


EPS = 1e-8

BASE_FEATURE_NAMES = (
    "area",
    "width",
    "height",
    "aspect_height_over_width",
    "perimeter_over_sqrt_area",
    "solidity",
    "eccentricity",
    "top_asymmetry_abs",
    "top_left_minus_right",
    "height_asymmetry_abs",
    "height_left_minus_right",
    "top_notch_left",
    "top_notch_right",
    "top_notch_max",
    "top_roughness",
    "column_multirun_fraction",
    "row_multirun_fraction",
    "hole_fraction",
)

FOLD_COMPONENTS = (
    "top_asymmetry_abs",
    "height_asymmetry_abs",
    "top_notch_max",
    "top_roughness",
    "column_multirun_fraction",
)


@dataclass(frozen=True)
class CaseFeatures:
    """Per-slice union geometry and the label-derived evaluation target."""

    case_name: str
    low: int
    high: int
    target_cut: int
    target_cost: int
    second_best_gap: int
    features: dict[int, dict[str, float]]

    @property
    def candidates(self) -> list[int]:
        return list(range(self.low + 1, self.high + 1))


def _count_runs(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return 0
    padded = np.pad(values.astype(np.int8), 1)
    return int(np.count_nonzero(np.diff(padded) == 1))


def _valley_depth(profile: np.ndarray) -> float:
    """Largest point lying below peaks on both sides of a 1-D profile."""

    profile = np.asarray(profile, dtype=np.float64)
    if profile.size < 3:
        return 0.0
    left_peak = np.maximum.accumulate(profile)
    right_peak = np.maximum.accumulate(profile[::-1])[::-1]
    depth = np.minimum(left_peak, right_peak) - profile
    depth[[0, -1]] = 0.0
    return float(max(0.0, depth.max(initial=0.0)))


def _perimeter(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.int8), 1)
    horizontal = np.abs(np.diff(padded, axis=0)).sum()
    vertical = np.abs(np.diff(padded, axis=1)).sum()
    return float(horizontal + vertical)


def _convex_hull_area(mask: np.ndarray) -> float:
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] < 3:
        return float(coordinates.shape[0])
    # Pixel corners avoid the systematic under-estimate obtained from a hull of
    # pixel centres, especially on the small native Task04 slices.
    corners = np.concatenate(
        [coordinates + offset for offset in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))],
        axis=0,
    )
    try:
        return float(ConvexHull(corners).volume)
    except QhullError:
        return float(coordinates.shape[0])


def extract_slice_features(mask: np.ndarray) -> dict[str, float]:
    """Measure one native coronal union-mask slice.

    Array axis 0 is R/L and axis 1 is I/S after the verified RAS loading used by
    this audit.  Left/right variants are retained because Task04 does not carry
    reliable laterality metadata; side-invariant features use an absolute value
    or maximum across the two sides.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a non-empty 2-D boolean array")

    coordinates = np.argwhere(mask)
    x0, z0 = coordinates.min(axis=0)
    x1, z1 = coordinates.max(axis=0)
    crop = mask[x0 : x1 + 1, z0 : z1 + 1]
    width, height = crop.shape
    area = float(crop.sum())

    occupied_x = np.flatnonzero(crop.any(axis=1))
    top = np.array([np.flatnonzero(crop[x]).max() for x in occupied_x], dtype=np.float64)
    bottom = np.array([np.flatnonzero(crop[x]).min() for x in occupied_x], dtype=np.float64)
    column_height = top - bottom + 1.0
    normalizer = max(float(height), 1.0)
    top = top / normalizer
    column_height = column_height / normalizer

    split = max(1, occupied_x.size // 2)
    left_top, right_top = top[:split], top[split:]
    left_height, right_height = column_height[:split], column_height[split:]
    if right_top.size == 0:
        right_top, right_height = left_top, left_height

    top_difference = float(left_top.mean() - right_top.mean())
    height_difference = float(left_height.mean() - right_height.mean())

    left_notch = _valley_depth(left_top)
    right_notch = _valley_depth(right_top)
    if top.size >= 3:
        smoothed = ndimage.gaussian_filter1d(top, sigma=0.75, mode="nearest")
        top_roughness = float(np.abs(np.diff(smoothed, n=2)).mean())
    else:
        top_roughness = 0.0

    column_excess = sum(max(0, _count_runs(crop[x]) - 1) for x in range(width))
    row_excess = sum(max(0, _count_runs(crop[:, z]) - 1) for z in range(height))
    filled = ndimage.binary_fill_holes(crop)

    centred = coordinates.astype(np.float64) - coordinates.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(coordinates.shape[0], 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eccentricity = float(
        np.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[-1], EPS)))
    )

    hull_area = _convex_hull_area(crop)
    values = {
        "area": area,
        "width": float(width),
        "height": float(height),
        "aspect_height_over_width": float(height / max(width, 1)),
        "perimeter_over_sqrt_area": _perimeter(crop) / np.sqrt(max(area, 1.0)),
        "solidity": float(area / max(hull_area, area)),
        "eccentricity": eccentricity,
        "top_asymmetry_abs": abs(top_difference),
        "top_left_minus_right": top_difference,
        "height_asymmetry_abs": abs(height_difference),
        "height_left_minus_right": height_difference,
        "top_notch_left": left_notch,
        "top_notch_right": right_notch,
        "top_notch_max": max(left_notch, right_notch),
        "top_roughness": top_roughness,
        "column_multirun_fraction": float(column_excess / max(width, 1)),
        "row_multirun_fraction": float(row_excess / max(height, 1)),
        "hole_fraction": float((filled.sum() - area) / max(filled.sum(), 1)),
    }
    if set(values) != set(BASE_FEATURE_NAMES):
        raise AssertionError("slice descriptor schema drift")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("slice descriptors must be finite")
    return values


def best_fit_first_anterior_slice(labels: np.ndarray) -> tuple[int, int, int]:
    """Return the unique best RAS-y cut, its cost, and the second-best gap.

    A candidate ``c`` assigns occupied voxels with y < c to posterior class 2
    and voxels with y >= c to anterior class 1.  The target is therefore the
    posterior-most/first stored anterior slice, i.e. the last head slice when
    traversing the hippocampus from anterior toward posterior.
    """

    labels = np.asarray(labels)
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("labels must be a 3-D array containing only 0, 1, and 2")
    support_by_y = (labels != 0).sum(axis=(0, 2))
    occupied = np.flatnonzero(support_by_y)
    if occupied.size < 2:
        raise ValueError("at least two occupied coronal slices are required")
    candidates = np.arange(int(occupied[0]) + 1, int(occupied[-1]) + 1)
    costs = []
    for cut in candidates:
        posterior_wrong = np.count_nonzero(labels[:, :cut, :] == 1)
        anterior_wrong = np.count_nonzero(labels[:, cut:, :] == 2)
        costs.append(int(posterior_wrong + anterior_wrong))
    order = np.argsort(np.asarray(costs), kind="stable")
    best_index = int(order[0])
    gap = int(costs[int(order[1])] - costs[best_index]) if len(order) > 1 else 0
    return int(candidates[best_index]), int(costs[best_index]), gap


def extract_case_features(case_name: str, labels: np.ndarray) -> CaseFeatures:
    """Extract per-slice union descriptors and the held-out scoring target."""

    labels = np.asarray(labels)
    support = labels != 0
    support_by_y = support.sum(axis=(0, 2))
    occupied = np.flatnonzero(support_by_y)
    if occupied.size < 2:
        raise ValueError(f"{case_name}: insufficient foreground extent")
    low, high = int(occupied[0]), int(occupied[-1])
    target, target_cost, second_best_gap = best_fit_first_anterior_slice(labels)
    features = {
        y: extract_slice_features(support[:, y, :])
        for y in range(low, high + 1)
    }
    return CaseFeatures(
        case_name=case_name,
        low=low,
        high=high,
        target_cut=target,
        target_cost=target_cost,
        second_best_gap=second_best_gap,
        features=features,
    )


def within_case_zscores(case: CaseFeatures) -> dict[int, dict[str, float]]:
    """Z-score each descriptor over occupied slices without reading A/P labels."""

    slices = list(range(case.low, case.high + 1))
    matrix = np.asarray(
        [[case.features[y][name] for name in BASE_FEATURE_NAMES] for y in slices],
        dtype=np.float64,
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < EPS] = 1.0
    normalized = (matrix - mean) / scale
    return {
        y: {name: float(normalized[index, column]) for column, name in enumerate(BASE_FEATURE_NAMES)}
        for index, y in enumerate(slices)
    }


def handcrafted_fold_scores(case: CaseFeatures) -> dict[int, float]:
    """Fixed, side-invariant foldedness score used before any model fitting."""

    normalized = within_case_zscores(case)
    return {
        y: float(np.mean([normalized[y][name] for name in FOLD_COMPONENTS]))
        for y in range(case.low, case.high + 1)
    }
