"""Spacing-aware hard-segmentation metrics used by counterfactual audits."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


_STRUCTURE = ndimage.generate_binary_structure(3, 1)


def binary_surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be three-dimensional.")
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~ndimage.binary_erosion(mask, structure=_STRUCTURE, border_value=0)


def binary_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    denominator = int(reference.sum() + prediction.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(reference, prediction).sum()) / denominator


def _directed_surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    source_surface = binary_surface(source)
    target_surface = binary_surface(target)
    if not source_surface.any() or not target_surface.any():
        return np.asarray([], dtype=np.float64)
    distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance[source_surface].astype(np.float64)


def surface_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    tolerances_mm: tuple[float, ...] = (1.0, 2.0),
) -> dict[str, float]:
    """Return symmetric surface Dice, ASSD, and HD95 in physical units."""

    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    spacing = tuple(float(v) for v in spacing)
    result: dict[str, float] = {}
    if not reference.any() and not prediction.any():
        result.update({"assd_mm": 0.0, "hd95_mm": 0.0})
        result.update({f"surface_dice_{t:g}mm": 1.0 for t in tolerances_mm})
        return result
    if not reference.any() or not prediction.any():
        result.update({"assd_mm": math.inf, "hd95_mm": math.inf})
        result.update({f"surface_dice_{t:g}mm": 0.0 for t in tolerances_mm})
        return result
    forward = _directed_surface_distances(reference, prediction, spacing)
    backward = _directed_surface_distances(prediction, reference, spacing)
    both = np.concatenate((forward, backward))
    result["assd_mm"] = float(both.mean())
    result["hd95_mm"] = float(np.quantile(both, 0.95))
    for tolerance in tolerances_mm:
        numerator = int(np.count_nonzero(forward <= tolerance)) + int(
            np.count_nonzero(backward <= tolerance)
        )
        result[f"surface_dice_{tolerance:g}mm"] = numerator / (len(forward) + len(backward))
    return result


def segmentation_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float | int]:
    """Return union surface/overlap metrics plus directed three-class errors."""

    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if labels.shape != prediction.shape or labels.ndim != 3:
        raise ValueError("labels and prediction must be matching 3-D arrays.")
    reference_union = labels > 0
    predicted_union = prediction > 0
    foreground_fp = int(np.count_nonzero(~reference_union & predicted_union))
    foreground_fn = int(np.count_nonzero(reference_union & ~predicted_union))
    swaps = int(
        np.count_nonzero(
            ((labels == 1) & (prediction == 2)) | ((labels == 2) & (prediction == 1))
        )
    )
    components = int(ndimage.label(predicted_union, structure=_STRUCTURE)[1])
    result: dict[str, float | int] = {
        "union_dice": binary_dice(reference_union, predicted_union),
        "anterior_dice": binary_dice(labels == 1, prediction == 1),
        "posterior_dice": binary_dice(labels == 2, prediction == 2),
        "foreground_fp": foreground_fp,
        "foreground_fn": foreground_fn,
        "ap_swaps": swaps,
        "predicted_foreground_voxels": int(predicted_union.sum()),
        "reference_foreground_voxels": int(reference_union.sum()),
        "predicted_components": components,
    }
    result.update(surface_metrics(reference_union, predicted_union, spacing))
    return result
