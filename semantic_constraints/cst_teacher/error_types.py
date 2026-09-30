"""Auditable voxel-error taxonomy for CST-guided slice-quality studies.

These are operational labels from reference masks, not claims about clinical
anatomy. Reference masks are used only to build training/evaluation targets.
"""

from __future__ import annotations

import numpy as np


TYPE_NAMES = ("minimal", "missing", "extra", "ap_swap", "mixed")
MIN_ERROR_VOXELS = 5
DOMINANCE_FRACTION = 0.60


def count_slice_errors(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return [missing, extra, A/P swap] counts for one coronal 2-D slice."""
    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    if reference.shape != predicted.shape or reference.ndim != 2:
        raise ValueError("reference and prediction must be matching 2-D slices")
    if np.any((reference < 0) | (reference > 2)) or np.any((predicted < 0) | (predicted > 2)):
        raise ValueError("hippocampus labels must be in {0,1,2}")
    missing = np.count_nonzero((reference > 0) & (predicted == 0))
    extra = np.count_nonzero((reference == 0) & (predicted > 0))
    swap = np.count_nonzero((reference > 0) & (predicted > 0) & (reference != predicted))
    return np.asarray((missing, extra, swap), dtype=np.int32)


def classify_counts(
    counts: np.ndarray,
    *,
    minimum_error_voxels: int = MIN_ERROR_VOXELS,
    dominance_fraction: float = DOMINANCE_FRACTION,
) -> np.ndarray:
    """Classify count rows as minimal/missing/extra/A-P swap/mixed.

    A class is dominant only if it accounts for at least 60% of wrong voxels.
    Mixed is kept explicit instead of forcing an unsafe correction direction.
    """
    counts = np.asarray(counts)
    if counts.ndim < 1 or counts.shape[-1] != 3 or np.any(counts < 0):
        raise ValueError("counts must have a final [missing,extra,swap] axis")
    if minimum_error_voxels < 1 or not 1 / 3 < dominance_fraction <= 1:
        raise ValueError("invalid taxonomy thresholds")
    total = counts.sum(axis=-1)
    dominant = np.argmax(counts, axis=-1) + 1
    fraction = counts.max(axis=-1) / np.maximum(total, 1)
    result = np.where(total < minimum_error_voxels, 0, np.where(fraction >= dominance_fraction, dominant, 4))
    return result.astype(np.int8)


def audit_case(reference: np.ndarray, prediction: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    """Classify sampled coronal slices and flag isolated/terminal errors."""
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    positions = np.asarray(positions, dtype=np.int64)
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError("reference and prediction must be matching (X,Y,Z) arrays")
    if positions.ndim != 1 or len(positions) < 2 or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be strictly increasing")
    if positions[0] < 0 or positions[-1] >= reference.shape[1]:
        raise ValueError("position outside coronal axis")
    counts = np.stack(
        [count_slice_errors(reference[:, index, :], prediction[:, index, :]) for index in positions]
    )
    ref_area = np.asarray([np.count_nonzero(reference[:, index, :]) for index in positions], dtype=np.int32)
    pred_area = np.asarray([np.count_nonzero(prediction[:, index, :]) for index in positions], dtype=np.int32)
    ref_active = ref_area >= MIN_ERROR_VOXELS
    pred_active = pred_area >= MIN_ERROR_VOXELS
    interior = np.zeros(len(positions), dtype=bool)
    interior[1:-1] = pred_active[:-2] & pred_active[2:]
    isolated_disappearance = ref_active & ~pred_active & interior
    adjacent_ref = np.zeros(len(positions), dtype=bool)
    adjacent_ref[1:] |= ref_active[:-1]
    adjacent_ref[:-1] |= ref_active[1:]
    terminal_extension = ~ref_active & pred_active & adjacent_ref
    return {
        "error_counts": counts,
        "error_type": classify_counts(counts),
        "reference_area": ref_area,
        "prediction_area": pred_area,
        "isolated_disappearance": isolated_disappearance,
        "terminal_extension": terminal_extension,
    }
