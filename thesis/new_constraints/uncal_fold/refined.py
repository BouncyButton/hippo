"""Mirror-invariant, anatomically interpretable uncal-fold descriptors.

The Task04 collection contains left- and right-hippocampus crops without a
reliable laterality field in the released NIfTI metadata.  These descriptors
therefore avoid assuming that low-x or high-x is medial.  They measure the
stronger of the two possible superior-side protrusions, the central notch
between superior lobes, and spatially localized change from the immediately
posterior coronal slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .foldedness import EPS


REFINED_SLICE_FEATURE_NAMES = (
    "area",
    "width",
    "height",
    "height_over_width",
    "solidity_proxy",
    "superior_notch_depth",
    "superior_notch_centrality",
    "superior_bilobedness",
    "superior_peak_separation",
    "superior_peak_balance",
    "superior_protrusion_max",
    "superior_protrusion_difference",
    "superior_extension_area_max",
    "superior_quadrant_asymmetry",
    "multirun_side_max",
    "multirun_side_difference",
    "hole_fraction",
    "hole_count",
)

TRANSITION_FEATURE_NAMES = (
    "relative_area_change",
    "relative_width_change",
    "relative_height_change",
    "slice_dice",
    "added_fraction",
    "removed_fraction",
    "novel_fraction_after_dilation",
    "novel_superior_fraction",
    "novel_superior_side_max",
    "novel_superior_side_difference",
    "centroid_shift_x",
    "centroid_shift_z",
    "superior_boundary_rise_max",
    "superior_boundary_rise_mean",
)

IMAGE_FEATURE_NAMES = (
    "mask_intensity_mean",
    "mask_intensity_std",
    "mask_intensity_range_80",
    "mask_gradient_mean",
    "mask_gradient_p90",
    "superior_inferior_intensity_difference_abs",
    "left_right_intensity_difference_abs",
    "superior_band_intensity_mean",
    "superior_band_intensity_std",
    "superior_band_left_right_difference_abs",
    "central_superior_band_intensity_mean",
)


@dataclass(frozen=True)
class RefinedCaseFeatures:
    slice_shape: dict[int, dict[str, float]]
    transition_shape: dict[int, dict[str, float]]
    slice_image: dict[int, dict[str, float]]


def _count_runs(values: np.ndarray) -> int:
    padded = np.pad(np.asarray(values, dtype=np.int8), 1)
    return int(np.count_nonzero(np.diff(padded) == 1))


def _top_profile(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupied_x = np.flatnonzero(mask.any(axis=1))
    top = np.asarray(
        [np.flatnonzero(mask[x]).max() for x in occupied_x], dtype=np.float64
    )
    return occupied_x, top


def _superior_profile_features(mask: np.ndarray) -> dict[str, float]:
    occupied_x, raw_top = _top_profile(mask)
    height = max(mask.shape[1], 1)
    top = raw_top / height
    smooth = ndimage.gaussian_filter1d(top, sigma=0.65, mode="nearest")
    count = smooth.size
    if count < 5:
        return {
            "superior_notch_depth": 0.0,
            "superior_notch_centrality": 0.0,
            "superior_bilobedness": 0.0,
            "superior_peak_separation": 0.0,
            "superior_peak_balance": 0.0,
        }

    best = (0.0, count // 2, 0, count - 1, 0.0)
    for valley in range(2, count - 2):
        left_peak_index = int(np.argmax(smooth[:valley]))
        right_peak_index = valley + int(np.argmax(smooth[valley + 1 :])) + 1
        left_peak = float(smooth[left_peak_index])
        right_peak = float(smooth[right_peak_index])
        depth = max(0.0, min(left_peak, right_peak) - float(smooth[valley]))
        centrality = 1.0 - abs(2.0 * valley / (count - 1) - 1.0)
        score = depth * centrality
        if score > best[0] * best[4]:
            best = (depth, valley, left_peak_index, right_peak_index, centrality)

    depth, _, left_peak_index, right_peak_index, centrality = best
    left_peak = float(smooth[left_peak_index])
    right_peak = float(smooth[right_peak_index])
    vertical_range = max(float(smooth.max() - smooth.min()), 1.0 / height)
    peak_balance = 1.0 - min(1.0, abs(left_peak - right_peak) / vertical_range)
    separation = (right_peak_index - left_peak_index) / max(count - 1, 1)
    return {
        "superior_notch_depth": float(depth),
        "superior_notch_centrality": float(centrality),
        "superior_bilobedness": float(depth * centrality * peak_balance),
        "superior_peak_separation": float(separation),
        "superior_peak_balance": float(peak_balance),
    }


def _side_features(mask: np.ndarray) -> dict[str, float]:
    width, height = mask.shape
    midpoint = max(1, width // 2)
    left, right = mask[:midpoint], mask[midpoint:]
    if right.shape[0] == 0:
        right = left

    def side_top(side: np.ndarray) -> np.ndarray:
        return np.asarray(
            [np.flatnonzero(side[x]).max() for x in range(side.shape[0]) if side[x].any()],
            dtype=np.float64,
        )

    left_top, right_top = side_top(left), side_top(right)
    if left_top.size == 0:
        left_top = right_top
    if right_top.size == 0:
        right_top = left_top
    left_peak, right_peak = float(left_top.max()), float(right_top.max())
    left_extension = max(0.0, left_peak - float(np.quantile(right_top, 0.75))) / max(height, 1)
    right_extension = max(0.0, right_peak - float(np.quantile(left_top, 0.75))) / max(height, 1)

    left_threshold, right_threshold = float(np.quantile(right_top, 0.75)), float(np.quantile(left_top, 0.75))
    z = np.arange(height)[None, :]
    left_extension_area = float(np.count_nonzero(left & (z > left_threshold)))
    right_extension_area = float(np.count_nonzero(right & (z > right_threshold)))
    area = max(float(mask.sum()), 1.0)

    superior_start = height // 2
    left_superior = float(left[:, superior_start:].sum()) / max(float(left.sum()), 1.0)
    right_superior = float(right[:, superior_start:].sum()) / max(float(right.sum()), 1.0)
    left_multirun = np.mean([max(0, _count_runs(left[x]) - 1) for x in range(left.shape[0])])
    right_multirun = np.mean([max(0, _count_runs(right[x]) - 1) for x in range(right.shape[0])])
    return {
        "superior_protrusion_max": max(left_extension, right_extension),
        "superior_protrusion_difference": abs(left_extension - right_extension),
        "superior_extension_area_max": max(left_extension_area, right_extension_area) / area,
        "superior_quadrant_asymmetry": abs(left_superior - right_superior),
        "multirun_side_max": float(max(left_multirun, right_multirun)),
        "multirun_side_difference": float(abs(left_multirun - right_multirun)),
    }


def extract_refined_slice_features(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a non-empty 2-D array")
    coordinates = np.argwhere(mask)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    crop = mask[low[0] : high[0] + 1, low[1] : high[1] + 1]
    width, height = crop.shape
    area = float(crop.sum())
    filled = ndimage.binary_fill_holes(crop)
    _, hole_count = ndimage.label(filled & ~crop)

    profile_forward = _superior_profile_features(crop)
    profile_reverse = _superior_profile_features(crop[::-1])
    profile = {
        name: 0.5 * (profile_forward[name] + profile_reverse[name])
        for name in profile_forward
    }
    side_forward = _side_features(crop)
    side_reverse = _side_features(crop[::-1])
    side = {
        name: 0.5 * (side_forward[name] + side_reverse[name])
        for name in side_forward
    }

    # A cheap raster solidity proxy is sufficient here and remains readily
    # differentiable later via pooling/morphological approximations.
    closed = ndimage.binary_closing(crop, structure=np.ones((3, 3), dtype=bool))
    values = {
        "area": area,
        "width": float(width),
        "height": float(height),
        "height_over_width": float(height / max(width, 1)),
        "solidity_proxy": float(area / max(float(closed.sum()), area)),
        **profile,
        **side,
        "hole_fraction": float((filled.sum() - area) / max(float(filled.sum()), 1.0)),
        "hole_count": float(hole_count),
    }
    if set(values) != set(REFINED_SLICE_FEATURE_NAMES):
        raise AssertionError("refined slice descriptor schema drift")
    return values


def _centroid(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).mean(axis=0)


def extract_transition_features(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    previous = np.asarray(previous, dtype=bool)
    current = np.asarray(current, dtype=bool)
    if not previous.any() or not current.any() or previous.shape != current.shape:
        raise ValueError("transition masks must be non-empty and have matching shape")
    previous_area, current_area = float(previous.sum()), float(current.sum())
    intersection = float(np.count_nonzero(previous & current))
    added = current & ~previous
    removed = previous & ~current
    novel = current & ~ndimage.binary_dilation(previous, iterations=1)

    union_coordinates = np.argwhere(previous | current)
    low = union_coordinates.min(axis=0)
    high = union_coordinates.max(axis=0)
    x_mid = int((low[0] + high[0] + 1) // 2)
    z_mid = int((low[1] + high[1] + 1) // 2)
    superior = np.zeros_like(current)
    superior[:, z_mid : high[1] + 1] = True
    left = np.zeros_like(current)
    left[low[0] : x_mid, :] = True
    right = np.zeros_like(current)
    right[x_mid : high[0] + 1, :] = True
    novel_superior = novel & superior
    novel_left = float(np.count_nonzero(novel_superior & left)) / max(current_area, 1.0)
    novel_right = float(np.count_nonzero(novel_superior & right)) / max(current_area, 1.0)

    previous_bbox = np.ptp(np.argwhere(previous), axis=0) + 1
    current_bbox = np.ptp(np.argwhere(current), axis=0) + 1
    previous_top = np.full(current.shape[0], np.nan)
    current_top = np.full(current.shape[0], np.nan)
    for x in range(current.shape[0]):
        if previous[x].any():
            previous_top[x] = np.flatnonzero(previous[x]).max()
        if current[x].any():
            current_top[x] = np.flatnonzero(current[x]).max()
    shared = np.isfinite(previous_top) & np.isfinite(current_top)
    rise = current_top[shared] - previous_top[shared]
    positive_rise = np.maximum(rise, 0.0) if rise.size else np.zeros(1)

    values = {
        "relative_area_change": float((current_area - previous_area) / max(previous_area, 1.0)),
        "relative_width_change": float((current_bbox[0] - previous_bbox[0]) / max(previous_bbox[0], 1)),
        "relative_height_change": float((current_bbox[1] - previous_bbox[1]) / max(previous_bbox[1], 1)),
        "slice_dice": float(2.0 * intersection / max(previous_area + current_area, 1.0)),
        "added_fraction": float(added.sum() / max(current_area, 1.0)),
        "removed_fraction": float(removed.sum() / max(previous_area, 1.0)),
        "novel_fraction_after_dilation": float(novel.sum() / max(current_area, 1.0)),
        "novel_superior_fraction": float(novel_superior.sum() / max(current_area, 1.0)),
        "novel_superior_side_max": max(novel_left, novel_right),
        "novel_superior_side_difference": abs(novel_left - novel_right),
        "centroid_shift_x": float(abs(_centroid(current)[0] - _centroid(previous)[0])),
        "centroid_shift_z": float(abs(_centroid(current)[1] - _centroid(previous)[1])),
        "superior_boundary_rise_max": float(positive_rise.max(initial=0.0)),
        "superior_boundary_rise_mean": float(positive_rise.mean()),
    }
    if set(values) != set(TRANSITION_FEATURE_NAMES):
        raise AssertionError("transition descriptor schema drift")
    return values


def normalise_volume_intensity(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    finite = image[np.isfinite(image)]
    low, high = np.quantile(finite, (0.01, 0.99))
    return np.clip((image - low) / max(high - low, EPS), 0.0, 1.0)


def extract_image_features(image_slice: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    image_slice = np.asarray(image_slice, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if image_slice.shape != mask.shape or not mask.any():
        raise ValueError("image and non-empty mask must have identical 2-D shapes")
    coordinates = np.argwhere(mask)
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    x_mid = int((low[0] + high[0] + 1) // 2)
    z_mid = int((low[1] + high[1] + 1) // 2)
    values_in_mask = image_slice[mask]
    gx, gz = np.gradient(image_slice)
    gradient = np.hypot(gx, gz)[mask]

    superior_mask = mask.copy()
    superior_mask[:, :z_mid] = False
    inferior_mask = mask & ~superior_mask
    left_mask = mask.copy()
    left_mask[x_mid:, :] = False
    right_mask = mask & ~left_mask

    dilated = ndimage.binary_dilation(mask, iterations=2)
    superior_band = (dilated & ~mask)
    superior_band[:, :z_mid] = False
    left_band = superior_band.copy()
    left_band[x_mid:, :] = False
    right_band = superior_band & ~left_band
    centre_width = max(1, (high[0] - low[0] + 1) // 4)
    centre_low = max(int(low[0]), x_mid - centre_width)
    centre_high = min(int(high[0]) + 1, x_mid + centre_width + 1)
    central_band = np.zeros_like(mask)
    central_band[centre_low:centre_high, z_mid : int(high[1]) + 3] = True
    central_band &= dilated & ~mask

    def mean(region: np.ndarray, fallback: float) -> float:
        return float(image_slice[region].mean()) if region.any() else fallback

    global_mean = float(values_in_mask.mean())
    superior_mean = mean(superior_mask, global_mean)
    inferior_mean = mean(inferior_mask, global_mean)
    left_mean = mean(left_mask, global_mean)
    right_mean = mean(right_mask, global_mean)
    band_values = image_slice[superior_band] if superior_band.any() else values_in_mask
    result = {
        "mask_intensity_mean": global_mean,
        "mask_intensity_std": float(values_in_mask.std()),
        "mask_intensity_range_80": float(np.quantile(values_in_mask, 0.9) - np.quantile(values_in_mask, 0.1)),
        "mask_gradient_mean": float(gradient.mean()),
        "mask_gradient_p90": float(np.quantile(gradient, 0.9)),
        "superior_inferior_intensity_difference_abs": abs(superior_mean - inferior_mean),
        "left_right_intensity_difference_abs": abs(left_mean - right_mean),
        "superior_band_intensity_mean": float(band_values.mean()),
        "superior_band_intensity_std": float(band_values.std()),
        "superior_band_left_right_difference_abs": abs(
            mean(left_band, float(band_values.mean())) - mean(right_band, float(band_values.mean()))
        ),
        "central_superior_band_intensity_mean": mean(central_band, float(band_values.mean())),
    }
    if set(result) != set(IMAGE_FEATURE_NAMES):
        raise AssertionError("image descriptor schema drift")
    return result


def extract_refined_case_features(
    image: np.ndarray,
    union: np.ndarray,
    low: int,
    high: int,
) -> RefinedCaseFeatures:
    image = normalise_volume_intensity(image)
    slice_shape = {}
    transition_shape = {}
    slice_image = {}
    for y in range(low, high + 1):
        mask = union[:, y, :]
        slice_shape[y] = extract_refined_slice_features(mask)
        slice_image[y] = extract_image_features(image[:, y, :], mask)
        if y > low:
            transition_shape[y] = extract_transition_features(union[:, y - 1, :], mask)
    return RefinedCaseFeatures(slice_shape, transition_shape, slice_image)
