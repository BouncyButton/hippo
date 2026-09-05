"""Ray-level observability analysis for ordinal boundary constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import SurfaceSamples
from .objective import onecut_log_truth_from_values, sample_volume, semantic_field


CATEGORY_NAMES = ("correct", "shifted", "missing", "reversed", "multiple")
CATEGORY_TO_CODE = {name: index for index, name in enumerate(CATEGORY_NAMES)}


@dataclass(frozen=True)
class RayAnalysis:
    offsets_mm: np.ndarray
    values: np.ndarray
    pair_margin: np.ndarray
    violation: np.ndarray
    category_code: np.ndarray
    nearest_crossing_mm: np.ndarray
    crossing_count: np.ndarray
    onecut_loss: np.ndarray
    onecut_truth: np.ndarray
    onecut_allowed_mass: np.ndarray
    onecut_best_cut_mm: np.ndarray

    @property
    def error(self) -> np.ndarray:
        return self.category_code != CATEGORY_TO_CODE["correct"]

    def summary(self) -> dict[str, float | int | dict[str, int]]:
        error = self.error
        violation = self.violation
        error_count = int(error.sum())
        violation_count = int(violation.sum())
        joint = int(np.count_nonzero(error & violation))
        prevalence = float(error.mean()) if len(error) else float("nan")
        precision = joint / violation_count if violation_count else float("nan")
        coverage = joint / error_count if error_count else float("nan")
        lift = precision / prevalence if prevalence > 0 and np.isfinite(precision) else float("nan")
        return {
            "ray_count": int(len(error)),
            "error_count": error_count,
            "error_prevalence": prevalence,
            "violation_count": violation_count,
            "violation_rate": float(violation.mean()) if len(violation) else float("nan"),
            "error_coverage": coverage,
            "error_precision": precision,
            "precision_lift": lift,
            "mean_pair_margin": float(np.mean(self.pair_margin)),
            "median_pair_margin": float(np.median(self.pair_margin)),
            "mean_onecut_loss": float(np.mean(self.onecut_loss)),
            "median_onecut_loss": float(np.median(self.onecut_loss)),
            "mean_onecut_truth": float(np.mean(self.onecut_truth)),
            "mean_onecut_allowed_mass": float(np.mean(self.onecut_allowed_mass)),
            "categories": {
                name: int(np.count_nonzero(self.category_code == code))
                for name, code in CATEGORY_TO_CODE.items()
            },
        }


def _crossings(offsets: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return interpolated zero crossings and their decreasing orientation."""

    locations: list[float] = []
    decreasing: list[bool] = []
    for index in range(len(offsets) - 1):
        left, right = float(values[index]), float(values[index + 1])
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            location = float(offsets[index])
        elif right == 0.0:
            location = float(offsets[index + 1])
        elif left * right > 0.0:
            continue
        else:
            fraction = -left / (right - left)
            location = float(offsets[index] + fraction * (offsets[index + 1] - offsets[index]))
        if locations and abs(location - locations[-1]) < 1e-8:
            continue
        locations.append(location)
        decreasing.append(right < left)
    return np.asarray(locations), np.asarray(decreasing, dtype=bool)


def analyse_rays(
    logits: torch.Tensor,
    samples: SurfaceSamples,
    *,
    offsets_mm: np.ndarray,
    delta_mm: float,
    margin: float = 0.0,
    temperature: float = 1.0,
    localization_tolerance_mm: float = 1.0,
) -> RayAnalysis:
    """Classify crossings and measure which erroneous rays violate the rule."""

    offsets = np.asarray(offsets_mm, dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) < 3 or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets_mm must be a strictly increasing vector of length >= 3.")
    if offsets[0] > -delta_mm or offsets[-1] < delta_mm:
        raise ValueError("offsets_mm must cover both pair locations.")
    device = logits.device
    coordinates = torch.as_tensor(
        samples.coordinates_at(offsets), device=device, dtype=torch.float32
    )
    field = semantic_field(logits, samples.interface.value)
    with torch.no_grad():
        values = sample_volume(field, coordinates).cpu().numpy().astype(np.float64)
        pair_coordinates = torch.as_tensor(
            samples.coordinates_at([-delta_mm, delta_mm]),
            device=device,
            dtype=torch.float32,
        )
        pair = sample_volume(field, pair_coordinates).cpu().numpy().astype(np.float64)
    pair_margin = pair[:, 0] - pair[:, 1]
    violation = pair_margin < margin
    with torch.no_grad():
        log_truth, onecut_truth_t, allowed_mass_t, best_cut_t = onecut_log_truth_from_values(
            torch.as_tensor(values, dtype=torch.float32, device=device),
            torch.as_tensor(offsets, dtype=torch.float32, device=device),
            tolerance_mm=localization_tolerance_mm,
            margin=margin,
            temperature=temperature,
        )
        onecut_loss = (-log_truth / values.shape[1]).cpu().numpy().astype(np.float64)
        onecut_truth = onecut_truth_t.cpu().numpy().astype(np.float64)
        onecut_allowed_mass = allowed_mass_t.cpu().numpy().astype(np.float64)
        onecut_best_cut = best_cut_t.cpu().numpy().astype(np.float64)
    category = np.empty(len(values), dtype=np.int8)
    nearest = np.full(len(values), np.nan, dtype=np.float64)
    count = np.zeros(len(values), dtype=np.int16)
    for index, ray in enumerate(values):
        locations, decreasing = _crossings(offsets, ray)
        count[index] = len(locations)
        if len(locations) == 0:
            category[index] = CATEGORY_TO_CODE["missing"]
            continue
        chosen = int(np.argmin(np.abs(locations)))
        nearest[index] = locations[chosen]
        if len(locations) > 1:
            category[index] = CATEGORY_TO_CODE["multiple"]
        elif not bool(decreasing[chosen]):
            category[index] = CATEGORY_TO_CODE["reversed"]
        elif abs(locations[chosen]) > localization_tolerance_mm:
            category[index] = CATEGORY_TO_CODE["shifted"]
        else:
            category[index] = CATEGORY_TO_CODE["correct"]
    return RayAnalysis(
        offsets,
        values,
        pair_margin,
        violation,
        category,
        nearest,
        count,
        onecut_loss,
        onecut_truth,
        onecut_allowed_mass,
        onecut_best_cut,
    )
