"""Metrics and patient-level statistics for translation-equivalence closure analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .translation_equivariance import Shift3D, _classwise_soft_dice


FOREGROUND_CLASS_IDS = (1, 2)
CLASS_NAMES = {1: "anterior", 2: "posterior"}
AXIS_NAMES = ("x", "y", "z")


def signed_axis_shifts(magnitudes: Sequence[int]) -> tuple[Shift3D, ...]:
    """Return both signs of every axis-aligned shift in a stable order."""

    normalized = tuple(int(value) for value in magnitudes)
    if not normalized or any(value < 1 for value in normalized):
        raise ValueError("Shift magnitudes must contain positive integers.")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Shift magnitudes must be unique.")
    shifts: list[Shift3D] = []
    for magnitude in normalized:
        for axis in range(3):
            for sign in (-1, 1):
                values = [0, 0, 0]
                values[axis] = sign * magnitude
                shifts.append(tuple(values))
    return tuple(shifts)


def describe_shift(shift: Shift3D) -> tuple[str, int, int]:
    """Return ``(axis, sign, magnitude)`` for an axis-aligned nonzero shift."""

    nonzero = [index for index, value in enumerate(shift) if value != 0]
    if len(nonzero) != 1:
        raise ValueError(f"Expected one nonzero shift axis, found {shift}.")
    axis_index = nonzero[0]
    amount = int(shift[axis_index])
    return AXIS_NAMES[axis_index], 1 if amount > 0 else -1, abs(amount)


def _label_map(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4:
        raise ValueError(f"Expected labels shaped (B,[1,]X,Y,Z), found {labels.shape}.")
    return labels.long()


def _mask_for(probabilities: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(
            (probabilities.shape[0], 1, *probabilities.shape[-3:]),
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
    if valid_mask.ndim == 4:
        valid_mask = valid_mask.unsqueeze(1)
    expected_spatial = probabilities.shape[-3:]
    if valid_mask.ndim != 5 or valid_mask.shape[-3:] != expected_spatial:
        raise ValueError(
            f"Validity mask must end in {expected_spatial}, found {valid_mask.shape}."
        )
    if valid_mask.shape[0] not in (1, probabilities.shape[0]):
        raise ValueError("Validity-mask batch dimension is not broadcastable.")
    if valid_mask.shape[1] != 1:
        raise ValueError("Validity mask must have one channel.")
    return valid_mask.to(device=probabilities.device, dtype=probabilities.dtype)


def classwise_segmentation_dice(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
    hard: bool,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute class Dice on one shared mask, ignoring absent target classes."""

    if probabilities.ndim != 5:
        raise ValueError("Probabilities must have shape (B,C,X,Y,Z).")
    if not class_ids or max(class_ids) >= probabilities.shape[1]:
        raise ValueError("class_ids are incompatible with the probability channels.")
    target_labels = _label_map(labels)
    mask = _mask_for(probabilities, valid_mask)
    if hard:
        predicted_labels = probabilities.argmax(dim=1)
        selected_predictions = torch.stack(
            [(predicted_labels == class_id) for class_id in class_ids], dim=1
        ).to(probabilities.dtype)
    else:
        selected_predictions = probabilities[:, class_ids]
    selected_targets = torch.stack(
        [(target_labels == class_id) for class_id in class_ids], dim=1
    ).to(probabilities.dtype)
    selected_predictions = selected_predictions * mask
    selected_targets = selected_targets * mask
    spatial_dimensions = (2, 3, 4)
    intersection = (selected_predictions * selected_targets).sum(spatial_dimensions)
    prediction_volume = selected_predictions.sum(spatial_dimensions)
    target_volume = selected_targets.sum(spatial_dimensions)
    dice = (2.0 * intersection + epsilon) / (
        prediction_volume + target_volume + epsilon
    )
    return torch.where(target_volume > 0, dice, torch.full_like(dice, torch.nan))


def nanmean_classes(values: torch.Tensor) -> torch.Tensor:
    """Average class scores while preserving all-empty samples as NaN."""

    finite = torch.isfinite(values)
    count = finite.sum(dim=1)
    total = torch.where(finite, values, torch.zeros_like(values)).sum(dim=1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, torch.nan))


def equivariance_scores(
    base_probabilities: torch.Tensor,
    aligned_probabilities: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return the canonical pure score and secondary linear-denominator score."""

    pure = _classwise_soft_dice(
        base_probabilities,
        aligned_probabilities,
        valid_mask,
        class_ids,
        epsilon,
        squared_denominator=True,
    )
    confidence_weighted = _classwise_soft_dice(
        base_probabilities,
        aligned_probabilities,
        valid_mask,
        class_ids,
        epsilon,
        squared_denominator=False,
    )
    mask = _mask_for(base_probabilities, valid_mask)
    selected_base = base_probabilities[:, class_ids] * mask
    selected_aligned = aligned_probabilities[:, class_ids] * mask
    spatial_dimensions = (2, 3, 4)
    return {
        "pure_classwise": pure,
        "pure_mean": pure.mean(dim=1),
        "secondary_confidence_weighted_classwise": confidence_weighted,
        "secondary_confidence_weighted_mean": confidence_weighted.mean(dim=1),
        "pure_denominator_classwise": (
            selected_base.square().sum(spatial_dimensions)
            + selected_aligned.square().sum(spatial_dimensions)
        ),
        "secondary_denominator_classwise": (
            selected_base.sum(spatial_dimensions)
            + selected_aligned.sum(spatial_dimensions)
        ),
    }


@dataclass
class ValidityAwareViewAccumulator:
    """Streaming mean and variance for aligned probability views."""

    probability_sum: torch.Tensor
    squared_norm_sum: torch.Tensor
    count: torch.Tensor

    @classmethod
    def from_base(cls, base_probabilities: torch.Tensor) -> "ValidityAwareViewAccumulator":
        if base_probabilities.ndim != 5:
            raise ValueError("Base probabilities must have shape (B,C,X,Y,Z).")
        return cls(
            probability_sum=base_probabilities.clone(),
            squared_norm_sum=base_probabilities.square().sum(dim=1, keepdim=True),
            count=torch.ones(
                (base_probabilities.shape[0], 1, *base_probabilities.shape[-3:]),
                device=base_probabilities.device,
                dtype=base_probabilities.dtype,
            ),
        )

    def add(self, probabilities: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if probabilities.shape != self.probability_sum.shape:
            raise ValueError("Every probability view must match the base shape.")
        mask = _mask_for(probabilities, valid_mask)
        self.probability_sum.add_(probabilities * mask)
        self.squared_norm_sum.add_(
            probabilities.square().sum(dim=1, keepdim=True) * mask
        )
        self.count.add_(mask)

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.probability_sum / self.count
        disagreement = (
            self.squared_norm_sum / self.count
            - mean.square().sum(dim=1, keepdim=True)
        ).clamp_min(0.0)
        return mean, disagreement, self.count


def validity_aware_tta(
    base_probabilities: torch.Tensor,
    aligned_views: Sequence[torch.Tensor],
    valid_masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience wrapper used by tests and small in-memory callers."""

    if len(aligned_views) != len(valid_masks):
        raise ValueError("Each aligned view requires one validity mask.")
    accumulator = ValidityAwareViewAccumulator.from_base(base_probabilities)
    for probabilities, mask in zip(aligned_views, valid_masks, strict=True):
        accumulator.add(probabilities, mask)
    return accumulator.finalize()


def binary_dilation_3d(mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Dilate with a documented 3x3x3 (26-neighbour) structuring element."""

    if steps < 0:
        raise ValueError("Dilation steps must be non-negative.")
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("Binary masks must have shape (B,[1,]X,Y,Z).")
    result = mask.bool()
    for _ in range(steps):
        result = F.max_pool3d(result.float(), kernel_size=3, stride=1, padding=1).bool()
    return result


def binary_boundary_3d(mask: torch.Tensor) -> torch.Tensor:
    """Mark both sides of every 6-neighbour foreground/background transition."""

    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("Binary masks must have shape (B,[1,]X,Y,Z).")
    values = mask.bool()
    boundary = torch.zeros_like(values)
    for dimension in (2, 3, 4):
        first = [slice(None)] * 5
        second = [slice(None)] * 5
        first[dimension] = slice(1, None)
        second[dimension] = slice(None, -1)
        transitions = values[tuple(first)] != values[tuple(second)]
        boundary[tuple(first)] |= transitions
        boundary[tuple(second)] |= transitions
    return boundary


def foreground_relevant_roi(
    labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    dilation_steps: int,
) -> torch.Tensor:
    """Dilate the union of ground-truth and predicted foreground."""

    target = _label_map(labels).unsqueeze(1) > 0
    if predicted_labels.ndim == 4:
        predicted_labels = predicted_labels.unsqueeze(1)
    if predicted_labels.shape != target.shape:
        raise ValueError("Prediction and label maps must have the same shape.")
    return binary_dilation_3d(target | (predicted_labels > 0), dilation_steps)


def boundary_relevant_roi(
    labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    dilation_steps: int,
) -> torch.Tensor:
    """Dilate boundaries from both ground truth and the ordinary prediction."""

    target = _label_map(labels).unsqueeze(1) > 0
    if predicted_labels.ndim == 4:
        predicted_labels = predicted_labels.unsqueeze(1)
    predicted = predicted_labels > 0
    combined = binary_boundary_3d(target) | binary_boundary_3d(predicted)
    return binary_dilation_3d(combined, dilation_steps)


def uncertainty_error_metrics(
    disagreement: torch.Tensor,
    predicted_labels: torch.Tensor,
    labels: torch.Tensor,
    region: torch.Tensor,
) -> dict[str, float | int | None]:
    """Measure how well disagreement ranks voxel errors inside one ROI."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    target = _label_map(labels)
    if predicted_labels.ndim == 5 and predicted_labels.shape[1] == 1:
        predicted_labels = predicted_labels[:, 0]
    if predicted_labels.shape != target.shape:
        raise ValueError("Prediction and label maps must have the same shape.")
    if disagreement.ndim == 5 and disagreement.shape[1] == 1:
        disagreement = disagreement[:, 0]
    if region.ndim == 5 and region.shape[1] == 1:
        region = region[:, 0]
    selected = region.bool()
    scores = disagreement[selected].detach().float().cpu().numpy()
    errors = (predicted_labels != target)[selected].detach().cpu().numpy().astype(np.uint8)
    if scores.size == 0:
        raise ValueError("Uncertainty region contains no voxels.")
    correct_scores = scores[errors == 0]
    error_scores = scores[errors == 1]
    auroc: float | None = None
    auprc: float | None = None
    if np.unique(errors).size == 2:
        auroc = float(roc_auc_score(errors, scores))
        auprc = float(average_precision_score(errors, scores))
    return {
        "voxel_count": int(scores.size),
        "error_voxels": int(errors.sum()),
        "error_fraction": float(errors.mean()),
        "disagreement_mean": float(scores.mean()),
        "disagreement_q95": float(np.quantile(scores, 0.95)),
        "disagreement_correct_mean": (
            float(correct_scores.mean()) if correct_scores.size else None
        ),
        "disagreement_error_mean": (
            float(error_scores.mean()) if error_scores.size else None
        ),
        "voxel_auroc": auroc,
        "voxel_auprc": auprc,
    }


def label_free_disagreement_metrics(
    disagreement: torch.Tensor, region: torch.Tensor | None = None
) -> dict[str, float | int | None]:
    """Summarize disagreement when no segmentation labels are available."""

    if disagreement.ndim == 5 and disagreement.shape[1] == 1:
        disagreement = disagreement[:, 0]
    if region is None:
        selected = torch.ones_like(disagreement, dtype=torch.bool)
    else:
        if region.ndim == 5 and region.shape[1] == 1:
            region = region[:, 0]
        selected = region.bool()
    values = disagreement[selected].detach().float().cpu().numpy()
    if values.size == 0:
        raise ValueError("Disagreement region contains no voxels.")
    return {
        "voxel_count": int(values.size),
        "error_voxels": None,
        "error_fraction": None,
        "disagreement_mean": float(values.mean()),
        "disagreement_q95": float(np.quantile(values, 0.95)),
        "disagreement_correct_mean": None,
        "disagreement_error_mean": None,
        "voxel_auroc": None,
        "voxel_auprc": None,
    }


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float] | None:
    """Patient-resampled confidence interval for a mean."""

    finite = _finite_array(values)
    if finite.size == 0:
        return None
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive.")
    draws = rng.choice(finite, size=(samples, finite.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def paired_summary(
    baseline: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Summarize matched patient values without treating shifts as independent."""

    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.shape != candidate_array.shape:
        raise ValueError("Paired arrays must have the same shape.")
    finite = np.isfinite(baseline_array) & np.isfinite(candidate_array)
    baseline_array = baseline_array[finite]
    candidate_array = candidate_array[finite]
    if baseline_array.size == 0:
        return {
            "patients": 0,
            "baseline_mean": None,
            "candidate_mean": None,
            "mean_paired_difference": None,
            "median_paired_difference": None,
            "paired_bootstrap_95pct_ci": None,
            "improved": 0,
            "tied": 0,
            "worsened": 0,
        }
    difference = candidate_array - baseline_array
    tied = np.abs(difference) <= tie_tolerance
    return {
        "patients": int(difference.size),
        "baseline_mean": float(baseline_array.mean()),
        "candidate_mean": float(candidate_array.mean()),
        "mean_paired_difference": float(difference.mean()),
        "median_paired_difference": float(np.median(difference)),
        "paired_bootstrap_95pct_ci": bootstrap_mean_ci(
            difference, samples=bootstrap_samples, rng=rng
        ),
        "improved": int(np.count_nonzero(difference > tie_tolerance)),
        "tied": int(np.count_nonzero(tied)),
        "worsened": int(np.count_nonzero(difference < -tie_tolerance)),
    }


def aggregate_patient_shifts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse shift rows to one independent row per model and patient."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["patient_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, patient_id), patient_rows in sorted(grouped.items()):
        pure = _finite_array(row["pure_equivariance_mean"] for row in patient_rows)
        dice_delta = _finite_array(row["dice_delta_mean"] for row in patient_rows)
        output.append(
            {
                "model": model,
                "patient_id": patient_id,
                "shift_count": len(patient_rows),
                "pure_equivariance_mean": float(pure.mean()) if pure.size else None,
                "pure_equivariance_worst": float(pure.min()) if pure.size else None,
                "dice_delta_mean": float(dice_delta.mean()) if dice_delta.size else None,
                "dice_delta_worst": float(dice_delta.min()) if dice_delta.size else None,
            }
        )
    return output


def summarize_values(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = _finite_array(values)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def grouped_spectrum_summary(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate shift metrics while retaining the requested spectrum strata."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "pure_equivariance_anterior",
        "pure_equivariance_posterior",
        "pure_equivariance_mean",
        "secondary_confidence_weighted_mean",
        "dice_base_mean",
        "dice_shift_mean",
        "dice_delta_mean",
    )
    for key, group in sorted(grouped.items()):
        output.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "patients": len({str(row["patient_id"]) for row in group}),
                **{
                    metric: summarize_values(row.get(metric) for row in group)
                    for metric in metric_names
                },
            }
        )
    return output


def paired_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_model: str,
    candidate_model: str,
    metric_names: Sequence[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
    identity_fields: Sequence[str] = ("patient_id",),
) -> dict[str, dict[str, Any]]:
    """Pair rows by patient (and optional stratum) before computing statistics."""

    by_model: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = tuple(row[field] for field in identity_fields)
        model = str(row["model"])
        if identity in by_model[model]:
            raise ValueError(f"Duplicate row for model={model}, identity={identity}.")
        by_model[model][identity] = row
    baseline_rows = by_model.get(baseline_model, {})
    candidate_rows = by_model.get(candidate_model, {})
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("Compared models do not contain the same paired identities.")
    identities = sorted(baseline_rows)
    return {
        metric: paired_summary(
            [baseline_rows[key].get(metric) for key in identities],
            [candidate_rows[key].get(metric) for key in identities],
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for metric in metric_names
    }


def finite_or_none(value: Any) -> Any:
    """Recursively convert NumPy scalars and nonfinite floats for strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
