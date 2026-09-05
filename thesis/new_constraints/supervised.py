"""Selectable supervised objectives and streaming multiclass diagnostics.

The legacy Dice factory returns the original MONAI loss unchanged. Diagnostics
are observational: they detach logits and retain only scalar/bin totals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


SUPERVISED_LOSS_NAMES = ("dice", "dice_ce")
CALIBRATION_STRATA = (
    "whole",
    "gt_foreground",
    "foreground_union",
    "gt_boundary",
    "correct",
    "incorrect",
)
_MEAN_METRICS = (
    "nll",
    "brier",
    "entropy",
    "normalized_entropy",
    "confidence",
    "accuracy",
    "saturation_fraction",
)


def _validate_loss_choice(name: str, ce_weight: float) -> None:
    if name not in SUPERVISED_LOSS_NAMES:
        raise ValueError(f"Unknown supervised loss {name!r}; choose {SUPERVISED_LOSS_NAMES}.")
    if name == "dice_ce" and (not math.isfinite(ce_weight) or ce_weight <= 0.0):
        raise ValueError("dice_ce requires a finite, positive CE weight.")


def _label_indices(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Validate 3-D class-index targets before converting to integer indices."""

    if logits.ndim != 5 or logits.shape[1] < 2 or logits.numel() == 0:
        raise ValueError("logits must have nonempty shape [B, C>=2, X, Y, Z].")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point.")
    if labels.ndim == logits.ndim:
        if labels.shape[1] != 1:
            raise ValueError("labels must contain class indices, not one-hot channels.")
        labels = labels[:, 0]
    if labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("labels must have shape [B, 1, X, Y, Z] or [B, X, Y, Z].")
    if labels.device != logits.device:
        raise ValueError("logits and labels must use the same device.")
    if labels.is_complex() or not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must contain finite integer class indices.")
    if labels.is_floating_point() and not bool((labels == labels.round()).all()):
        raise ValueError("labels must contain integer class indices.")
    if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
        raise ValueError("label class indices are outside the logits class range.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite.")
    return labels.long()


class DiceCrossEntropyLoss(nn.Module):
    """Legacy multiclass Dice plus weighted, full-volume multiclass CE.

    Dice keeps its original precision and MONAI defaults. CE is evaluated from
    logits in float32 with autocast disabled; it has no clipping, foreground
    grouping, class weights, ignore region, or label smoothing.
    """

    def __init__(self, ce_weight: float = 1.0) -> None:
        super().__init__()
        _validate_loss_choice("dice_ce", ce_weight)
        self.ce_weight = float(ce_weight)
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        indices = _label_indices(logits, labels)
        dice = self.dice(logits, indices.unsqueeze(1))
        with torch.autocast(device_type=logits.device.type, enabled=False):
            ce = F.cross_entropy(logits.float(), indices, reduction="mean")
        return dice + self.ce_weight * ce


def build_supervised_loss(name: str = "dice", ce_weight: float = 1.0) -> nn.Module:
    """Return exact legacy Dice by default, or Dice + ``ce_weight`` * CE."""

    _validate_loss_choice(name, ce_weight)
    if name == "dice":
        return DiceLoss(to_onehot_y=True, softmax=True)
    return DiceCrossEntropyLoss(ce_weight)


def supervised_loss_config(name: str = "dice", ce_weight: float = 1.0) -> dict[str, Any]:
    """Serializable loss identity; an unused CE weight is canonicalized to zero."""

    _validate_loss_choice(name, ce_weight)
    dice = DiceLoss(to_onehot_y=True, softmax=True)
    # Read active defaults, so provenance also reflects the installed MONAI.
    dice_fields = (
        "include_background", "to_onehot_y", "sigmoid", "softmax",
        "squared_pred", "jaccard", "smooth_nr", "smooth_dr", "batch",
    )
    return {
        "name": name,
        "ce_weight": float(ce_weight) if name == "dice_ce" else 0.0,
        "dice": {
            "implementation": "monai.losses.DiceLoss",
            **{key: getattr(dice, key) for key in dice_fields},
            "other_act": None,
            "weight": None,
            "reduction": str(dice.reduction),
        },
        "ce": (
            {
                "implementation": "torch.nn.functional.cross_entropy",
                "input": "full_multiclass_logits",
                "reduction": "mean_over_batch_and_all_voxels",
                "dtype": "float32",
                "class_weights": None,
                "label_smoothing": 0.0,
                "ignored_voxels": False,
            }
            if name == "dice_ce" else None
        ),
    }


def calibration_diagnostics_config() -> dict[str, Any]:
    """Fixed default definitions suitable for an experiment configuration."""

    return {
        "num_bins": 15,
        "binning": "equal_width_floor(confidence * num_bins), confidence_1_in_last_bin",
        "ece": "top_label_confidence_vs_accuracy, voxel_weighted",
        "saturation_threshold": 0.99,
        "saturation_comparison": ">=",
        "foreground": "class_index_nonzero",
        "foreground_union": "GT_foreground_OR_argmax_foreground",
        "gt_boundary": "both_sides_of_6_neighbor_multiclass_GT_transitions_within_FOV",
        "nll": "negative_log_softmax_probability_of_true_class, nats, unclipped",
        "brier": "sum_over_all_classes_of_squared_probability_error, range_0_to_2",
        "entropy": "multiclass_entropy_in_nats",
        "normalized_entropy": "entropy_divided_by_log_num_classes",
        "empty_strata": "zero_count_and_null_means",
        "aggregation": "pooled_voxel_weighted; per_case_rows_returned_by_update",
        "dtype": "float32_with_autocast_disabled",
    }


def _ground_truth_boundary(labels: torch.Tensor) -> torch.Tensor:
    """Mark both sides of in-volume 6-neighbor label transitions."""

    boundary = torch.zeros_like(labels, dtype=torch.bool)
    for axis in range(1, labels.ndim):
        left = [slice(None)] * labels.ndim
        right = [slice(None)] * labels.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        transition = labels[tuple(left)] != labels[tuple(right)]
        boundary[tuple(left)] |= transition
        boundary[tuple(right)] |= transition
    return boundary


@dataclass
class _StratumTotals:
    num_bins: int
    count: int = 0
    sums: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_MEAN_METRICS, 0.0))
    bin_count: list[int] = field(init=False)
    bin_confidence: list[float] = field(init=False)
    bin_correct: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.bin_count = [0] * self.num_bins
        self.bin_confidence = [0.0] * self.num_bins
        self.bin_correct = [0.0] * self.num_bins

    def merge(self, other: _StratumTotals) -> None:
        self.count += other.count
        for key in _MEAN_METRICS:
            self.sums[key] += other.sums[key]
        for index in range(self.num_bins):
            self.bin_count[index] += other.bin_count[index]
            self.bin_confidence[index] += other.bin_confidence[index]
            self.bin_correct[index] += other.bin_correct[index]

    def metrics(self) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {"voxel_count": self.count}
        result.update({key: total / self.count if self.count else None for key, total in self.sums.items()})
        result["ece"] = (
            sum(abs(confidence - correct) for confidence, correct in zip(self.bin_confidence, self.bin_correct))
            / self.count
            if self.count else None
        )
        return result


class CalibrationDiagnostics:
    """Stream pooled and per-case multiclass confidence metrics from logits.

    ``update`` returns flattened rows in batch order; callers may attach case IDs
    and persist them. ``summary`` pools voxels across calls, rather than averaging
    per-case ECE values. Foreground means every class except background class 0.
    Correct/incorrect conditional ECE is descriptive, not an independent measure
    of calibration. Empty strata have null means, never an invented zero score.
    """

    def __init__(
        self,
        num_classes: int,
        num_bins: int = 15,
        saturation_threshold: float = 0.99,
    ) -> None:
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")
        if not isinstance(num_bins, int) or num_bins < 1:
            raise ValueError("num_bins must be a positive integer.")
        if not math.isfinite(saturation_threshold) or not 0.0 < saturation_threshold <= 1.0:
            raise ValueError("saturation_threshold must be finite and in (0, 1].")
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.saturation_threshold = float(saturation_threshold)
        self.case_count = 0
        self._totals = {name: _StratumTotals(num_bins) for name in CALIBRATION_STRATA}

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, int | float | None]]:
        indices = _label_indices(logits, labels)
        if logits.shape[1] != self.num_classes:
            raise ValueError("logits class count differs from diagnostic num_classes.")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probabilities = F.log_softmax(logits.detach().float(), dim=1)
            probabilities = log_probabilities.exp()
            confidence, prediction = probabilities.max(dim=1)
            correct = prediction == indices
            target_probability = probabilities.gather(1, indices.unsqueeze(1))[:, 0]
            values = {
                "nll": -log_probabilities.gather(1, indices.unsqueeze(1))[:, 0],
                "entropy": -(probabilities * log_probabilities).sum(dim=1),
                "confidence": confidence,
                "accuracy": correct.float(),
                "saturation_fraction": (confidence >= self.saturation_threshold).float(),
            }
            # Sum non-target p^2 and (1-p_y)^2 separately to avoid cancellation
            # near a correct probability of one, without a dense one-hot target.
            other_squared = probabilities.square()
            other_squared.scatter_(1, indices.unsqueeze(1), 0.0)
            values["brier"] = other_squared.sum(dim=1) + (1.0 - target_probability).square()
            values["normalized_entropy"] = values["entropy"] / math.log(self.num_classes)
            foreground = indices != 0
            masks = {
                "whole": torch.ones_like(indices, dtype=torch.bool),
                "gt_foreground": foreground,
                "foreground_union": foreground | (prediction != 0),
                "gt_boundary": _ground_truth_boundary(indices),
                "correct": correct,
                "incorrect": ~correct,
            }
            bins = (confidence * self.num_bins).long().clamp_max(self.num_bins - 1)
            rows: list[dict[str, int | float | None]] = []
            for case_index in range(indices.shape[0]):
                row: dict[str, int | float | None] = {}
                for stratum, mask in masks.items():
                    case_mask = mask[case_index]
                    totals = _StratumTotals(self.num_bins)
                    totals.count = int(case_mask.sum().item())
                    if totals.count:
                        # Device reductions stay float32 even under an outer AMP context.
                        reductions = torch.stack([
                            values[key][case_index][case_mask].sum()
                            for key in _MEAN_METRICS
                        ]).cpu().tolist()
                        totals.sums = dict(zip(_MEAN_METRICS, reductions))
                        selected_bins = bins[case_index][case_mask]
                        totals.bin_count = torch.bincount(selected_bins, minlength=self.num_bins).cpu().tolist()
                        totals.bin_confidence = torch.zeros(self.num_bins, device=logits.device, dtype=torch.float32).scatter_add_(0, selected_bins, confidence[case_index][case_mask]).cpu().tolist()
                        totals.bin_correct = torch.zeros(self.num_bins, device=logits.device, dtype=torch.float32).scatter_add_(0, selected_bins, correct[case_index][case_mask].float()).cpu().tolist()
                    self._totals[stratum].merge(totals)
                    row.update({f"calibration/{stratum}/{key}": value for key, value in totals.metrics().items()})
                rows.append(row)
            self.case_count += len(rows)
        return rows

    def summary(self) -> dict[str, int | float | None]:
        """Return a fresh flat mapping of pooled metrics without resetting state."""

        return {
            "calibration/case_count": self.case_count,
            **{
                f"calibration/{stratum}/{key}": value
                for stratum, totals in self._totals.items()
                for key, value in totals.metrics().items()
            },
        }
