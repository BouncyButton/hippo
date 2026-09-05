"""Supervised, case-balanced posterior over axis-aligned A/P cut positions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraint_result import ConstraintResult


AP_CUT_METRICS = (
    "raw_loss",
    "gt_cut_probability",
    "cut_abs_error",
    "exact_cut",
    "valid_patient",
    "skipped_patient",
)


@dataclass(frozen=True)
class APCutConfig:
    """Tensor geometry must be explicit; this module does not read NIfTI affines."""

    axis: int
    anterior_low: bool
    temperature: float = 1.0
    invalid_policy: str = "error"

    def __post_init__(self) -> None:
        if type(self.axis) is not int or self.axis not in (0, 1, 2):
            raise ValueError("axis must be a spatial tensor axis: 0, 1, or 2.")
        if type(self.anterior_low) is not bool:
            raise ValueError("anterior_low must explicitly be True or False.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if self.invalid_policy not in {"error", "skip"}:
            raise ValueError("invalid_policy must be 'error' or 'skip'.")


@dataclass
class APCutPosterior:
    """One case's candidates; cut k lies between tensor slices k and k+1."""

    valid: bool
    reason: str | None
    cut_indices: torch.Tensor
    scores: torch.Tensor
    log_probs: torch.Tensor
    gt_index: int | None
    gt_cut: int | None
    occupied_slices: int


def _validate_inputs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5 or logits.shape[1] != 3 or min(logits.shape) < 1:
        raise ValueError("logits must have shape [B,3,D,H,W] with nonempty dimensions.")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite.")
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if tuple(labels.shape) != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("labels must have shape [B,D,H,W] or [B,1,D,H,W].")
    labels = labels.detach().to(device=logits.device)
    if not bool(((labels == 0) | (labels == 1) | (labels == 2)).all()):
        raise ValueError("labels must contain only background=0, anterior=1, posterior=2.")
    return labels


def compute_ap_cut_posterior(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: APCutConfig,
) -> list[APCutPosterior]:
    """Compute GT-supported candidate scores for training or a frozen-logit audit.

    Candidate scores average conditional A/P log probability within each occupied
    GT-foreground slice, then across occupied slices. No background logit or
    predicted foreground mask enters the score. GT labels identify the target
    candidate and validity; they never choose or truncate the candidate interval
    around the known answer. This posterior needs labels and is not an inference
    decoder. Floating-point log probabilities are computed stably in FP32 (or
    FP64 when the input is FP64).
    """

    labels = _validate_inputs(logits, labels)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    conditional = F.log_softmax(logits[:, 1:3].to(dtype=dtype), dim=1)
    posteriors: list[APCutPosterior] = []
    low_class, high_class = (1, 2) if config.anterior_low else (2, 1)
    for case_index in range(logits.shape[0]):
        # Shape [axis_length, other_voxels]; counts are detached GT geometry.
        label = labels[case_index].movedim(config.axis, 0).reshape(
            labels.shape[config.axis + 1], -1
        )
        low_count = (label == low_class).sum(dim=1)
        high_count = (label == high_class).sum(dim=1)
        counts = low_count + high_count
        occupied = torch.nonzero(counts > 0, as_tuple=False).flatten()
        low_slices = torch.nonzero(low_count > 0, as_tuple=False).flatten()
        high_slices = torch.nonzero(high_count > 0, as_tuple=False).flatten()
        reason = None
        if low_slices.numel() == 0 or high_slices.numel() == 0:
            reason = "both A/P classes must be present"
        elif bool(((low_count > 0) & (high_count > 0)).any()):
            reason = "a foreground slice contains both A/P labels (nonplanar for this axis)"
        elif int(low_slices[-1]) >= int(high_slices[0]):
            reason = "labels violate the configured plane orientation or single cut"
        elif int(high_slices[0]) - int(low_slices[-1]) != 1:
            reason = "an empty-slice gap at the GT transition makes the cut ambiguous"

        if reason is not None:
            if config.invalid_policy == "error":
                raise ValueError(f"Invalid A/P geometry in batch case {case_index}: {reason}.")
            empty = conditional[case_index].reshape(-1)[:0]
            posteriors.append(
                APCutPosterior(
                    False, reason,
                    torch.empty(0, dtype=torch.long, device=logits.device),
                    empty, empty, None, None, int(occupied.numel()),
                )
            )
            continue

        log_probs = conditional[case_index].movedim(config.axis + 1, 1)
        log_probs = log_probs.reshape(2, label.shape[0], -1)
        support = (label != 0).to(dtype=dtype)
        slice_scores = (log_probs * support.unsqueeze(0)).sum(dim=2)
        slice_scores = slice_scores / counts.clamp_min(1).to(dtype=dtype).unsqueeze(0)
        low_prefix = slice_scores[low_class - 1].cumsum(dim=0)
        high_suffix = slice_scores[high_class - 1].flip(0).cumsum(dim=0).flip(0)
        cut_indices = torch.arange(
            int(occupied[0]), int(occupied[-1]), device=logits.device
        )
        scores = (low_prefix[cut_indices] + high_suffix[cut_indices + 1]) / occupied.numel()
        # Temperature refers to these mean-per-slice scores, never voxel totals.
        cut_log_probs = F.log_softmax(scores / config.temperature, dim=0)
        gt_cut = int(low_slices[-1])
        posteriors.append(
            APCutPosterior(
                True, None, cut_indices, scores, cut_log_probs,
                gt_cut - int(occupied[0]), gt_cut, int(occupied.numel()),
            )
        )
    return posteriors


class APCutPosteriorLoss(nn.Module):
    """Negative log posterior of the annotated cut, averaged over valid cases."""

    def __init__(
        self,
        *,
        axis: int,
        anterior_low: bool,
        temperature: float = 1.0,
        invalid_policy: str = "error",
        adherence_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        self.config = APCutConfig(axis, anterior_low, temperature, invalid_policy)
        if not 0.0 <= adherence_threshold <= 1.0:
            raise ValueError("adherence_threshold must lie in [0,1].")
        self.adherence_threshold = float(adherence_threshold)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        posteriors = compute_ap_cut_posterior(logits, labels, self.config)
        # A zero connected only to the A/P channels preserves a valid backward
        # call even for an audit batch with no usable geometry.
        zero = logits[:, 1:3].float().reshape(-1)[0] * 0.0
        losses, probabilities, errors, exact = [], [], [], []
        for posterior in posteriors:
            if not posterior.valid:
                losses.append(zero)
                probabilities.append(zero)
                errors.append(zero)
                exact.append(zero)
                continue
            log_probability = posterior.log_probs[posterior.gt_index]
            predicted_cut = posterior.cut_indices[posterior.log_probs.argmax()]
            error = (predicted_cut - posterior.gt_cut).abs().to(log_probability.dtype)
            losses.append(-log_probability)
            probabilities.append(log_probability.exp())
            errors.append(error)
            exact.append((error == 0).to(log_probability.dtype))
        valid = torch.tensor(
            [posterior.valid for posterior in posteriors],
            dtype=torch.bool, device=logits.device,
        )
        case_loss = torch.stack(losses)
        case_probability = torch.stack(probabilities)
        case_error = torch.stack(errors)
        case_exact = torch.stack(exact)
        truth = case_probability[valid]
        loss = case_loss[valid].mean() if bool(valid.any()) else zero
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=case_error[valid],
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid,
                "invalid_reasons": [posterior.reason for posterior in posteriors],
                "case_loss": case_loss,
                "case_truth": case_probability,
                "posteriors": posteriors,
                "metrics": {
                    "raw_loss": case_loss[valid],
                    "gt_cut_probability": truth,
                    "cut_abs_error": case_error[valid],
                    "exact_cut": case_exact[valid],
                    "valid_patient": valid.float(),
                    "skipped_patient": (~valid).float(),
                },
            },
        )
