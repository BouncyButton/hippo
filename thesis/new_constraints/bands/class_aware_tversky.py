"""Class-aware Tversky supervision restricted to anatomical boundary bands."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from ..constraint_result import ConstraintResult, differentiable_zero
from .outer_boundary import _edge_touching, _normalise_labels, build_boundary_bands


class ClassAwareBoundaryTverskyLoss(nn.Module):
    """Average one-vs-rest Tversky losses inside each class boundary band.

    Unlike grouped foreground supervision, each anatomical foreground class owns
    its own two-voxel inner/outer band.  The internal class interface is therefore
    supervised and anterior/posterior swaps cannot disappear inside a grouped
    foreground logit.
    """

    def __init__(
        self,
        *,
        foreground_class_ids: Sequence[int] = (1, 2),
        complement_class_ids: Sequence[int] = (0,),
        steps: int = 2,
        false_positive_weight: float = 0.60,
        false_negative_weight: float = 0.40,
        adherence_threshold: float = 0.90,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.foreground_class_ids = tuple(int(value) for value in foreground_class_ids)
        self.complement_class_ids = tuple(int(value) for value in complement_class_ids)
        if len(self.foreground_class_ids) != 2:
            raise ValueError("Class-aware boundary Tversky requires exactly two foreground classes.")
        if set(self.foreground_class_ids) & set(self.complement_class_ids):
            raise ValueError("Foreground and complement class IDs must be disjoint.")
        all_ids = self.foreground_class_ids + self.complement_class_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("Class IDs cannot contain duplicates.")
        if set(all_ids) != {0, 1, 2}:
            raise ValueError("Class-aware boundary Tversky requires classes 0, 1, and 2.")
        if steps != 2:
            raise ValueError("The canonical boundary Tversky loss requires exactly 2 steps.")
        for name, value in (
            ("false_positive_weight", false_positive_weight),
            ("false_negative_weight", false_negative_weight),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isclose(
            false_positive_weight + false_negative_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Tversky false-positive and false-negative weights must sum to 1.")
        if not 0.0 <= adherence_threshold <= 1.0:
            raise ValueError("adherence_threshold must be between zero and one.")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive.")
        self.steps = int(steps)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)
        self.adherence_threshold = float(adherence_threshold)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        labels = _normalise_labels(labels, logits)
        if logits.ndim != 5 or logits.shape[1] != 3:
            raise ValueError("logits must have shape [B, 3, X, Y, Z].")
        probabilities = torch.softmax(logits.float(), dim=1)
        spatial_dimensions = tuple(range(1, labels.ndim))
        class_losses: list[torch.Tensor] = []
        class_scores: list[torch.Tensor] = []
        band_counts: list[torch.Tensor] = []
        inner_counts: list[torch.Tensor] = []
        outer_counts: list[torch.Tensor] = []
        soft_false_positives: list[torch.Tensor] = []
        soft_false_negatives: list[torch.Tensor] = []
        class_validity: list[torch.Tensor] = []

        with torch.autocast(device_type=logits.device.type, enabled=False):
            for class_id in self.foreground_class_ids:
                target_bool = labels == class_id
                inner, outer = build_boundary_bands(
                    target_bool.unsqueeze(1), steps=self.steps
                )
                inner_float = inner[:, 0].float()
                outer_float = outer[:, 0].float()
                band = inner_float + outer_float
                target = target_bool.float()
                probability = probabilities[:, class_id]
                true_positive = (band * probability * target).sum(spatial_dimensions)
                false_positive = (band * probability * (1.0 - target)).sum(
                    spatial_dimensions
                )
                false_negative = (band * (1.0 - probability) * target).sum(
                    spatial_dimensions
                )
                denominator = (
                    true_positive
                    + self.false_positive_weight * false_positive
                    + self.false_negative_weight * false_negative
                    + self.smooth
                )
                score = (true_positive + self.smooth) / denominator
                class_losses.append(1.0 - score)
                class_scores.append(score)
                band_counts.append(band.sum(spatial_dimensions))
                current_inner_count = inner_float.sum(spatial_dimensions)
                current_outer_count = outer_float.sum(spatial_dimensions)
                inner_counts.append(current_inner_count)
                outer_counts.append(current_outer_count)
                soft_false_positives.append(false_positive)
                soft_false_negatives.append(false_negative)
                class_validity.append(
                    (current_inner_count > 0) & (current_outer_count > 0)
                )

            losses = torch.stack(class_losses, dim=1)
            scores = torch.stack(class_scores, dim=1)
            bands = torch.stack(band_counts, dim=1)
            inners = torch.stack(inner_counts, dim=1)
            outers = torch.stack(outer_counts, dim=1)
            soft_fp = torch.stack(soft_false_positives, dim=1)
            soft_fn = torch.stack(soft_false_negatives, dim=1)
            valid = torch.stack(class_validity, dim=1).all(dim=1)
            case_loss = losses.mean(dim=1)
            if bool(valid.any()):
                loss = case_loss[valid].mean()
            else:
                loss = differentiable_zero(logits.float())

        truth = scores[valid].mean(dim=1)
        foreground = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.foreground_class_ids:
            foreground |= labels == class_id
        edge_touching = _edge_touching(foreground.unsqueeze(1))
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=scores[valid],
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid,
                "case_loss": case_loss,
                "class_losses": losses,
                "class_scores": scores,
                "class_band_voxels": bands,
                "class_inner_voxels": inners,
                "class_outer_voxels": outers,
                "class_soft_false_positives": soft_fp,
                "class_soft_false_negatives": soft_fn,
                # Legacy component slots keep calibration and gradient probes generic.
                "inner_loss": losses[:, 0],
                "outer_loss": losses[:, 1],
                "inner_voxels": bands[:, 0],
                "outer_voxels": bands[:, 1],
                "component_labels": (
                    f"class_{self.foreground_class_ids[0]}_tversky",
                    f"class_{self.foreground_class_ids[1]}_tversky",
                ),
                "edge_touching": edge_touching,
                "metrics": {
                    "raw_loss": case_loss[valid],
                    "inner_loss": losses[valid, 0],
                    "outer_loss": losses[valid, 1],
                    "inner_voxels": bands[:, 0],
                    "outer_voxels": bands[:, 1],
                    "class_1_soft_fp": soft_fp[valid, 0],
                    "class_1_soft_fn": soft_fn[valid, 0],
                    "class_2_soft_fp": soft_fp[valid, 1],
                    "class_2_soft_fn": soft_fn[valid, 1],
                    "valid_patient": valid.float(),
                    "skipped_patient": (~valid).float(),
                    "edge_touching": edge_touching.float(),
                },
            },
        )
