"""Unified objective for selectable segmentation constraints."""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .bands import ClassAwareBoundaryTverskyLoss, OuterBoundaryBandLoss
from .constraint_result import ConstraintResult
from .equivariance import Shift3D, TranslationEquivarianceLoss
from .onecut import OuterOneCutLogLTNLoss
from .teacher import TranslationTeacherKLLoss
from .ap_cut import APCutPosteriorLoss


@dataclass(frozen=True)
class NewConstraintConfig:
    """Configuration shared by the selectable constraint presets."""

    equivariance_weight: float = 0.10
    translation_size: int = 2
    equivariance_max_samples: int | None = None
    bands_weight: float = 0.0
    band_steps: int = 2
    bands_focal_gamma: float = 0.0
    bands_inner_focal_gamma: float | None = None
    bands_outer_focal_gamma: float | None = None
    bands_loss_type: str = "focal_bce"
    tversky_false_positive_weight: float = 0.60
    tversky_false_negative_weight: float = 0.40
    foreground_class_ids: tuple[int, ...] = (1, 2)
    complement_class_ids: tuple[int, ...] = (0,)
    onecut_weight: float = 0.0
    onecut_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    onecut_radius_mm: float = 3.0
    onecut_ray_step_mm: float = 0.5
    onecut_tolerance_mm: float = 1.0
    onecut_margin: float = 0.0
    onecut_temperature: float = 1.0
    onecut_max_surface_points: int = 4096
    onecut_geometry_seed: int = 0
    teacher_weight: float = 0.0
    teacher_views: int = 2
    teacher_temperature: float = 1.0
    teacher_support: str = "union"
    ap_cut_weight: float = 0.0
    ap_axis: int = 1
    ap_anterior_low: bool = True
    ap_temperature: float = 1.0


class NewConstraintObjective(nn.Module):
    """Apply one selected auxiliary constraint without owning the model."""

    def __init__(self, config: NewConstraintConfig | None = None) -> None:
        super().__init__()
        self.config = config or NewConstraintConfig()
        for name, weight in (
            ("equivariance_weight", self.config.equivariance_weight),
            ("bands_weight", self.config.bands_weight),
            ("onecut_weight", self.config.onecut_weight),
            ("teacher_weight", self.config.teacher_weight),
            ("ap_cut_weight", self.config.ap_cut_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        active_weights = sum(
            weight > 0
            for weight in (
                self.config.equivariance_weight,
                self.config.bands_weight,
                self.config.onecut_weight,
                self.config.teacher_weight,
                self.config.ap_cut_weight,
            )
        )
        if active_weights > 1:
            raise ValueError(
                "Only one auxiliary constraint weight may be positive in a run."
            )
        if self.config.translation_size < 1:
            raise ValueError("translation_size must be positive.")
        if self.config.bands_weight > 0 and self.config.band_steps != 2:
            raise ValueError(
                "The canonical bands experiment requires exactly two band steps."
            )
        effective_inner_gamma = (
            self.config.bands_focal_gamma
            if self.config.bands_inner_focal_gamma is None
            else self.config.bands_inner_focal_gamma
        )
        effective_outer_gamma = (
            self.config.bands_focal_gamma
            if self.config.bands_outer_focal_gamma is None
            else self.config.bands_outer_focal_gamma
        )
        for name, gamma in (
            ("bands_focal_gamma", self.config.bands_focal_gamma),
            ("bands_inner_focal_gamma", effective_inner_gamma),
            ("bands_outer_focal_gamma", effective_outer_gamma),
        ):
            if not math.isfinite(gamma) or gamma < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.config.bands_weight == 0 and (
            effective_inner_gamma != 0 or effective_outer_gamma != 0
        ):
            raise ValueError("Focal band exponents require a positive bands_weight.")
        if self.config.bands_loss_type not in {"focal_bce", "class_tversky"}:
            raise ValueError("bands_loss_type must be 'focal_bce' or 'class_tversky'.")
        if self.config.bands_loss_type == "class_tversky" and (
            effective_inner_gamma != 0 or effective_outer_gamma != 0
        ):
            raise ValueError("Class-aware Tversky cannot be combined with focal exponents.")
        if (
            self.config.equivariance_max_samples is not None
            and self.config.equivariance_max_samples < 1
        ):
            raise ValueError("equivariance_max_samples must be positive or None.")

        size = self.config.translation_size
        shifts = (
            (size, 0, 0),
            (-size, 0, 0),
            (0, size, 0),
            (0, -size, 0),
            (0, 0, size),
            (0, 0, -size),
        )
        self.equivariance = TranslationEquivarianceLoss(shifts=shifts)
        if self.config.bands_loss_type == "class_tversky":
            self.bands = ClassAwareBoundaryTverskyLoss(
                foreground_class_ids=self.config.foreground_class_ids,
                complement_class_ids=self.config.complement_class_ids,
                steps=self.config.band_steps,
                false_positive_weight=self.config.tversky_false_positive_weight,
                false_negative_weight=self.config.tversky_false_negative_weight,
            )
        else:
            self.bands = OuterBoundaryBandLoss(
                foreground_class_ids=self.config.foreground_class_ids,
                complement_class_ids=self.config.complement_class_ids,
                steps=self.config.band_steps,
                focal_gamma=self.config.bands_focal_gamma,
                inner_focal_gamma=self.config.bands_inner_focal_gamma,
                outer_focal_gamma=self.config.bands_outer_focal_gamma,
            )
        self.onecut = OuterOneCutLogLTNLoss(
            foreground_class_ids=self.config.foreground_class_ids,
            complement_class_ids=self.config.complement_class_ids,
            spacing=self.config.onecut_spacing,
            radius_mm=self.config.onecut_radius_mm,
            ray_step_mm=self.config.onecut_ray_step_mm,
            tolerance_mm=self.config.onecut_tolerance_mm,
            margin=self.config.onecut_margin,
            temperature=self.config.onecut_temperature,
            max_surface_points=self.config.onecut_max_surface_points,
            geometry_seed=self.config.onecut_geometry_seed,
        )
        self.teacher = TranslationTeacherKLLoss(
            num_views=self.config.teacher_views,
            temperature=self.config.teacher_temperature,
            support=self.config.teacher_support,
        )
        self.ap_cut = APCutPosteriorLoss(
            axis=self.config.ap_axis,
            anterior_low=self.config.ap_anterior_low,
            temperature=self.config.ap_temperature,
            invalid_policy="error",
        )

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        shift: Shift3D | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Return the weighted scalar loss and component diagnostics."""

        if logits is None:
            logits = model(images)
        total_loss = logits.float().reshape(-1)[0] * 0.0
        results: dict[str, ConstraintResult] = {}

        if self.config.equivariance_weight > 0:
            equivariance_images = images
            equivariance_logits = logits
            if self.config.equivariance_max_samples is not None:
                limit = min(self.config.equivariance_max_samples, images.shape[0])
                equivariance_images = images[:limit]
                equivariance_logits = logits[:limit]
            result = self.equivariance(
                model,
                equivariance_images,
                equivariance_logits,
                shift=shift,
                generator=generator,
            )
            results["translation_equivariance"] = result
            total_loss = total_loss + self.config.equivariance_weight * result.loss

        if self.config.bands_weight > 0:
            if labels is None:
                raise ValueError("The bands constraint requires transformed labels.")
            result = self.bands(logits, labels)
            results["outer_boundary_band"] = result
            total_loss = total_loss + self.config.bands_weight * result.loss

        if self.config.onecut_weight > 0:
            if labels is None:
                raise ValueError("The one-cut constraint requires transformed labels.")
            result = self.onecut(logits, labels)
            results["outer_onecut"] = result
            total_loss = total_loss + self.config.onecut_weight * result.loss

        if self.config.teacher_weight > 0:
            result = self.teacher(model, images, logits, generator=generator)
            results["translation_teacher_kl"] = result
            total_loss = total_loss + self.config.teacher_weight * result.loss

        if self.config.ap_cut_weight > 0:
            if labels is None:
                raise ValueError("The A/P cut constraint requires transformed labels.")
            result = self.ap_cut(logits, labels)
            results["ap_cut_posterior"] = result
            total_loss = total_loss + self.config.ap_cut_weight * result.loss

        return {
            "loss": total_loss,
            "results": results,
            "truth": {name: result.truth for name, result in results.items()},
            "value": {name: result.value for name, result in results.items()},
            "confidence_weighted_agreement": {
                name: result.details["confidence_weighted_agreement"]
                for name, result in results.items()
            },
            "confidence_adherent": {
                name: result.details["confidence_adherent"]
                for name, result in results.items()
            },
        }
