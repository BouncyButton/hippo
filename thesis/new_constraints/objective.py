"""Unified objective for selectable segmentation constraints."""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .bands import OuterBoundaryBandLoss
from .constraint_result import ConstraintResult
from .equivariance import Shift3D, TranslationEquivarianceLoss


@dataclass(frozen=True)
class NewConstraintConfig:
    """Configuration shared by the none, equivariance, and bands presets."""

    equivariance_weight: float = 0.10
    translation_size: int = 2
    equivariance_max_samples: int | None = None
    bands_weight: float = 0.0
    band_steps: int = 2
    foreground_class_ids: tuple[int, ...] = (1, 2)
    complement_class_ids: tuple[int, ...] = (0,)


class NewConstraintObjective(nn.Module):
    """Apply one selected auxiliary constraint without owning the model."""

    def __init__(self, config: NewConstraintConfig | None = None) -> None:
        super().__init__()
        self.config = config or NewConstraintConfig()
        for name, weight in (
            ("equivariance_weight", self.config.equivariance_weight),
            ("bands_weight", self.config.bands_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.config.equivariance_weight > 0 and self.config.bands_weight > 0:
            raise ValueError(
                "equivariance_weight and bands_weight cannot both be positive in a "
                "single constraint-set run."
            )
        if self.config.translation_size < 1:
            raise ValueError("translation_size must be positive.")
        if self.config.bands_weight > 0 and self.config.band_steps != 2:
            raise ValueError(
                "The canonical bands experiment requires exactly two band steps."
            )
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
        self.bands = OuterBoundaryBandLoss(
            foreground_class_ids=self.config.foreground_class_ids,
            complement_class_ids=self.config.complement_class_ids,
            steps=self.config.band_steps,
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
