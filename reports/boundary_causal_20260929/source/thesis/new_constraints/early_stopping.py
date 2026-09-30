"""Validation-Dice stopping policy, independent of best-checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    patience: int = 0
    min_delta: float = 0.0005
    min_epochs: int = 25
    epoch: int = 0
    reference_score: float = -math.inf
    reference_epoch: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0 or self.min_epochs < 1:
            raise ValueError("Early-stopping patience must be nonnegative and min epochs positive.")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early-stopping min delta must be finite and nonnegative.")

    def update(self, epoch: int, score: float) -> None:
        if epoch != self.epoch + 1:
            raise ValueError("Early stopping requires consecutive epoch observations.")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Early stopping requires finite validation Dice in [0, 1].")
        self.epoch = epoch
        if score > self.reference_score + self.min_delta:
            self.reference_score = score
            self.reference_epoch = epoch

    @property
    def should_stop(self) -> bool:
        return (
            self.patience > 0
            and self.epoch >= self.min_epochs
            and self.epoch - self.reference_epoch >= self.patience
        )

    def summary(self) -> dict[str, int | float | bool]:
        return {
            "enabled": self.patience > 0,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "stopped_epoch": self.epoch,
            "reference_score": self.reference_score,
            "reference_epoch": self.reference_epoch,
            "epochs_without_significant_improvement": self.epoch - self.reference_epoch,
            "triggered": self.should_stop,
        }
