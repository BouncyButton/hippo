"""Trainable slice-error predictors from frozen CST/Swin feature arrays.

Inputs are patient-major arrays of 32 ordered coronal slices. Targets are
needed for fitting/evaluation only; scoring a saved model never reads labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FEATURE_KEYS = {
    "uncertainty": ("slice_uncertainty",),
    "portable": ("slice_uncertainty", "slice_relationship_residuals"),
    "combined": ("slice_combined",),
}


@dataclass(frozen=True)
class PatientFeatures:
    names: np.ndarray
    features: dict[str, np.ndarray]
    target: np.ndarray | None
    source: Path

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def slices(self) -> int:
        return next(iter(self.features.values())).shape[1]


def load_patient_features(path: Path, *, require_target: bool) -> PatientFeatures:
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["case_names"]).astype(str)
        groups = np.asarray(data["slice_groups"]).astype(str)
        if len(names) == 0 or len(set(names)) != len(names):
            raise ValueError("case names must be nonempty and unique")
        if len(groups) % len(names):
            raise ValueError("slice count must be divisible by patient count")
        slices = len(groups) // len(names)
        expected_groups = np.broadcast_to(names[:, None], (len(names), slices))
        if slices < 2 or not np.array_equal(groups.reshape(len(names), slices), expected_groups):
            raise ValueError("slice groups must be patient-major and aligned with case names")
        features = {}
        for kind, keys in FEATURE_KEYS.items():
            blocks = [np.asarray(data[key], dtype=np.float32) for key in keys]
            if any(block.ndim != 2 or len(block) != len(groups) for block in blocks):
                raise ValueError(f"invalid {kind} feature shape")
            array = np.concatenate(blocks, axis=1).reshape(len(names), slices, -1)
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite {kind} features")
            features[kind] = array
        target = None
        if "slice_target" in data.files and require_target:
            target = np.asarray(data["slice_target"], dtype=np.float32).reshape(len(names), slices)
            if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
                raise ValueError("slice targets must be finite errors in [0,1]")
        elif require_target:
            raise ValueError("slice_target is required for training/evaluation")
    return PatientFeatures(names=names, features=features, target=target, source=path)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0).astype(np.float32)
        scale = flat.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.clip((features - self.mean) / self.scale, -5.0, 5.0).astype(np.float32)


def ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    standardizer = Standardizer.fit(features)
    x = standardizer.transform(features).reshape(-1, features.shape[-1]).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    intercept = float(y.mean())
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - intercept))
    return {
        "mean": standardizer.mean,
        "scale": standardizer.scale,
        "weights": weights.astype(np.float32),
        "intercept": intercept,
        "alpha": float(alpha),
    }


def ridge_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardizer = Standardizer(model["mean"], model["scale"])
    transformed = standardizer.transform(features)
    return np.clip(transformed @ model["weights"] + model["intercept"], 0, 1)


class SliceQCNet(nn.Module):
    """Small context model; convolutions operate along coronal position only."""

    def __init__(self, input_features: int, hidden: int = 32, dropout: float = 0.20):
        super().__init__()
        self.input_features = input_features
        self.hidden = hidden
        self.dropout = dropout
        self.input = nn.Sequential(nn.Linear(input_features, hidden), nn.GELU())
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.mix = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_features:
            raise ValueError(f"expected (patients,slices,{self.input_features})")
        hidden = self.input(features)
        contextual = self.mix(self.depthwise(hidden.transpose(1, 2))).transpose(1, 2)
        hidden = hidden + F.dropout(F.gelu(contextual), self.dropout, training=self.training)
        return self.output(hidden).squeeze(-1)


def quality_loss(logits: torch.Tensor, target: torch.Tensor, ranking_weight: float = 0.01) -> torch.Tensor:
    predicted = logits.sigmoid()
    regression = F.smooth_l1_loss(predicted, target, beta=0.10)
    differences = target[:, :, None] - target[:, None, :]
    mask = differences.abs() > 0.10
    if not bool(mask.any()) or ranking_weight == 0:
        return regression
    prediction_differences = logits[:, :, None] - logits[:, None, :]
    ranking = F.softplus(-differences.sign()[mask] * prediction_differences[mask]).mean()
    return regression + ranking_weight * ranking


def fit_temporal(
    features: np.ndarray,
    target: np.ndarray,
    *,
    selection_train: np.ndarray,
    selection_val: np.ndarray,
    seed: int,
    max_epochs: int = 80,
    patience: int = 10,
    hidden: int = 32,
    dropout: float = 0.20,
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[dict[str, Any], int]:
    """Select epoch on inner patients, then refit on all supplied patients."""
    if max_epochs < 1 or patience < 1 or len(selection_train) == 0 or len(selection_val) == 0:
        raise ValueError("nonempty inner patient splits and positive training limits are required")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dimension = features.shape[-1]
    inner_standardizer = Standardizer.fit(features[selection_train])
    inner_x = torch.from_numpy(inner_standardizer.transform(features)).to(device)
    all_y = torch.from_numpy(target).to(device)
    torch.manual_seed(seed)
    model = SliceQCNet(dimension, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best_mse = float("inf")
    best_epoch = 1
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in np.array_split(rng.permutation(selection_train), max(1, int(np.ceil(len(selection_train) / batch_size)))):
            optimizer.zero_grad()
            loss = quality_loss(model(inner_x[batch]), all_y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model(inner_x[selection_val]).sigmoid()
            mse = float(F.mse_loss(predicted, all_y[selection_val]))
        if mse < best_mse - 1e-5:
            best_mse, best_epoch, stale = mse, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break

    full_standardizer = Standardizer.fit(features)
    full_x = torch.from_numpy(full_standardizer.transform(features)).to(device)
    torch.manual_seed(seed)
    final_model = SliceQCNet(dimension, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    for _ in range(best_epoch):
        final_model.train()
        for batch in np.array_split(rng.permutation(len(features)), max(1, int(np.ceil(len(features) / batch_size)))):
            optimizer.zero_grad()
            loss = quality_loss(final_model(full_x[batch]), all_y[batch])
            loss.backward()
            optimizer.step()
    return {
        "schema": "semantic_constraints.cst_teacher.slice_qc.temporal.v1",
        "input_features": dimension,
        "hidden": hidden,
        "dropout": dropout,
        "mean": torch.from_numpy(full_standardizer.mean.copy()),
        "scale": torch.from_numpy(full_standardizer.scale.copy()),
        "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "epochs": best_epoch,
        "seed": seed,
    }, best_epoch


@torch.no_grad()
def temporal_predict(artifact: dict[str, Any], features: np.ndarray, device: str = "cpu") -> np.ndarray:
    if features.shape[-1] != artifact["input_features"]:
        raise ValueError("feature dimension does not match saved model")
    model = SliceQCNet(
        artifact["input_features"], hidden=artifact["hidden"], dropout=artifact["dropout"]
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    normalized = Standardizer(
        torch.as_tensor(artifact["mean"]).cpu().numpy(),
        torch.as_tensor(artifact["scale"]).cpu().numpy(),
    ).transform(features)
    return model(torch.from_numpy(normalized).to(device)).sigmoid().cpu().numpy()


def save_ridge(path: Path, model: dict[str, Any], *, kind: str, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema="semantic_constraints.cst_teacher.slice_qc.ridge.v1",
        kind=kind,
        metadata=json.dumps(metadata),
        **model,
    )


def load_ridge(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != "semantic_constraints.cst_teacher.slice_qc.ridge.v1":
            raise ValueError("unsupported ridge model schema")
        kind = str(data["kind"])
        metadata = json.loads(str(data["metadata"]))
        model = {key: data[key] for key in ("mean", "scale", "weights")}
        model["intercept"] = float(data["intercept"])
        model["alpha"] = float(data["alpha"])
    return kind, model, metadata
