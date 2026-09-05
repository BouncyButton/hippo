"""Case-bundle input/output and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuditCase:
    case_name: str
    logits: np.ndarray
    labels: np.ndarray
    spacing: tuple[float, float, float]
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probabilities_to_logits(probabilities: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError("probabilities must have shape [3,D,H,W].")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities contain invalid values.")
    normalizer = probabilities.sum(axis=0, keepdims=True)
    if np.any(normalizer <= 0):
        raise ValueError("probabilities must have positive per-voxel sums.")
    normalized = probabilities / normalizer
    # log probabilities are a valid representative of the softmax-logit
    # equivalence class; all objectives here are invariant to common offsets.
    return np.log(np.clip(normalized, epsilon, 1.0)).astype(np.float32)


def save_case_bundle(
    path: Path,
    *,
    case_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    spacing: tuple[float, float, float],
) -> None:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if logits.ndim != 4 or logits.shape[0] != 3 or logits.shape[1:] != labels.shape:
        raise ValueError("Expected logits [3,D,H,W] matching labels [D,H,W].")
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("labels must be a three-class [D,H,W] array.")
    if spacing_array.shape != (3,) or not np.isfinite(spacing_array).all() or np.any(spacing_array <= 0):
        raise ValueError("spacing must contain three finite positive values.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinity.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case_name=np.asarray(case_name),
        logits=logits,
        labels=labels,
        spacing=spacing_array,
        schema_version=np.asarray(1, dtype=np.int16),
    )


def load_case_bundle(path: Path) -> AuditCase:
    with np.load(path, allow_pickle=False) as payload:
        required = {"case_name", "logits", "labels", "spacing", "schema_version"}
        if not required.issubset(payload.files):
            raise ValueError(f"Case bundle {path} is missing {sorted(required - set(payload.files))}.")
        if int(payload["schema_version"]) != 1:
            raise ValueError(f"Unsupported case bundle schema in {path}.")
        case_name = str(payload["case_name"].item())
        logits = payload["logits"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        spacing = tuple(float(v) for v in payload["spacing"])
    if logits.shape != (3, *labels.shape):
        raise ValueError(f"Shape mismatch in {path}.")
    if not np.isfinite(logits).all() or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError(f"Invalid values in {path}.")
    return AuditCase(case_name, logits, labels, spacing, path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
