"""Spacing-aware, head-free one-cut logic on outer hippocampal normal rays."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from ..bands.outer_boundary import _normalise_labels, foreground_log_odds
from ..constraint_result import ConstraintResult, differentiable_zero


@dataclass(frozen=True)
class _CachedRays:
    coordinates: np.ndarray
    edge_touching: bool


def _validate_spacing(spacing: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in spacing)
    if len(values) != 3 or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("spacing must contain three finite positive values.")
    return values


def _outer_face_centres(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for axis in range(3):
        for direction in (-1, 1):
            foreground_slice = [slice(None)] * 3
            neighbour_slice = [slice(None)] * 3
            if direction > 0:
                foreground_slice[axis] = slice(0, -1)
                neighbour_slice[axis] = slice(1, None)
            else:
                foreground_slice[axis] = slice(1, None)
                neighbour_slice[axis] = slice(0, -1)
            faces = mask[tuple(foreground_slice)] & ~mask[tuple(neighbour_slice)]
            indices = np.argwhere(faces)
            if indices.size == 0:
                continue
            if direction < 0:
                indices[:, axis] += 1
            centres = indices.astype(np.float64)
            centres[:, axis] += 0.5 * direction
            face_direction = np.zeros((len(indices), 3), dtype=np.float64)
            face_direction[:, axis] = direction
            points.append(centres)
            directions.append(face_direction)
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(points), np.concatenate(directions)


def _signed_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return outside - inside


def _surface_normals(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    points: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    sdf = _signed_distance(mask, spacing)
    gradients = np.stack(np.gradient(sdf, *spacing, edge_order=1), axis=0)
    sampled = np.stack(
        [
            ndimage.map_coordinates(
                gradients[axis], points.T, order=1, mode="nearest", prefilter=False
            )
            for axis in range(3)
        ],
        axis=1,
    )
    norms = np.linalg.norm(sampled, axis=1, keepdims=True)
    unreliable = (~np.isfinite(norms[:, 0])) | (norms[:, 0] < 1e-6)
    sampled = sampled / np.maximum(norms, 1e-6)
    sampled[unreliable] = fallback[unreliable]
    sampled[np.sum(sampled * fallback, axis=1) < 0] *= -1.0
    return sampled


def _edge_touching(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def _sample_volume(volume: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    size = torch.tensor(volume.shape, device=coordinates.device, dtype=coordinates.dtype)
    grid = (2.0 * coordinates / (size - 1.0) - 1.0).flip(-1)
    grid = grid.reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        volume.reshape(1, 1, *volume.shape),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(coordinates.shape[:-1])


def _onecut_components(
    values: torch.Tensor,
    offsets_mm: torch.Tensor,
    *,
    tolerance_mm: float,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cut_midpoints = 0.5 * (offsets_mm[:-1] + offsets_mm[1:])
    allowed = cut_midpoints.abs() <= tolerance_mm + 1e-7
    if not bool(allowed.any()):
        raise ValueError("No one-cut candidate lies inside the tolerance interval.")
    foreground_log_truth = F.logsigmoid((values - margin) / temperature)
    background_log_truth = F.logsigmoid((-values - margin) / temperature)
    foreground_prefix = torch.cumsum(foreground_log_truth, dim=1)[:, :-1]
    background_prefix = torch.cumsum(background_log_truth, dim=1)
    background_suffix = background_prefix[:, -1:] - background_prefix[:, :-1]
    all_cut_scores = foreground_prefix + background_suffix
    allowed_scores = all_cut_scores[:, allowed]
    log_truth = torch.logsumexp(allowed_scores, dim=1)
    all_log_mass = torch.logsumexp(all_cut_scores, dim=1)
    normalized_truth = torch.exp(log_truth / values.shape[1]).clamp(0.0, 1.0)
    allowed_mass = torch.exp(log_truth - all_log_mass).clamp(0.0, 1.0)
    return -log_truth / values.shape[1], normalized_truth, allowed_mass


class OuterOneCutLogLTNLoss(nn.Module):
    """Patient-balanced one-cut loss using deterministic cached GT-normal rays."""

    def __init__(
        self,
        *,
        foreground_class_ids: Sequence[int] = (1, 2),
        complement_class_ids: Sequence[int] = (0,),
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        radius_mm: float = 3.0,
        ray_step_mm: float = 0.5,
        tolerance_mm: float = 1.0,
        margin: float = 0.0,
        temperature: float = 1.0,
        max_surface_points: int = 4096,
        geometry_seed: int = 0,
        cache_size: int = 512,
        adherence_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        self.foreground_class_ids = tuple(int(value) for value in foreground_class_ids)
        self.complement_class_ids = tuple(int(value) for value in complement_class_ids)
        self.spacing = _validate_spacing(spacing)
        for name, value, allow_zero in (
            ("radius_mm", radius_mm, False),
            ("ray_step_mm", ray_step_mm, False),
            ("tolerance_mm", tolerance_mm, True),
            ("margin", margin, True),
            ("temperature", temperature, False),
        ):
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"{name} has an invalid value.")
        if tolerance_mm >= radius_mm:
            raise ValueError("tolerance_mm must be smaller than radius_mm.")
        if max_surface_points < 1 or cache_size < 1:
            raise ValueError("max_surface_points and cache_size must be positive.")
        if not 0.0 <= adherence_threshold <= 1.0:
            raise ValueError("adherence_threshold must lie in [0,1].")
        offsets = np.arange(-radius_mm, radius_mm + 0.5 * ray_step_mm, ray_step_mm)
        offsets = np.unique(np.concatenate((offsets, [0.0])))
        if len(offsets) < 3 or offsets[0] > -radius_mm or offsets[-1] < radius_mm:
            raise ValueError("Ray sampling failed to cover the requested radius.")
        self.radius_mm = float(radius_mm)
        self.ray_step_mm = float(ray_step_mm)
        self.tolerance_mm = float(tolerance_mm)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.max_surface_points = int(max_surface_points)
        self.geometry_seed = int(geometry_seed)
        self.cache_size = int(cache_size)
        self.adherence_threshold = float(adherence_threshold)
        self.register_buffer("offsets_mm", torch.as_tensor(offsets, dtype=torch.float32))
        self._cache: OrderedDict[str, _CachedRays] = OrderedDict()

    def _cache_key(self, mask: np.ndarray) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        return digest.hexdigest()

    def _build_rays(self, mask: np.ndarray, key: str) -> _CachedRays:
        points, fallback = _outer_face_centres(mask)
        if len(points) == 0:
            return _CachedRays(np.empty((0, len(self.offsets_mm), 3), dtype=np.float32), _edge_touching(mask))
        normals = _surface_normals(mask, self.spacing, points, fallback)
        spacing = np.asarray(self.spacing, dtype=np.float64)
        displacement = self.radius_mm * normals / spacing[None, :]
        lower = points - displacement
        upper = points + displacement
        shape_limit = np.asarray(mask.shape, dtype=np.float64) - 1.0
        valid = np.all(
            (lower >= 0.0) & (lower <= shape_limit) & (upper >= 0.0) & (upper <= shape_limit),
            axis=1,
        )
        points = points[valid]
        normals = normals[valid]
        if len(points) > self.max_surface_points:
            deterministic_seed = self.geometry_seed ^ int(key[:16], 16)
            selected = np.random.default_rng(deterministic_seed).choice(
                len(points), self.max_surface_points, replace=False
            )
            selected.sort()
            points = points[selected]
            normals = normals[selected]
        offsets = self.offsets_mm.detach().cpu().numpy().astype(np.float64)
        coordinates = (
            points[:, None, :]
            + offsets[None, :, None] * normals[:, None, :] / spacing[None, None, :]
        ).astype(np.float32)
        return _CachedRays(coordinates, _edge_touching(mask))

    def _rays(self, mask: np.ndarray) -> _CachedRays:
        key = self._cache_key(mask)
        cached = self._cache.pop(key, None)
        if cached is None:
            cached = self._build_rays(mask, key)
        self._cache[key] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        labels = _normalise_labels(labels, logits)
        if logits.shape[1] != 3:
            raise ValueError("The canonical one-cut experiment requires three logits.")
        fields = foreground_log_odds(
            logits, self.foreground_class_ids, self.complement_class_ids
        )
        case_losses: list[torch.Tensor] = []
        case_truths: list[torch.Tensor] = []
        case_masses: list[torch.Tensor] = []
        ray_counts: list[int] = []
        edge_touching: list[bool] = []
        valid: list[bool] = []
        for batch_index in range(logits.shape[0]):
            label = labels[batch_index]
            foreground = torch.zeros_like(label, dtype=torch.bool)
            for class_id in self.foreground_class_ids:
                foreground |= label == class_id
            cached = self._rays(foreground.detach().cpu().numpy())
            count = int(cached.coordinates.shape[0])
            ray_counts.append(count)
            edge_touching.append(cached.edge_touching)
            valid.append(count > 0)
            if count == 0:
                zero = differentiable_zero(fields[batch_index])
                case_losses.append(zero)
                case_truths.append(zero)
                case_masses.append(zero)
                continue
            coordinates = torch.as_tensor(
                cached.coordinates, device=logits.device, dtype=torch.float32
            )
            values = _sample_volume(fields[batch_index], coordinates)
            ray_loss, ray_truth, ray_mass = _onecut_components(
                values,
                self.offsets_mm,
                tolerance_mm=self.tolerance_mm,
                margin=self.margin,
                temperature=self.temperature,
            )
            case_losses.append(ray_loss.mean())
            case_truths.append(ray_truth.mean())
            case_masses.append(ray_mass.mean())

        case_loss = torch.stack(case_losses)
        case_truth = torch.stack(case_truths)
        allowed_mass = torch.stack(case_masses)
        valid_tensor = torch.as_tensor(valid, device=logits.device, dtype=torch.bool)
        if bool(valid_tensor.any()):
            loss = case_loss[valid_tensor].mean()
            truth = case_truth[valid_tensor]
            value = torch.stack((case_truth[valid_tensor], allowed_mass[valid_tensor]), dim=1)
        else:
            loss = differentiable_zero(logits.float())
            truth = case_truth[valid_tensor]
            value = torch.empty((0, 2), device=logits.device, dtype=torch.float32)
        ray_count_tensor = torch.as_tensor(ray_counts, device=logits.device, dtype=torch.float32)
        edge_tensor = torch.as_tensor(edge_touching, device=logits.device, dtype=torch.bool)
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=value,
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid_tensor,
                "case_loss": case_loss,
                "case_truth": case_truth,
                "allowed_cut_mass": allowed_mass,
                "ray_count": ray_count_tensor,
                "edge_touching": edge_tensor,
                "metrics": {
                    "raw_loss": case_loss[valid_tensor],
                    "normalized_truth": case_truth[valid_tensor],
                    "allowed_cut_mass": allowed_mass[valid_tensor],
                    "ray_count": ray_count_tensor,
                    "valid_patient": valid_tensor.float(),
                    "skipped_patient": (~valid_tensor).float(),
                    "edge_touching": edge_tensor.float(),
                },
            },
        )
