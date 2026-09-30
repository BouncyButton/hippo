"""Frozen SwinUNETR feature pooling and subject-wise cut-ranking probes.

The probe never updates the segmentation network.  Hidden activations are
pooled over anatomically defined regions of each predicted hippocampal slice,
normalised within subject, and used by a small linear classifier to rank the
possible anterior/posterior cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


REGION_NAMES = (
    "union_mean",
    "union_max",
    "superior_band_mean",
    "context_mean",
)
OPERATION_NAMES = ("current", "delta", "next_delta", "local_contrast")


@dataclass(frozen=True)
class ProbeCase:
    name: str
    candidates: np.ndarray
    target: int
    low: int
    high: int
    layer_features: dict[str, np.ndarray]
    feature_names: dict[str, tuple[str, ...]]


def _resized_mask_plane(mask: np.ndarray, y: int, shape: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    plane = torch.as_tensor(mask[:, y, :], dtype=torch.float32)[None, None]
    native_shape = tuple(int(value) for value in plane.shape[-2:])
    if all(target <= native for target, native in zip(shape, native_shape, strict=True)):
        # Max pooling, unlike nearest-neighbour interpolation, cannot erase a
        # thin endpoint or narrow superior protrusion at decoder resolution.
        resized = functional.adaptive_max_pool2d(plane, output_size=shape)[0, 0]
    else:
        resized = functional.interpolate(plane, size=shape, mode="nearest")[0, 0]
    return resized.numpy() > 0.5


def _region_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    if mask.ndim != 2 or not mask.any():
        raise ValueError("slice mask must be non-empty and two-dimensional")
    coordinates = np.argwhere(mask)
    superior_start = int(np.median(coordinates[:, 1]))
    superior_half = np.zeros_like(mask)
    superior_half[:, superior_start:] = True
    superior_band = (
        ndimage.binary_dilation(mask, iterations=2) & ~mask & superior_half
    )
    context = ndimage.binary_dilation(mask, iterations=3)
    if not superior_band.any():
        superior_band = context & ~mask
    return {
        "union_mean": mask,
        "union_max": mask,
        "superior_band_mean": superior_band,
        "context_mean": context,
    }


def pool_layer_sequence(
    activation: "object",
    union: np.ndarray,
    *,
    low: int,
    high: int,
    layer_name: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Pool one ``[C, X, Y, Z]`` activation over each occupied coronal slice."""

    import torch
    import torch.nn.functional as functional

    if activation.ndim != 4:
        raise ValueError("activation must have shape [channels, x, y, z]")
    channels, size_x, _, size_z = activation.shape
    # Interpolate only along the A/P dimension. This gives each 64-grid
    # candidate a distinct, smoothly varying vector even for half-resolution
    # decoder features while retaining native in-plane feature resolution.
    sequence = functional.interpolate(
        activation.detach().float()[None],
        size=(size_x, union.shape[1], size_z),
        mode="trilinear",
        align_corners=False,
    )[0].cpu().numpy()

    base_rows = []
    for y in range(low, high + 1):
        mask = _resized_mask_plane(union, y, (size_x, size_z))
        regions = _region_masks(mask)
        plane = sequence[:, :, y, :]
        row = []
        for region in REGION_NAMES:
            values = plane[:, regions[region]]
            if values.shape[1] == 0:
                pooled = np.zeros(channels, dtype=np.float32)
            elif region == "union_max":
                pooled = values.max(axis=1)
            else:
                pooled = values.mean(axis=1)
            row.extend(pooled.tolist())
        base_rows.append(row)
    base = np.asarray(base_rows, dtype=np.float32)
    mean = base.mean(axis=0)
    scale = base.std(axis=0)
    scale[scale < 1e-6] = 1.0
    base = (base - mean) / scale

    rows = []
    for index in range(1, base.shape[0]):
        current = base[index]
        previous = base[index - 1]
        following = base[min(index + 1, base.shape[0] - 1)]
        rows.append(
            np.concatenate(
                (
                    current,
                    current - previous,
                    following - current,
                    current - 0.5 * (previous + following),
                )
            )
        )
    matrix = np.asarray(rows, dtype=np.float32)
    names = tuple(
        f"{layer_name}__{operation}__{region}__channel_{channel:02d}"
        for operation in OPERATION_NAMES
        for region in REGION_NAMES
        for channel in range(channels)
    )
    if matrix.shape != (high - low, len(names)):
        raise AssertionError("feature matrix schema drift")
    if not np.isfinite(matrix).all():
        raise ValueError("pooled features must be finite")
    return matrix, names


def region_columns(feature_names: Iterable[str], regions: tuple[str, ...]) -> np.ndarray:
    selected = [
        index
        for index, name in enumerate(feature_names)
        if any(f"__{region}__" in name for region in regions)
    ]
    if not selected:
        raise ValueError(f"no feature columns matched regions {regions}")
    return np.asarray(selected, dtype=np.int64)


def position_statistics(cases: Iterable[ProbeCase]) -> tuple[float, float]:
    positions = np.asarray(
        [
            (case.target - case.low) / max(case.high - case.low, 1)
            for case in cases
        ],
        dtype=np.float64,
    )
    median = float(np.median(positions))
    robust_scale = float(1.4826 * np.median(np.abs(positions - median)))
    return median, max(robust_scale, 0.075)


def position_matrix(case: ProbeCase, statistics: tuple[float, float]) -> np.ndarray:
    relative = (case.candidates - case.low) / max(case.high - case.low, 1)
    median, scale = statistics
    soft_prior = -0.5 * ((relative - median) / scale) ** 2
    return np.column_stack((relative, relative**2, soft_prior)).astype(np.float32)


def make_design(
    case: ProbeCase,
    *,
    layer: str | None,
    regions: tuple[str, ...],
    include_position: bool,
    statistics: tuple[float, float],
) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks = []
    names: list[str] = []
    if layer is not None:
        indices = region_columns(case.feature_names[layer], regions)
        blocks.append(case.layer_features[layer][:, indices])
        names.extend(case.feature_names[layer][index] for index in indices)
    if include_position:
        blocks.append(position_matrix(case, statistics))
        names.extend(
            (
                "position__relative",
                "position__relative_squared",
                "position__soft_log_prior",
            )
        )
    if not blocks:
        raise ValueError("design requires hidden features or position")
    return np.column_stack(blocks), tuple(names)


def fit_ranker(
    cases: list[ProbeCase],
    *,
    layer: str | None,
    regions: tuple[str, ...],
    include_position: bool,
    c_value: float,
    statistics: tuple[float, float],
) -> tuple[Pipeline, tuple[str, ...], list[str]]:
    rows, targets, weights = [], [], []
    used_names = []
    feature_names: tuple[str, ...] | None = None
    for case in cases:
        if case.target not in set(case.candidates.tolist()):
            continue
        matrix, current_names = make_design(
            case,
            layer=layer,
            regions=regions,
            include_position=include_position,
            statistics=statistics,
        )
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise AssertionError("feature names differ across cases")
        rows.append(matrix)
        targets.extend((case.candidates == case.target).astype(np.uint8).tolist())
        weights.extend([1.0 / len(case.candidates)] * len(case.candidates))
        used_names.append(case.name)
    if not rows or feature_names is None:
        raise ValueError("no cases have an in-support target")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=0,
        ),
    )
    model.fit(np.row_stack(rows), np.asarray(targets), logisticregression__sample_weight=weights)
    return model, feature_names, used_names


def predict_ranker(
    model: Pipeline,
    cases: Iterable[ProbeCase],
    *,
    layer: str | None,
    regions: tuple[str, ...],
    include_position: bool,
    statistics: tuple[float, float],
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    predictions, scores = {}, {}
    for case in cases:
        matrix, _ = make_design(
            case,
            layer=layer,
            regions=regions,
            include_position=include_position,
            statistics=statistics,
        )
        probabilities = model.predict_proba(matrix)[:, 1]
        predictions[case.name] = int(case.candidates[int(np.argmax(probabilities))])
        scores[case.name] = probabilities
    return predictions, scores


def cut_metrics(predictions: dict[str, int], cases: Iterable[ProbeCase]) -> dict[str, float | int]:
    by_name = {case.name: case for case in cases}
    errors = np.asarray(
        [abs(cut - by_name[name].target) for name, cut in sorted(predictions.items())],
        dtype=np.float64,
    )
    endpoint = np.asarray(
        [
            min(cut - by_name[name].low, by_name[name].high - cut) <= 1
            for name, cut in sorted(predictions.items())
        ],
        dtype=bool,
    )
    return {
        "count": int(errors.size),
        "mae_slices": float(errors.mean()),
        "median_absolute_error": float(np.median(errors)),
        "exact_fraction": float((errors == 0).mean()),
        "within_1_fraction": float((errors <= 1).mean()),
        "within_2_fraction": float((errors <= 2).mean()),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
        "maximum_absolute_error": int(errors.max()),
        "endpoint_prediction_fraction": float(endpoint.mean()),
    }
