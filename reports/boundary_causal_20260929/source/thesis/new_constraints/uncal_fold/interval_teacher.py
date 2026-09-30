"""Train-only calibrated intervals from the existing shape/MRI cut proxy.

Intervals describe label-compatible cut planes, not verified anatomical apexes.
No guarantee transfers from GT-union calibration to model-predicted unions.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .audit_refined import LoadedCase, _candidate_matrix, _training_position_statistics
from .foldedness import CaseFeatures
from .refined import extract_refined_case_features

MODE = "compact_geometry_image_plus_soft_position"


def label_transition_interval(labels: np.ndarray) -> tuple[float, float]:
    """Smallest plane-coordinate interval that leaves all mixed slices free.

    Outside [min_A_y-.5, max_P_y+.5], pure RAS-y side implications agree
    with every original label. Endpoints are ordered for reversed annotations,
    in which case the large interval exposes rather than hides incompatibility.
    """
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Expected a 3-D background/A/P label array")
    a = np.flatnonzero((labels == 1).any(axis=(0, 2)))
    p = np.flatnonzero((labels == 2).any(axis=(0, 2)))
    if not a.size or not p.size:
        raise ValueError("Both A/P labels are required")
    # Disjoint classes separated by an empty gap permit any intervening plane.
    return tuple(sorted((float(a.min() - .5), float(p.max() + .5))))


def calibration_radius(predicted_planes: np.ndarray, reference_intervals: np.ndarray,
                       alpha: float = .10) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    centres = np.asarray(predicted_planes, dtype=float)
    intervals = np.asarray(reference_intervals, dtype=float)
    if centres.ndim != 1 or not centres.size or intervals.shape != (centres.size, 2):
        raise ValueError("Expected n centres and n reference intervals")
    if not np.isfinite(centres).all() or not np.isfinite(intervals).all() or (intervals[:, 0] > intervals[:, 1]).any():
        raise ValueError("Centres and ordered intervals must be finite")
    # Score includes the complete annotation interval, not only its best plane.
    scores = np.max(np.abs(intervals - centres[:, None]), axis=1)
    rank = math.ceil((centres.size + 1) * (1 - alpha))
    if rank > centres.size:
        raise ValueError("Too few calibration cases for this requested coverage")
    return float(np.sort(scores)[rank - 1])


class FoldIntervalTeacher:
    """Portable fixed linear ranker over anatomically motivated descriptors."""

    def fit(self, names: list[str], cases: dict[str, LoadedCase]) -> "FoldIntervalTeacher":
        self.position_statistics = _training_position_statistics(names, cases)
        matrices, targets, weights = [], [], []
        for name in names:
            matrix, self.feature_names = _candidate_matrix(cases[name], MODE, self.position_statistics)
            matrices.append(matrix)
            targets.extend(int(c == cases[name].geometry.target_cut) for c in cases[name].geometry.candidates)
            weights.extend([1 / len(matrix)] * len(matrix))
        x = np.concatenate(matrices)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=.20, class_weight="balanced", max_iter=5000)
        model.fit(scaler.transform(x), targets, sample_weight=weights)
        self.mean, self.scale = scaler.mean_, scaler.scale_
        self.coef, self.intercept = model.coef_[0], float(model.intercept_[0])
        return self

    def predict_cut(self, case: LoadedCase) -> int:
        matrix, names = _candidate_matrix(case, MODE, self.position_statistics)
        if names != self.feature_names:
            raise ValueError("Descriptor schema mismatch")
        scores = ((matrix - self.mean) / self.scale) @ self.coef + self.intercept
        return int(case.geometry.candidates[int(np.argmax(scores))])

    def ground_native(self, image: np.ndarray, foreground: np.ndarray,
                      affine: np.ndarray, *, radius_mm: float) -> dict[str, np.ndarray]:
        """Create label-free anchors in a native 1-mm RAS crop.

        Coordinates use the crop's local physical frame (zero at voxel zero),
        not MNI or scanner-world y. The full field must follow spatial image
        augmentation. Empty/discontinuous support raises rather than inventing
        a landmark. The caller must explicitly skip such samples.
        """
        from .audit_predicted_foreground import largest_foreground_component

        image, foreground, affine = np.asarray(image), np.asarray(foreground), np.asarray(affine)
        if image.ndim != 3 or image.shape != foreground.shape or foreground.dtype != bool:
            raise ValueError("Expected matching native 3-D image and Boolean foreground")
        if not np.isfinite(image).all():
            raise ValueError("Image must contain finite intensities")
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[:3, :3], np.eye(3)):
            raise ValueError("Teacher currently requires axis-aligned 1-mm RAS")
        if not math.isfinite(radius_mm) or radius_mm < 0:
            raise ValueError("radius_mm must be finite and nonnegative")
        support, _, _ = largest_foreground_component(foreground)
        occupied = np.flatnonzero(support.any(axis=(0, 2)))
        if len(occupied) < 3 or np.any(np.diff(occupied) != 1):
            raise ValueError("Foreground must span at least three contiguous A/P slices")
        low, high = int(occupied[0]), int(occupied[-1])
        geometry = CaseFeatures("inference", low, high, -1, 0, 0, {})
        case = LoadedCase(geometry, extract_refined_case_features(image, support, low, high))
        plane = self.predict_cut(case) - .5
        coordinates = np.broadcast_to(np.arange(image.shape[1], dtype=np.float32)[None, :, None], image.shape).copy()
        return {"support": support, "anterior_coordinate_mm": coordinates,
                "interval_mm": np.asarray([plane-radius_mm, plane+radius_mm], dtype=np.float32)}

    def as_dict(self) -> dict:
        return {"mode": MODE, "position_statistics": list(self.position_statistics),
                "feature_names": self.feature_names, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, data: dict) -> "FoldIntervalTeacher":
        if data["mode"] != MODE:
            raise ValueError("Unexpected teacher mode")
        teacher = cls()
        teacher.position_statistics = tuple(data["position_statistics"])
        teacher.feature_names = list(data["feature_names"])
        for field in ("mean", "scale", "coef"):
            setattr(teacher, field, np.asarray(data[field], dtype=float))
        teacher.intercept = float(data["intercept"])
        return teacher
