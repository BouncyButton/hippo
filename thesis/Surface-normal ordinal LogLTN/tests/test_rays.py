import numpy as np
import torch

from surface_normal_ordinal.geometry import Interface, SurfaceSamples
from surface_normal_ordinal.rays import CATEGORY_TO_CODE, analyse_rays


def _samples() -> SurfaceSamples:
    return SurfaceSamples(
        points_voxel=np.asarray([[2.0, 2.0, 2.0]]),
        normals_physical=np.asarray([[1.0, 0.0, 0.0]]),
        spacing=(1.0, 1.0, 1.0),
        interface=Interface.OUTER,
    )


def _logits(crossing: float, slope: float = 5.0) -> torch.Tensor:
    coordinates = torch.arange(5.0)[:, None, None]
    field = slope * (crossing - coordinates)
    logits = torch.full((3, 5, 5, 5), -40.0)
    logits[0] = 0.0
    logits[1] = field.expand(5, 5, 5)
    return logits


def test_correct_crossing_is_not_a_violation() -> None:
    result = analyse_rays(
        _logits(2.0),
        _samples(),
        offsets_mm=np.arange(-2.0, 2.1, 1.0),
        delta_mm=1.0,
        localization_tolerance_mm=0.6,
    )
    assert result.category_code[0] == CATEGORY_TO_CODE["correct"]
    assert not result.violation[0]


def test_steep_shifted_crossing_demonstrates_ordinal_blind_spot() -> None:
    result = analyse_rays(
        _logits(3.0),
        _samples(),
        offsets_mm=np.arange(-2.0, 2.1, 1.0),
        delta_mm=1.0,
        localization_tolerance_mm=0.5,
    )
    assert result.category_code[0] == CATEGORY_TO_CODE["shifted"]
    assert not result.violation[0]
    assert result.summary()["error_coverage"] == 0.0


def test_reversed_crossing_is_detected_and_violates() -> None:
    result = analyse_rays(
        _logits(2.0, slope=-5.0),
        _samples(),
        offsets_mm=np.arange(-2.0, 2.1, 1.0),
        delta_mm=1.0,
    )
    assert result.category_code[0] == CATEGORY_TO_CODE["reversed"]
    assert result.violation[0]
