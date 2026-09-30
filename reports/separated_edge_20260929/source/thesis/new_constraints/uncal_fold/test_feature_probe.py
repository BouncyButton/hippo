import numpy as np
import torch

from .feature_probe import (
    REGION_NAMES,
    ProbeCase,
    cut_metrics,
    pool_layer_sequence,
    position_matrix,
)


def test_pool_layer_sequence_has_one_row_per_candidate() -> None:
    activation = torch.arange(3 * 6 * 4 * 5, dtype=torch.float32).reshape(3, 6, 4, 5)
    union = np.zeros((12, 8, 10), dtype=bool)
    union[3:9, 1:7, 2:8] = True
    matrix, names = pool_layer_sequence(
        activation,
        union,
        low=1,
        high=6,
        layer_name="decoder",
    )
    assert matrix.shape == (5, 3 * len(REGION_NAMES) * 4)
    assert len(names) == matrix.shape[1]
    assert np.isfinite(matrix).all()


def test_pool_layer_sequence_preserves_thin_downsampled_endpoint() -> None:
    activation = torch.ones((2, 8, 4, 8), dtype=torch.float32)
    union = np.zeros((64, 8, 64), dtype=bool)
    union[1, 1, 1] = True
    union[20:40, 2:7, 20:40] = True
    matrix, _ = pool_layer_sequence(
        activation,
        union,
        low=1,
        high=6,
        layer_name="decoder",
    )
    assert matrix.shape[0] == 5


def test_position_and_cut_metrics() -> None:
    case = ProbeCase(
        name="case",
        candidates=np.arange(3, 8),
        target=5,
        low=2,
        high=7,
        layer_features={},
        feature_names={},
    )
    matrix = position_matrix(case, (0.5, 0.1))
    assert matrix.shape == (5, 3)
    assert int(case.candidates[np.argmax(matrix[:, 2])]) in (4, 5)
    metrics = cut_metrics({"case": 6}, [case])
    assert metrics["mae_slices"] == 1.0
    assert metrics["within_1_fraction"] == 1.0
