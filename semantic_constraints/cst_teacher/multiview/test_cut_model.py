"""Geometry, view-sampling, and model-shape tests for matched CST cut probes."""

import numpy as np
import torch

from .cut_model import CSTCutRegressor, VIEW_SPECS, best_cut_from_labels, make_view_profiles, make_view_set


def test_best_cut_from_labels():
    label = np.zeros((5, 10, 6), dtype=np.int8)
    label[:, 2:5, :] = 2
    label[:, 5:8, :] = 1
    assert best_cut_from_labels(label) == (4, True, 0)


def test_all_views_and_shared_model_shape():
    image = torch.randn(1, 16, 18, 20)
    model = CSTCutRegressor(channels=(4, 8), heads=2)
    for specification in VIEW_SPECS.values():
        adjusted = {key: min(value, image.shape[{"sagittal": 1, "coronal": 2, "axial": 3}[key]]) for key, value in specification.items()}
        slabs, metadata = make_view_set(image, adjusted, output_size=16)
        assert slabs.shape == (sum(adjusted.values()), 3, 16, 16)
        assert metadata.shape == (sum(adjusted.values()), 4)
        prediction = model(slabs.unsqueeze(0), metadata.unsqueeze(0))
        assert prediction.shape == (1,)
        assert 0 <= float(prediction[0]) <= 1


def test_triview_profile_targets_normalize_by_view():
    label = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    label[:, 1:7, 2:4, 1:7] = 2
    label[:, 1:7, 4:6, 1:7] = 1
    specification = {"sagittal": 4, "coronal": 4, "axial": 4}
    profiles = make_view_profiles(label, specification)
    assert profiles.shape == (12, 2)
    assert profiles.min() >= 0
    assert profiles.max() <= 1
    for start in (0, 4, 8):
        assert torch.isclose(profiles[start:start + 4].sum(dim=1).max(), torch.tensor(1.0))
