"""Small shape-prior checks."""

import numpy as np

from .audit_boundary_shape import area_profile, fit_profile_pca, reconstruction_error


def test_profiles_and_pca():
    label = np.zeros((8, 8, 8), dtype=np.int8)
    label[2:6, 2:4, 2:6] = 2
    label[2:6, 4:6, 2:6] = 1
    assert area_profile(label, "coronal", include_classes=False).shape == (8,)
    assert area_profile(label, "axial", include_classes=True).shape == (16,)
    examples = np.stack([area_profile(label, "coronal", include_classes=False)] * 6)
    mean, vectors = fit_profile_pca(examples, components=2)
    assert reconstruction_error(examples, mean, vectors).max() < 1e-6
