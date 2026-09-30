"""Fixed cross-fold ridge smoke test."""

import numpy as np

from .train_crossfold_boundary_risk import fixed_ridge_predict


def test_fixed_ridge_has_finite_predictions():
    x = np.arange(18, dtype=np.float64).reshape(6, 3)
    y = np.arange(1, 7, dtype=np.float64)
    predicted = fixed_ridge_predict(x, y, x[:2])
    assert predicted.shape == (2,)
    assert np.isfinite(predicted).all()
    assert (predicted >= 0).all()
