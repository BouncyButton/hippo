"""Known Dirichlet solutions and invariance checks for the offline solver."""

import numpy as np
import pytest

from thesis.new_constraints.harmonic_partition import (
    build_laplacian, field_labels, harmonic_partition, make_cores,
)


def chain(n=8):
    shape = (1, n, 1)
    image = np.zeros(shape)
    support = np.ones(shape, bool)
    seeds = np.full(shape, -1, np.int8)
    seeds[0, 0, 0], seeds[0, -1, 0] = 1, 0
    return image, support, seeds, np.full(shape, .5)


def test_uniform_chain_has_exact_linear_solution():
    args = chain()
    result = harmonic_partition(*args)
    np.testing.assert_allclose(result.field.ravel(), np.linspace(1, 0, 8), atol=1e-12)
    assert result.diagnostics["max_linear_residual"] < 1e-12


def test_image_jump_matches_series_resistance_solution():
    image, support, seeds, fallback = chain()
    image[:, 3:] = 1.
    beta = 4.
    result = harmonic_partition(image, support, seeds, fallback, beta=beta)
    weights = np.ones(7)
    # Three zeros and five ones give IQR=1. The jump is at edge 2.
    weights[2] = 1e-4 + (1-1e-4)*np.exp(-beta)
    expected = 1-np.r_[0, np.cumsum(1/weights)] / (1/weights).sum()
    np.testing.assert_allclose(result.field.ravel(), expected, atol=1e-11)
    assert result.field[0, 2, 0] > .5 > result.field[0, 3, 0]


def test_seed_swap_complements_field():
    image, support, seeds, fallback = chain()
    image[:, 2:] = 1
    first = harmonic_partition(image, support, seeds, fallback, beta=3)
    swapped = np.where(seeds >= 0, 1-seeds, -1)
    second = harmonic_partition(image, support, swapped, 1-fallback, beta=3)
    np.testing.assert_allclose(first.field, 1-second.field, atol=1e-11)


def test_missing_seed_components_preserve_fallback():
    image, support, seeds, fallback = chain()
    support[:, 3:5] = False
    fallback[:, :4] = .25
    fallback[:, 4:] = .75
    result = harmonic_partition(image, support, seeds, fallback, beta=1)
    np.testing.assert_array_equal(result.field, fallback)
    assert result.diagnostics["solved_components"] == 0
    assert result.diagnostics["active_seed_voxels"] == 0


def test_laplacian_and_intensity_affine_invariance():
    rng = np.random.default_rng(4)
    image = rng.normal(size=(4, 5, 6))
    support = np.ones(image.shape, bool)
    lap, _ = build_laplacian(image, support, beta=3, spacing=(1, 2, 3))
    shifted, _ = build_laplacian(7*image+20, support, beta=3, spacing=(1, 2, 3))
    np.testing.assert_allclose(lap.toarray(), lap.toarray().T, atol=1e-12)
    np.testing.assert_allclose(lap @ np.ones(support.sum()), 0, atol=1e-12)
    np.testing.assert_allclose(lap.toarray(), shifted.toarray(), atol=1e-12)


def test_cores_do_not_use_anatomical_labels_and_respect_mm_band():
    support = np.ones((5, 30, 5), bool)
    seeds = make_cores(support, mode="band", cut=15, spacing_y=2.)
    assert seeds[2, 12, 2] == 1
    assert seeds[2, 13, 2] == -1
    assert seeds[2, 16, 2] == -1
    assert seeds[2, 17, 2] == 0
    assert np.all(seeds[0] == -1)  # Seeds avoid outer boundary.
    ends = make_cores(support, mode="ends")
    assert ends[2, 3, 2] == 1 and ends[2, 26, 2] == 0


def test_field_labels_preserve_support():
    image, support, seeds, fallback = chain()
    result = harmonic_partition(image, support, seeds, fallback)
    pred = np.full(support.shape, 1, dtype=np.uint8)
    out = field_labels(result, pred)
    np.testing.assert_array_equal(out.ravel(), [2, 2, 2, 2, 1, 1, 1, 1])
    np.testing.assert_array_equal(out > 0, pred > 0)
    pred[0, 3, 0] = 0
    with pytest.raises(ValueError, match="foreground"):
        field_labels(result, pred)


def test_three_dimensional_lattice_matches_linear_field():
    shape = (4, 7, 5)
    image = np.zeros(shape)
    support = np.ones(shape, bool)
    seeds = np.full(shape, -1, dtype=np.int8)
    seeds[:, 0], seeds[:, -1] = 1, 0
    result = harmonic_partition(image, support, seeds, np.zeros(shape))
    expected = np.broadcast_to(np.linspace(1, 0, 7)[None, :, None], shape)
    np.testing.assert_allclose(result.field, expected, atol=1e-12)
    np.testing.assert_array_equal(result.field[seeds >= 0], seeds[seeds >= 0])


def test_scoring_reference_does_not_change_field():
    from evaluation.pilot_harmonic_ap import field_measurements, run_field

    prediction = np.ones((5, 24, 5), dtype=np.uint8)
    prediction[:, :12] = 2
    image = np.zeros(prediction.shape)
    field, fitted = run_field(image, prediction, mode="band", beta=1, cut=12)
    before_field, before_fit = field.field.copy(), fitted.copy()
    field_measurements(field, fitted, prediction, prediction, 12)
    wrong_reference = 3-prediction
    field_measurements(field, fitted, prediction, wrong_reference, 8)
    np.testing.assert_array_equal(field.field, before_field)
    np.testing.assert_array_equal(fitted, before_fit)


@pytest.mark.parametrize("beta", [-1, np.nan, np.inf])
def test_bad_beta_rejected(beta):
    with pytest.raises(ValueError):
        harmonic_partition(*chain(), beta=beta)


def test_bad_seed_and_empty_foreground_handling():
    image, support, seeds, fallback = chain()
    bad_seeds = seeds.copy()
    bad_seeds[0, 3, 0] = 2
    with pytest.raises(ValueError):
        harmonic_partition(image, support, bad_seeds, fallback)
    empty = np.zeros_like(support)
    with pytest.raises(ValueError):
        harmonic_partition(image, empty, seeds, fallback)
    result = harmonic_partition(image, empty, np.full_like(seeds, -1), fallback)
    np.testing.assert_array_equal(result.field, fallback)
