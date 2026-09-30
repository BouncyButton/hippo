"""Independent transform, graph-energy, and split checks for the atlas pilot."""
import itertools
import json

import numpy as np
import pytest

from evaluation.atlas_registration import (
    REPORT, itk_image, array, transfer_labels, compose, sitk, normalize,
)
from evaluation.atlas_pilot import (
    lattice_edges, binary_cut, whole_refine, plane_from_margin, ap_refine, measure,
)


def test_identity_transfer_and_native_axis_order():
    labels = np.zeros((7, 9, 11), np.uint8)
    labels[1:4, 2:5, 3:7] = 2
    labels[2:5, 5:8, 2:6] = 1
    image = itk_image(labels)
    np.testing.assert_array_equal(array(image), labels)
    result = transfer_labels(labels, image, sitk.Transform(3, sitk.sitkIdentity))
    np.testing.assert_array_equal(result.argmax(0), labels)
    np.testing.assert_allclose(result.sum(0), 1)


def test_known_translation_and_outside_background():
    labels = np.zeros((7, 8, 9), np.uint8)
    labels[3, 4, 5] = 1
    transform = sitk.TranslationTransform(3, (1., 0., 0.))
    result = transfer_labels(labels, itk_image(labels), transform)
    assert result[1, 2, 4, 5] == 1  # fixed-to-moving mapping x+1 samples source x=3
    np.testing.assert_array_equal(result[0, -1], 1)
    np.testing.assert_allclose(result.sum(0), 1)


def test_composition_order_and_optimized_residual_used():
    affine = sitk.AffineTransform(3)
    affine.SetMatrix((2., 0., 0., 0., 1., 0., 0., 0., 1.))
    residual = sitk.TranslationTransform(3)
    stale = compose(affine, residual)
    residual.SetOffset((1., 0., 0.))
    updated = compose(affine, residual)
    assert stale.TransformPoint((0., 0., 0.)) == (0., 0., 0.)
    assert updated.TransformPoint((0., 0., 0.)) == (2., 0., 0.)


def test_fractional_transfer_retains_soft_probabilities():
    labels = np.zeros((5, 5, 5), np.uint8)
    labels[2, 2, 2] = 2
    transform = sitk.TranslationTransform(3, (.5, 0., 0.))
    result = transfer_labels(labels, itk_image(labels), transform)
    assert result[2, 1, 2, 2] == result[2, 2, 2, 2] == .5
    np.testing.assert_allclose(result.sum(0), 1)


def test_graph_solver_matches_exhaustive_float_energy():
    rng = np.random.default_rng(245)
    for _ in range(40):
        image = rng.random((2, 2, 2))
        edges, weights = lattice_edges(image)
        margin = rng.normal(size=image.shape)
        lam = .3
        result = binary_cut(margin, edges, weights, lam).ravel()
        def energy(labels):
            return float(-np.dot(margin.ravel(), labels) + lam * weights[labels[edges[:, 0]] != labels[edges[:, 1]]].sum())
        optimum = min(energy(np.array(v)) for v in itertools.product((False, True), repeat=8))
        # Quantization may choose another optimum within this conservative bound.
        assert energy(result) - optimum <= (len(result) + len(edges)) / 10000 + 1e-10


def test_whole_refinement_identity_and_support_can_expand():
    logits = np.zeros((3, 2, 3, 2))
    logits[0] = 1
    prior = np.zeros_like(logits)
    prior[1] = .99
    prior[0] = .01
    edges, weights = lattice_edges(np.zeros(logits.shape[1:]))
    raw = logits.argmax(0)
    np.testing.assert_array_equal(whole_refine(logits, prior, edges, weights, 0, 0), raw)
    assert np.all(whole_refine(logits, prior, edges, weights, 1, 0) == 1)
    assert not binary_cut(np.zeros((2, 2, 2)), np.empty((0, 2), int), np.empty(0), 0).any()


def test_plane_solver_matches_direct_energy_and_preserves_support():
    rng = np.random.default_rng(1)
    for _ in range(20):
        margin = rng.normal(size=(3, 7, 3))
        support = rng.random(margin.shape) > .3
        pred, cut = plane_from_margin(margin, support)
        energies = [-(margin * support * (np.arange(7)[None, :, None] >= c)).sum() for c in range(1, 7)]
        assert cut == np.argmin(energies) + 1
        np.testing.assert_array_equal(pred > 0, support)


def test_ap_only_preserves_whole_foreground():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(3, 3, 8, 3))
    prior = np.full(logits.shape, 1/3)
    for a in (0, .1, 3):
        pred, _ = ap_refine(logits, prior, a)
        np.testing.assert_array_equal(pred > 0, logits.argmax(0) > 0)


def test_cohorts_disjoint_and_no_development_atlas():
    cohort = json.loads((REPORT / "cohort.json").read_text())
    groups = [cohort[k] for k in ("atlas_bank", "calibration", "assessment", "development", "unused_training")]
    assert list(map(len, groups)) == [16, 24, 24, 52, 144]
    assert len(set(sum(groups, []))) == 260


def test_perfect_surface_and_corrected_introduced_accounting():
    labels = np.zeros((5, 8, 5), np.uint8)
    labels[1:4, 1:4, 1:4] = 2
    labels[1:4, 4:7, 1:4] = 1
    metrics = measure(labels, labels, labels)
    assert metrics["foreground_dice"] == metrics["ap_dice"] == 1
    assert metrics["assd_mm"] == metrics["hd95_mm"] == metrics["cut_error_mm"] == 0
    wrong = labels.copy()
    wrong[2, 4, 2] = 2
    metrics = measure(labels, labels, wrong)
    assert metrics["corrected_voxels"] == 1 and metrics["introduced_voxels"] == 0


def test_normalization_handles_constant_image():
    assert np.isfinite(normalize(np.ones((3, 3, 3)))).all()
    with pytest.raises(ValueError):
        binary_cut(np.ones(1), np.empty((0, 2), int), np.empty(0), -1)
