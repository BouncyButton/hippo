"""Independent energy checks and feature/target separation for the graph pilot."""
import itertools

import numpy as np
import pytest

from evaluation.graph_partition_probe import (
    ARMS, design, fit, graph_cut, interface_features, paired, synthetic_base,
)


def energy(labels, margin, lam):
    value = np.where(labels == 1, np.logaddexp(0, -margin), np.logaddexp(0, margin)).sum()
    for axis in range(3):
        value += lam * np.count_nonzero(np.diff(labels, axis=axis))
    return value


@pytest.mark.parametrize("lam", [0., .05, .2, 1.])
def test_cut_matches_exhaustive_binary_energy(lam):
    margin = np.random.default_rng(91).normal(size=(2, 2, 2))
    initial = np.where(margin >= 0, 1, 2).astype(np.uint8)
    result, info = graph_cut(margin, initial, lam)
    optimum = min(energy(np.array(z).reshape(margin.shape), margin, lam)
                  for z in itertools.product((1, 2), repeat=8))
    assert energy(result, margin, lam) <= optimum + info["float_optimality_gap_bound"] + 1e-10
    assert energy(result, margin, lam) == pytest.approx(info["energy"])


def test_disconnected_foreground_is_preserved_and_unaries_orient_labels():
    initial = np.zeros((5, 5, 5), dtype=np.uint8)
    initial[1, 1, 1], initial[3, 3, 3] = 1, 2
    margin = np.zeros_like(initial, dtype=float)
    margin[1, 1, 1], margin[3, 3, 3] = 30., -30.
    result, info = graph_cut(margin, initial, 1.)
    np.testing.assert_array_equal(result, initial)
    assert info["edges"] == 0


def test_cut_can_remove_a_weak_island_but_respects_strong_evidence():
    margin = np.full((3, 3, 3), 10.)
    margin[1, 1, 1] = -.1
    pred = np.where(margin >= 0, 1, 2).astype(np.uint8)
    result, _ = graph_cut(margin, pred, .2)
    assert (result == 1).all()
    margin[1, 1, 1] = -10.
    assert graph_cut(margin, pred, .2)[0][1, 1, 1] == 2


def test_features_ignore_class_assignments_and_respect_mri_affine_scaling():
    shape = (14, 14, 14)
    support = np.zeros(shape, dtype=bool)
    support[3:11, 2:12, 3:11] = True
    image = np.random.default_rng(6).normal(size=shape)
    cuts = np.arange(3, 12)
    g, m = interface_features(image, support, cuts)
    g2, m2 = interface_features(3 * image + 8, support, cuts)
    np.testing.assert_array_equal(g, g2)
    np.testing.assert_allclose(m, m2, atol=2e-5)
    assert g.shape == (9, 110)
    assert m.shape == (9, 120)
    assert np.isfinite(interface_features(np.ones(shape), support, cuts)[1]).all()


def make_case(name="case"):
    cuts = np.arange(1, 7)
    base = np.column_stack((cuts, -(cuts-3.)**2))
    return {"name": np.asarray(name), "target": 3, "candidates": cuts,
            "base": base, "synthetic_base": np.stack((base, base, base)),
            "old_mri": np.ones((6, 4)), "surface": np.arange(18).reshape(6, 3),
            "edge_mri": np.arange(24).reshape(6, 4)}


@pytest.mark.parametrize("arm", ARMS)
def test_targets_and_score_tables_do_not_enter_features(arm):
    case = make_case()
    before = design(case, arm, "natural")
    case.update(target=6, cut_dice=np.arange(6.), original_dice=.1, truth=np.zeros((8, 8, 8)))
    np.testing.assert_array_equal(before, design(case, arm, "natural"))


def test_shuffle_preserves_rows_and_is_shared_across_synthetic_variants():
    case = make_case()
    original = design(case, "graph_context", "natural")[:, 2:]
    shuffled = design(case, "shuffled_graph", "natural")[:, 2:]
    assert set(map(tuple, original)) == set(map(tuple, shuffled))
    assert not np.array_equal(original, shuffled)
    np.testing.assert_array_equal(shuffled, design(case, "shuffled_graph", "displaced", 2)[:, 2:])


def test_synthetic_likelihood_tracks_intervention_not_original_classes():
    support = np.zeros((12, 12, 12), dtype=bool)
    support[3:9, 2:10, 3:9] = True
    pred = support.astype(np.uint8)
    cuts = np.arange(3, 10)
    for cut in (4, 6, 8):
        features = synthetic_base(pred, cuts, cut)
        assert cuts[features[:, 6].argmax()] == cut
        np.testing.assert_array_equal(features, synthetic_base(pred * 2, cuts, cut))


def test_readout_and_case_paired_interval():
    cases = [make_case(str(i)) for i in range(12)]
    estimator = fit(cases, "base", "displaced")
    case = make_case("held")
    assert case["candidates"][estimator.decision_function(design(case, "base", "natural")).argmax()] == 3
    assert paired([-.2] * 10)["ci95"] == pytest.approx([-.2, -.2])
    with pytest.raises(ValueError, match="Target outside"):
        fit([{**case, "target": 99}], "base", "natural")
