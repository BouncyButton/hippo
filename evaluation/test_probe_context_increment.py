"""Checks for the feature/target boundary and the small cut-ranking probe."""
import numpy as np
import pytest

from evaluation.probe_context_increment import (
    ARMS, context_maps, design, fit, paired_interval, predict, prediction_features,
)


def make_case(name="case", target=3):
    candidates = np.arange(1, 6)
    return {"name": np.asarray(name), "candidates": candidates, "target": np.asarray(target),
            "base": np.column_stack((candidates, -(candidates - 3.)**2)),
            "mri": np.arange(15.).reshape(5, 3), "decoder": np.ones((5, 4))}


@pytest.mark.parametrize("arm", ARMS)
def test_targets_and_metrics_never_enter_design(arm):
    case = make_case()
    expected = design(case, arm)
    case.update(target=np.asarray(1), cut_dice=np.arange(5.), original_dice=np.asarray(.2))
    np.testing.assert_array_equal(design(case, arm), expected)


def test_shuffled_control_is_reproducible_and_preserves_rows():
    case = make_case()
    shuffled = design(case, "base_shuffled_mri")
    np.testing.assert_array_equal(shuffled, design(case, "base_shuffled_mri"))
    np.testing.assert_array_equal(shuffled[:, :2], case["base"])
    assert set(map(tuple, shuffled[:, 2:])) == set(map(tuple, case["mri"]))
    assert not np.array_equal(shuffled[:, 2:], case["mri"])


def test_conditional_likelihood_prefers_true_transition_and_is_shift_invariant():
    union = np.zeros((12, 12, 12), dtype=bool)
    union[3:9, 2:10, 3:9] = True
    prediction = np.where(union, np.where(np.arange(12)[None, :, None] >= 6, 1, 2), 0)
    logits = np.zeros((3, 12, 12, 12), dtype=float)
    logits[1, :, 6:, :] = 5
    logits[2, :, :6, :] = 5
    candidates = np.arange(3, 10)
    features = prediction_features(logits, prediction, union, candidates, 6)
    assert candidates[features[:, 6].argmax()] == 6
    np.testing.assert_allclose(features, prediction_features(logits + 10, prediction, union, candidates, 6))


def test_constant_mri_is_finite_and_normalization_is_affine_invariant():
    union = np.ones((12, 12, 12), dtype=bool)
    assert np.isfinite(context_maps(np.ones(union.shape), union)).all()
    image = np.random.default_rng(2).normal(size=union.shape)
    np.testing.assert_allclose(context_maps(image, union), context_maps(3 * image + 7, union), atol=1e-5)


def test_ranker_recovers_known_candidate_and_rejects_unavailable_target():
    cases = [make_case(str(i)) for i in range(12)]
    estimator = fit(cases, "base", 1.)
    assert predict(estimator, make_case("held_out"), "base") == 3
    with pytest.raises(ValueError, match="outside candidate"):
        fit([make_case(target=99)], "base", 1.)


def test_bootstrap_pairs_are_not_independent_group_samples():
    result = paired_interval(np.full(52, -.5))
    assert result == {"mean": -.5, "ci95": [-.5, -.5]}
