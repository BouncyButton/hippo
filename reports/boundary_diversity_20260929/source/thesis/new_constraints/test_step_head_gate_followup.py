"""Small checks for the patient-grouped contour trust gate."""

import numpy as np

from thesis.new_constraints.step_head_gate_followup import Case, choose_gate, gated_mask


def synthetic_case(index: int, head_wins: bool) -> Case:
    truth = np.ones((1, 1, 4), dtype=np.uint8)
    seg = np.zeros_like(truth) if head_wins else truth.copy()
    head = truth.copy() if head_wins else np.zeros_like(truth)
    features = np.zeros((1, 15), dtype=np.float32)
    features[0, 0] = 0.95 if head_wins else 0.05
    return Case(
        str(index), truth, seg, head, np.ones((1, 1), dtype=bool),
        features, np.array([4 if head_wins else -4]),
        np.array([head_wins]), np.array([not head_wins]),
    )


def test_gate_learns_reliability_from_grouped_patients():
    cases = [synthetic_case(i, i % 4 != 0) for i in range(42)]
    gate, model = choose_gate(cases)
    assert gate["enabled"] and gate["cv_gain"] > 0
    assert model is not None
    for case in cases:
        accept = model.predict_proba(case.features)[:, 1] >= gate["threshold"]
        chosen = gated_mask(case, accept)
        assert np.array_equal(chosen, case.truth)


def test_gate_defaults_to_segmentation_when_head_never_helps():
    cases = [synthetic_case(i, False) for i in range(42)]
    gate, model = choose_gate(cases)
    assert not gate["enabled"] and model is None
    for case in cases:
        assert np.array_equal(gated_mask(case, np.zeros(1, dtype=bool)), case.seg)
