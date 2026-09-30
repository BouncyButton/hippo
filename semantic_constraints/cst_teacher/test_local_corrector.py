"""Shape, zero-start, and acceptance-policy checks for local correction."""

import numpy as np
import torch

from .local_corrector import LocalResidualCorrector, apply_slice_policy, correction_loss, make_slice_contexts
from .run_local_correction_study import select_policy


def test_contexts_and_zero_start():
    image = np.random.default_rng(1).normal(size=(8, 8, 8)).astype(np.float32)
    probabilities = np.zeros((3, 8, 8, 8), dtype=np.float32)
    probabilities[0] = 0.6
    probabilities[1] = 0.3
    probabilities[2] = 0.1
    positions = np.array([0, 3, 7])
    profile = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    context = make_slice_contexts(image, probabilities, profile, positions, use_cst=True)
    assert context.shape == (8, 23, 8, 8)
    assert np.allclose(context[0, 20], 0.1)
    assert np.allclose(context[7, 21], 0.6, atol=1e-3)
    model = LocalResidualCorrector()
    logits, residual = model(torch.from_numpy(context[:2].astype(np.float32)))
    assert torch.allclose(residual, torch.zeros_like(residual))
    assert torch.all(logits.argmax(dim=1) == 0)
    target = torch.ones((2, 8, 8), dtype=torch.long)
    loss = correction_loss(logits, residual, torch.from_numpy(context[:2, 11:14].astype(np.float32)), target)
    assert torch.isfinite(loss)


def test_policy():
    baseline = np.zeros((2, 5, 2), dtype=np.int8)
    proposal = np.ones_like(baseline)
    risk = np.array([0, 5, 1, 4, 2], dtype=float)
    assert np.array_equal(apply_slice_policy(baseline, proposal, risk, 0), baseline)
    assert np.array_equal(apply_slice_policy(baseline, proposal, risk, 1), proposal)
    gated = apply_slice_policy(baseline, proposal, risk, 0.4)
    assert np.array_equal(np.flatnonzero(gated.sum(axis=(0, 2))), np.array([1, 3]))


def test_inner_policy_rejects_harmful_edit():
    gains = {
        0.0: np.zeros(8),
        0.2: np.array([0.01, 0.01, 0.01, 0.01, 0.01, -0.001, 0, 0]),
        0.4: np.array([0.02, 0.02, 0.02, -0.01, -0.01, -0.01, 0, 0]),
        1.0: np.array([-0.02] * 8),
    }
    assert select_policy(gains) == 0.2
    gains[0.2] = np.array([-0.01] * 8)
    assert select_policy(gains) == 0.0
