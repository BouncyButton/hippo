"""Geometry and gradient checks for the four-arm sagittal pilot."""

import torch

from thesis.new_constraints.step_head_pilot import (
    ContourHead,
    decoded_edges,
    head_loss,
    same_run_pairs,
    targets,
)


def test_targets_mark_empty_columns_without_edges():
    labels = torch.zeros(1, 1, 2, 5, 8, dtype=torch.long)
    labels[0, 0, 0, 0, 2:5] = 1
    labels[0, 0, 0, 3, 1:7] = 2
    present, lower, upper = targets(labels)
    assert present[0, 0].tolist() == [True, False, False, True, False]
    assert lower[0, 0].tolist() == [2, -1, -1, 1, -1]
    assert upper[0, 0].tolist() == [4, -1, -1, 6, -1]
    pair = same_run_pairs(present)
    assert not pair[0, 0, 0, 3]


def test_head_orders_edges_and_reaches_shared_features():
    features = torch.randn(1, 24, 2, 5, 8, requires_grad=True)
    head = ContourHead()
    output = head(features)
    labels = torch.zeros(1, 1, 2, 5, 8, dtype=torch.long)
    labels[0, 0, 0, 0, 3:5] = 1
    labels[0, 0, 0, 1, 4:7] = 1
    labels[0, 0, 0, 3, 1:3] = 1
    loss, details = head_loss(output, labels, True)
    lo, hi = decoded_edges(output[1])
    assert torch.all(lo <= hi)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in details.values())
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
