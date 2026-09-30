"""Geometry, model, and soft-ordering checks for the auxiliary cut head."""

import numpy as np
import torch

from semantic_constraints.ap_cut_head.logic import ap_cut_consistency_loss
from semantic_constraints.ap_cut_head.model import APCutHead, soft_cut_targets
from semantic_constraints.cst_teacher.multiview.cut_model import best_cut_from_labels


def test_cut_target_is_between_slices() -> None:
    label = np.zeros((4, 6, 4), dtype=np.int8)
    label[:, 1:3, :] = 2
    label[:, 3:5, :] = 1
    cut, high, wrong = best_cut_from_labels(label)
    assert (cut, high, wrong) == (2, True, 0)


def test_soft_target_gives_adjacent_cuts_partial_credit() -> None:
    probabilities = soft_cut_targets(torch.tensor([3]), 8, sigma=0.75)[0]
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[3] > probabilities[2] > probabilities[1]
    assert torch.isclose(probabilities[2], probabilities[4])


def test_head_outputs_one_score_per_between_slice_cut() -> None:
    torch.manual_seed(1)
    image = torch.randn(1, 1, 16, 16, 16)
    probabilities = torch.rand(1, 3, 16, 16, 16).softmax(dim=1)
    cut = torch.tensor([7])
    head = APCutHead(image_enabled=False).eval()
    with torch.no_grad():
        first = head(image, probabilities, cut)
        second = head(torch.zeros_like(image), probabilities, cut)
    assert first.shape == (1, 15)
    assert torch.allclose(first, second)


def test_soft_cut_logic_prefers_correct_ap_order_and_detaches_cut() -> None:
    # A cut between Y=1 and Y=2 means posterior at Y=0,1 and anterior at Y=2,3.
    correct = torch.full((1, 3, 1, 4, 1), -8.0, requires_grad=True)
    with torch.no_grad():
        correct[0, 2, 0, :2, 0] = 8
        correct[0, 1, 0, 2:, 0] = 8
    cut_logits = torch.tensor([[-10.0, 10.0, -10.0]], requires_grad=True)
    support = torch.ones((1, 1, 4, 1))
    good = ap_cut_consistency_loss(correct, cut_logits, support)
    swapped = ap_cut_consistency_loss(correct[:, [0, 2, 1]], cut_logits, support)
    assert good < 0.01
    assert swapped > good + 5
    good.backward()
    assert cut_logits.grad is None
    assert correct.grad is not None
