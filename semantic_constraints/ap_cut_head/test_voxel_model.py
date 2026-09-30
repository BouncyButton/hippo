"""Tests for the literal cut-voxel auxiliary target and decoder."""

import torch

from semantic_constraints.ap_cut_head.voxel_model import (
    candidate_cut_scores,
    cut_band_target,
    cut_voxel_loss,
)


def test_cut_band_is_foreground_only_on_both_adjacent_slices() -> None:
    foreground = torch.ones(1, 2, 6, 2)
    foreground[:, 0, 2, 0] = 0
    target = cut_band_target(foreground, torch.tensor([2]))
    assert target.sum() == 7
    assert target[:, :, 1].sum() == 0
    assert target[:, :, 4].sum() == 0
    assert target[0, 0, 2, 0] == 0


def test_voxel_loss_and_pair_scores_reward_true_cut() -> None:
    foreground = torch.ones(1, 2, 6, 2)
    target = cut_band_target(foreground, torch.tensor([2]))
    logits = ((target * 2 - 1) * 8).requires_grad_()
    loss = cut_voxel_loss(logits, foreground, target)
    assert loss < 0.01
    scores = candidate_cut_scores(logits, foreground)
    assert scores.shape == (1, 5)
    assert int(scores.argmax()) == 2
    loss.backward()
    assert logits.grad is not None
