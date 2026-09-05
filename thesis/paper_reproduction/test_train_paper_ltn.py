#!/usr/bin/env python3
"""Focused tests for the paper-reproduction LTN volume groundings."""

from __future__ import annotations

import unittest

import torch

from train_paper_ltn import (
    PaperLTNObjective,
    build_model,
    evaluate,
    hard_masks,
    hard_volume_difference,
    paper_dimension,
    soft_volume_difference,
    soft_volume_similarity,
)


class TinySegmentator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        background = torch.zeros_like(images)
        return torch.cat(
            [background, self.scale * images, -self.scale * images],
            dim=1,
        )


class IdentityLabelSegmentator(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        labels = images[:, 0].long()
        return torch.nn.functional.one_hot(labels, num_classes=3).movedim(-1, 1).float() * 20.0


class VolumeGroundingTests(unittest.TestCase):
    def test_real_swinunetr_constructor_matches_installed_monai(self) -> None:
        model = build_model(torch.device("cpu"), (64, 64, 64))
        self.assertEqual(model.out.conv.out_channels, 3)

    def test_hard_volume_formula_matches_equation_7(self) -> None:
        hard = torch.tensor(
            [
                [[[1, 1], [1, 2]], [[0, 0], [0, 0]]],
                [[[1, 2], [2, 2]], [[0, 0], [0, 0]]],
            ]
        )
        self.assertTrue(torch.equal(hard_volume_difference(hard), torch.tensor([2.0, 2.0])))
        expected = torch.exp(torch.tensor([-0.25, -0.25]))
        actual = paper_dimension(hard, gamma=0.25, epsilon=1.0)
        self.assertTrue(torch.allclose(actual, expected))

    def test_soft_volume_formula_has_a_logit_gradient(self) -> None:
        logits = torch.zeros((1, 3, 2, 2, 2), requires_grad=True)
        with torch.no_grad():
            logits[:, 1] = 2.0
            logits[:, 2] = -2.0
        truth = soft_volume_similarity(logits, gamma=0.01, epsilon=0.0)
        truth.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_soft_volume_difference_matches_probability_sums(self) -> None:
        logits = torch.zeros((1, 3, 2, 2, 2))
        with torch.no_grad():
            logits[:, 1] = 1.0
            logits[:, 2] = -1.0
        probabilities = torch.softmax(logits, dim=1)
        expected = (probabilities[:, 1].sum() - probabilities[:, 2].sum()).abs()
        self.assertTrue(torch.allclose(soft_volume_difference(logits), expected))

    def test_hard_volume_formula_is_disconnected_from_logits(self) -> None:
        logits = torch.randn((1, 3, 2, 2, 2), requires_grad=True)
        truth = paper_dimension(hard_masks(logits), gamma=0.01, epsilon=0.0)
        self.assertFalse(truth.requires_grad)

    def test_ltn_objective_is_scalar_and_backpropagates_in_both_modes(self) -> None:
        images = torch.linspace(-1.0, 1.0, 128).reshape(2, 1, 4, 4, 4)
        labels = torch.where(images > 0, 1, 2).long()
        for grounding in ("paper-hard", "soft-probability"):
            model = TinySegmentator()
            objective = PaperLTNObjective(
                model,
                volume_epsilon=0.0,
                volume_gamma=0.01,
                volume_grounding=grounding,
            )
            _, loss, diagnostics = objective(images, labels)
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertEqual(
                set(diagnostics),
                {
                    "constraint_satisfaction",
                    "dice_truth",
                    "connectedness_truth",
                    "volume_truth",
                    "nesting_truth",
                    "hard_volume_gap_voxels",
                    "soft_volume_gap_voxels",
                },
            )
            self.assertTrue(torch.allclose(loss, 1.0 - diagnostics["constraint_satisfaction"]))
            for name in (
                "constraint_satisfaction",
                "dice_truth",
                "connectedness_truth",
                "volume_truth",
                "nesting_truth",
            ):
                self.assertGreaterEqual(float(diagnostics[name].detach()), 0.0)
                self.assertLessEqual(float(diagnostics[name].detach()), 1.0)
            self.assertGreaterEqual(float(diagnostics["hard_volume_gap_voxels"]), 0.0)
            self.assertGreaterEqual(float(diagnostics["soft_volume_gap_voxels"]), 0.0)
            loss.backward()
            self.assertIsNotNone(model.scale.grad)
            self.assertGreater(float(model.scale.grad.abs()), 0.0)

    def test_evaluate_reports_each_segmentation_class(self) -> None:
        labels = torch.tensor(
            [
                [[[[0, 0], [1, 1]], [[2, 2], [0, 0]]]],
                [[[[2, 2], [1, 1]], [[0, 0], [2, 1]]]],
            ]
        )
        dataset = [{"image": label.float(), "label": label} for label in labels]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        metrics = evaluate(IdentityLabelSegmentator(), loader, torch.device("cpu"))
        for name in (
            "dice_all_classes",
            "dice_foreground",
            "dice_background",
            "dice_anterior",
            "dice_posterior",
        ):
            self.assertAlmostEqual(metrics[name], 1.0)


if __name__ == "__main__":
    unittest.main()
