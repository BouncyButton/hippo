"""Analytic and numerical falsifiers for the proposed depth-profile loss."""
import unittest

import torch

from .losses import RayGeometry, grouped_masks, profile_discrepancy, wrong_foreground_bce


class TestTomography(unittest.TestCase):
    def test_same_mass_shift_visible_to_depth(self):
        g = torch.zeros(1, 1, 12, dtype=torch.double)
        g[..., 2:5] = 1
        p = torch.zeros_like(g)
        p[..., 5:8] = 1
        p.requires_grad_()
        self.assertEqual(float(profile_discrepancy(p - g, 1, 12, "thickness")), 0)
        loss = profile_discrepancy(p - g, 1, 12, "depth")
        self.assertGreater(float(loss), 0)
        grad, = torch.autograd.grad(loss, p)
        self.assertTrue(bool((grad[..., 2:5] < 0).all()))
        self.assertTrue(bool((grad[..., 5:8] > 0).all()))

    def test_equal_first_two_moments_not_equal_profile(self):
        # Occupancies {1,6} and {2,5} have identical mass and first moment.
        g = torch.zeros(8, dtype=torch.double)
        p = torch.zeros_like(g)
        g[[1, 6]], p[[2, 5]] = 1, 1
        r = p - g
        self.assertEqual(float(r.sum()), 0)
        self.assertEqual(float((r * torch.arange(8)).sum()), 0)
        self.assertGreater(float(profile_discrepancy(r, 1, 8, "depth")), 0)

    def test_reverse_invariance(self):
        r = torch.tensor([.1, -.7, .3, .4], dtype=torch.double)
        torch.testing.assert_close(profile_discrepancy(r, 1, 4, "depth"),
                                   profile_discrepancy(r.flip(-1), 1, 4, "depth"))

    def test_perfect_masks_zero_loss_and_gradient(self):
        torch.manual_seed(0)
        p = torch.rand(1, 3, 5, 5, 5, dtype=torch.double, requires_grad=True)
        geometry = RayGeometry((5, 5, 5), dtype=torch.double)
        for mode in ("thickness", "depth"):
            loss = geometry.loss(p, p.detach(), mode)
            grad, = torch.autograd.grad(loss, p)
            self.assertEqual(float(loss), 0)
            self.assertEqual(float(grad.abs().max()), 0)

    def test_axis_mass_matches_direct_sum_and_physical_spacing(self):
        p = torch.zeros(1, 1, 4, 6, 8, dtype=torch.double)
        p[..., 1, 2, 3] = 1
        geometry = RayGeometry((4, 6, 8), spacing=(2, 1, .5), oblique=False)
        expected = sum(float(p.sum(dim=a+2).abs().mean()) * geometry.spacing[a] / 8
                       for a in range(3)) / 3
        self.assertAlmostEqual(float(geometry.loss(p, torch.zeros_like(p), "thickness")), expected)

    def test_depth_length_normalization_is_scale_invariant(self):
        r = torch.tensor([.2, -.3, .4])
        torch.testing.assert_close(profile_discrepancy(r, 1, 3, "depth"),
                                   profile_discrepancy(r, 2, 6, "depth"))

    def test_oblique_coordinate_interpolation(self):
        geometry = RayGeometry((5, 5, 5), dtype=torch.double)
        axes = torch.linspace(-2, 2, 5, dtype=torch.double)
        d, h, w = torch.meshgrid(axes, axes, axes, indexing="ij")
        ramp = (d + 2 * h + 3 * w)[None, None]
        grid = geometry.grids[0]
        sample = torch.nn.functional.grid_sample(ramp, grid, align_corners=True)[0, 0]
        interior = (grid[0].abs() < .9).all(-1)
        expected = 2 * (grid[0, ..., 2] + 2 * grid[0, ..., 1] + 3 * grid[0, ..., 0])
        self.assertGreater(int(interior.sum()), 0)
        torch.testing.assert_close(sample[interior], expected[interior])

    def test_oblique_directional_derivative(self):
        torch.manual_seed(7)
        p = torch.rand(1, 1, 5, 5, 5, dtype=torch.double, requires_grad=True)
        target = torch.zeros_like(p)
        direction = torch.randn_like(p)
        geometry = RayGeometry((5, 5, 5), dtype=torch.double)
        for mode in ("thickness", "depth"):
            grad, = torch.autograd.grad(geometry.loss(p, target, mode), p)
            eps = 1e-6
            finite = (geometry.loss(p + eps * direction, target, mode)
                      - geometry.loss(p - eps * direction, target, mode)) / (2 * eps)
            torch.testing.assert_close((grad * direction).sum(), finite, atol=1e-9, rtol=1e-6)

    def test_semantic_swap_union_blind_classes_visible(self):
        y = torch.zeros(1, 6, 6, 6, dtype=torch.long)
        y[:, 1:3, 1:5, 1:5] = 1
        y[:, 3:5, 1:5, 1:5] = 2
        p = torch.nn.functional.one_hot(y, 3).movedim(-1, 1).float()[:, [0, 2, 1]]
        pred, target = grouped_masks(p, y)
        geometry = RayGeometry((6, 6, 6), oblique=False)
        self.assertEqual(float(geometry.loss(pred[:, :1], target[:, :1], "depth")), 0)
        self.assertGreater(float(geometry.loss(pred, target, "depth")), 0)

    def test_finite_logit_gradient_at_high_confidence(self):
        z = torch.zeros(1, 3, 6, 6, 6)
        z[:, 0] = 30
        z.requires_grad_()
        y = torch.ones(1, 6, 6, 6, dtype=torch.long)
        p, g = grouped_masks(z.softmax(1), y)
        loss = RayGeometry((6, 6, 6), oblique=False).loss(p, g, "depth")
        grad, = torch.autograd.grad(loss, z)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().max()), 0)

    def test_wrong_only_control_preserves_correct_logit_gradients(self):
        z = torch.zeros(1, 3, 4, 4, 4, requires_grad=True)
        y = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        y[:, 2, 2, 2] = 1
        grad, = torch.autograd.grad(wrong_foreground_bce(z, y), z)
        self.assertEqual(float(grad.movedim(1, -1)[y == 0].abs().max()), 0)
        self.assertLess(float(grad[0, 1, 2, 2, 2]), 0)


if __name__ == "__main__":
    unittest.main()
