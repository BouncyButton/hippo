"""Geometry and preservation checks; run with python -m unittest <this module>."""
import unittest

import numpy as np
import torch
from scipy.interpolate import BSpline

from .model import CoefficientHead, SplineWarp, cubic_basis, jacobian_determinant, refinement_loss
from .train import make_features, restore_labels, signed_distance


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_basis_matches_scipy_and_constant_field(self):
        b = cubic_basis(17, 4).double().numpy()
        t = np.arange(17)[:, None] / 4 - (np.arange(b.shape[1])[None] - 1)
        expected = BSpline.basis_element([-2, -1, 0, 1, 2], extrapolate=False)(t)
        np.testing.assert_allclose(b, np.nan_to_num(expected), atol=3e-8)
        geometry = SplineWarp((17, 17, 17))
        result = geometry.interpolate(torch.ones(2, 3, *geometry.control_shape))
        torch.testing.assert_close(result, torch.ones_like(result))

    def test_axis_order_and_jacobian(self):
        geometry = SplineWarp((16, 16, 16)).double()
        identity = geometry.identity
        source = identity[:, :1] + 10 * identity[:, 1:2] + 100 * identity[:, 2:3]
        displacement = torch.ones_like(identity) * .1
        shifted = geometry.sample(source, displacement)
        torch.testing.assert_close(shifted[:, :, 1:-2, 1:-2, 1:-2],
                                   source[:, :, 1:-2, 1:-2, 1:-2] + 11.1)
        affine_displacement = identity * identity.new_tensor([.1, -.2, .3])[None, :, None, None, None]
        determinant = jacobian_determinant(affine_displacement)
        torch.testing.assert_close(determinant, torch.full_like(determinant, 1.1 * .8 * 1.3))

    def test_identity_support_labels_and_training_gradient(self):
        grid = np.indices((16, 16, 16))
        mask = ((grid - 7.5) ** 2).sum(0) < 16
        labels = mask.astype(np.uint8)
        labels[mask & (grid[1] > 7)] = 2
        features = make_features(np.zeros(mask.shape), mask.astype(float), labels)
        geometry = SplineWarp(mask.shape)
        head = CoefficientHead(geometry.control_shape)
        support = features["support"][None]
        source = features["source_sdf"][None]
        coefficients = head(features["inputs"][None])
        warped, backward, velocity = geometry(coefficients, source, features["taper"][None], support)
        self.assertTrue(np.array_equal(warped[0, 0].detach().numpy() < 0, mask))
        truth = np.roll(mask, 1, axis=0)
        target = torch.from_numpy(signed_distance(truth))[None, None]
        loss, _ = refinement_loss(warped, target, torch.from_numpy(truth)[None, None],
                                  support, velocity, jacobian_determinant(backward))
        loss.backward()
        self.assertGreater(float(head.output.weight.grad.abs().max()), 0)
        random_coefficients = torch.randn_like(coefficients) * .2
        warped, _, _ = geometry(random_coefficients, source, features["taper"][None], support)
        self.assertTrue(torch.equal(warped[support == 0], source[support == 0]))
        changed = warped[0, 0].numpy() < 0
        output = restore_labels(changed, labels)
        self.assertTrue(np.array_equal(output[changed & mask], labels[changed & mask]))
        self.assertTrue(np.array_equal(output[~features["support"][0].numpy().astype(bool)],
                                       labels[~features["support"][0].numpy().astype(bool)]))

    def test_nonzero_geometry_gradient(self):
        geometry = SplineWarp((9, 9, 9)).double()
        coefficients = (torch.randn(1, 3, *geometry.control_shape, dtype=torch.double) * .07 + .12).requires_grad_()
        source = torch.randn(1, 1, 9, 9, 9, dtype=torch.double)
        support = torch.ones_like(source)

        def objective():
            warped, _, _ = geometry(coefficients, source, support, support)
            return warped[:, :, 2:-2, 2:-2, 2:-2].square().mean()

        objective().backward()
        index = np.unravel_index(coefficients.grad.abs().argmax().item(), coefficients.shape)
        analytic = float(coefficients.grad[index])
        with torch.no_grad():
            coefficients[index] += 1e-5
            plus = float(objective())
            coefficients[index] -= 2e-5
            minus = float(objective())
        self.assertAlmostEqual(analytic, (plus - minus) / 2e-5, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
