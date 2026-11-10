"""Checks for the eight-case diagnostic, executed on a Slurm compute node."""
import unittest
import numpy as np
import torch
from .fit_eight import CoefficientRegressor, field_objective, select_cases, translate_image


class FittingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_selection_does_not_need_annotations(self):
        records = [dict(name=f"case_{i:03d}", prediction=np.ones(i+1)) for i in range(208)]
        selected = select_cases(records)
        self.assertEqual(selected, select_cases(records[::-1]))
        ids = [int(name.split("_")[-1]) for name in selected]
        for quartile in range(4):
            self.assertEqual(sum(quartile*52 <= i < (quartile+1)*52 for i in ids), 2)

    def test_identity_image_translation(self):
        image = np.arange(16**3).reshape((16,)*3).astype(np.float32)
        translated = translate_image(image, [3, 4, 5])
        np.testing.assert_array_equal(image, translated[3:19, 4:20, 5:21])

    def test_evaluation_is_independent_of_batch_partition(self):
        model = CoefficientRegressor().eval()
        torch.nn.init.normal_(model.head[-1].weight, std=.01)
        inputs = torch.randn(3, 5, 32, 32, 32)
        with torch.no_grad():
            together = model(inputs)
            separate = torch.cat([model(batch) for batch in inputs.split(1)])
        torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-6)

    def test_field_supervision_and_network_gradient(self):
        model = CoefficientRegressor()
        inputs = torch.randn(2, 5, 32, 32, 32)
        coefficients = model(inputs)
        self.assertEqual(tuple(coefficients.shape), (2, 32))
        torch.testing.assert_close(coefficients, torch.zeros_like(coefficients))
        target_coefficients = torch.randn(2, 32) * .05
        cases = []
        for index in range(2):
            design = torch.randn(100, 32)
            offset = torch.randn(100) * .01
            target = offset + design @ target_coefficients[index]
            cases.append(dict(design=design, offset=offset, oracle_delta=target,
                energy=(target-offset).square().mean(), source_band=torch.randn(100),
                truth_band=torch.randint(0, 2, (100,)).float(), outside_intersection=40,
                outside_prediction=50, truth_size=100))
        loss, relative_error, _ = field_objective(coefficients, cases)
        self.assertAlmostEqual(float(relative_error.detach()), 1., places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())
        self.assertGreater(float(model.head[-1].weight.grad.abs().max()), 0)
        _, oracle_error, _ = field_objective(target_coefficients, cases)
        self.assertLess(float(oracle_error), 1e-10)


if __name__ == "__main__":
    unittest.main()
