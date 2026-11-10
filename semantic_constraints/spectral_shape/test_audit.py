"""Focused correctness checks; numerical execution occurs inside sbatch."""
import unittest
import numpy as np
from scipy import fft
import torch

from .audit import bands, band_fit, dct_design, dct_frequencies, fit_pca, shift_into_canvas


class SpectralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_dct_basis_matches_scipy(self):
        size = 8
        frequencies = dct_frequencies(size=size, count=16)
        points = np.indices((size,) * 3).reshape(3, -1).T
        design = dct_design(torch.from_numpy(points), torch.from_numpy(frequencies), size=size).numpy()
        for index, frequency in enumerate(frequencies):
            coeff = np.zeros((size,) * 3)
            coeff[tuple(frequency)] = 1
            expected = fft.idctn(coeff, type=2, norm="ortho").ravel()
            np.testing.assert_allclose(design[:, index], expected, atol=7e-8)
        np.testing.assert_allclose(design.T @ design, np.eye(16), atol=2e-6)

    def test_pca_reconstruction_and_heldout_component(self):
        directions = torch.linalg.qr(torch.randn(40, 3)).Q
        offset = torch.randn(40)
        scores = torch.randn(12, 3)
        train = scores @ directions.T + offset
        mean, modes, explained = fit_pca(train, 3)
        reconstructed = mean + ((train - mean) @ modes.T) @ modes
        torch.testing.assert_close(train, reconstructed, atol=3e-6, rtol=3e-6)
        self.assertGreater(float(explained[:3].sum()), .99999)
        off_span = torch.randn(40)
        off_span -= (off_span @ modes.T) @ modes
        heldout = mean + off_span
        projected = mean + ((heldout - mean) @ modes.T) @ modes
        self.assertGreater(float((heldout - projected).square().sum()), 1)

    def test_band_fit_and_exact_identity(self):
        design = torch.randn(100, 8)
        weights = torch.randn(8)
        mean = torch.randn(100)
        target = mean + design @ weights
        reconstructed, coefficients = band_fit(design, target, mean, ridge=0)
        torch.testing.assert_close(target, reconstructed, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(weights, coefficients, atol=2e-6, rtol=2e-6)

    def test_integer_alignment_and_bands(self):
        mask = np.zeros((16,) * 3, bool)
        mask[5:11, 4:12, 6:10] = True
        shifted = shift_into_canvas(mask, [5, 6, 7], size=32)
        self.assertTrue(np.array_equal(mask, shifted[5:21, 6:22, 7:23]))
        self.assertTrue(np.array_equal(bands(shifted), shift_into_canvas(bands(mask), [5, 6, 7], size=32)))


if __name__ == "__main__":
    unittest.main()
