"""An outer-weighted BAND must change only the intended side gradients."""
import unittest
import torch
import torch.nn.functional as F
from scripts.medsam3_supervised_constraints import sampled_losses, VolumeSupport
from scripts.colab_hfh.medsam_constraints_loss_variants import side_weighted


class LossTests(unittest.TestCase):
    def test_canonical_is_exact_reference(self):
        self.assertIs(side_weighted(sampled_losses, .5), sampled_losses)

    def test_outer_weighting_changes_only_band_side_gradients(self):
        foreground = torch.zeros(6, 10, 10, dtype=torch.bool)
        foreground[1:5, 3:7, 3:7] = True
        support = VolumeSupport.build(foreground)
        x = torch.linspace(-2, 2, 100).reshape(10, 10).requires_grad_()
        kw = dict(log_probability=F.logsigmoid(x), log_complement=F.logsigmoid(-x))
        b0, e0 = sampled_losses(x.sigmoid(), x.sigmoid(), support, 2, **kw)
        b1, e1 = side_weighted(sampled_losses, .25)(x.sigmoid(), x.sigmoid(), support, 2, **kw)
        g0, = torch.autograd.grad(b0, x, retain_graph=True)
        g1, = torch.autograd.grad(b1, x)
        self.assertTrue(torch.equal(e0, e1))
        self.assertTrue(torch.isfinite(g1).all())
        self.assertTrue(torch.allclose(g1[support.inner[2]], .5*g0[support.inner[2]]))
        self.assertTrue(torch.allclose(g1[support.outer[2]], 1.5*g0[support.outer[2]]))

    def test_missing_side_is_not_silently_accepted(self):
        for value in (0, 1, -.1, float('nan')):
            with self.assertRaises(ValueError): side_weighted(sampled_losses, value)


if __name__ == '__main__': unittest.main()
