"""Compare sampled objective values AND gradients to original full 3-D losses."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.medsam3_supervised_constraints import (VolumeSupport, sampled_losses,
                                                   semantic_probability, semantic_log_fields)
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss


class EstimatorTests(unittest.TestCase):
    def test_full_volume_values_and_gradients(self):
        torch.manual_seed(18)
        for kind in ('interior', 'border', 'empty', 'full'):
            with self.subTest(kind=kind):
                label = torch.zeros(5, 9, 7, dtype=torch.long)
                if kind == 'interior':
                    label[1:4, 2:7, 2:5] = 1
                elif kind == 'border':
                    label[:4, 1:8, :3] = 2
                elif kind == 'full':
                    label[:] = 1
                p = (.05+.9*torch.rand(label.shape)).requires_grad_()
                logits = torch.stack((torch.log1p(-p), p.log()-math.log(2),
                                      p.log()-math.log(2)), dim=0)[None]
                expected = (OuterBoundaryBandLoss()(logits, label[None]).loss,
                            BoundaryEdgeConsistencyLoss()(logits, label[None]))
                support = VolumeSupport.build(label > 0)
                sampled = [sampled_losses(p[z], p[(z+1)%5], support, z) for z in range(5)]
                for index in (0, 1):
                    actual = torch.stack([pair[index] for pair in sampled]).mean()
                    torch.testing.assert_close(actual, expected[index], atol=2e-6, rtol=2e-5)
                    ga = torch.autograd.grad(actual, p, retain_graph=True)[0]
                    ge = torch.autograd.grad(expected[index], p, retain_graph=True)[0]
                    torch.testing.assert_close(ga, ge, atol=2e-6, rtol=3e-5)

    def test_last_anchor_does_not_create_wraparound_face(self):
        label = torch.zeros(5, 9, 7, dtype=torch.bool)
        label[-1, 2:7, 1:5] = True
        support = VolumeSupport.build(label)
        current = torch.rand(9, 7, requires_grad=True)
        wrapped = torch.rand(9, 7, requires_grad=True)
        band, edge = sampled_losses(current, wrapped, support, 4)
        gradient = torch.autograd.grad(band+edge, wrapped)[0]
        self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def test_confidences_and_mask_receive_gradients(self):
        output = {'pred_masks': torch.zeros(1, 2, 3, 4, requires_grad=True),
                  'pred_logits': torch.tensor([[[1.], [-1.]]], requires_grad=True),
                  'presence_logit_dec': torch.ones(1, 1, requires_grad=True)}
        p = semantic_probability(output, (9, 7))
        self.assertEqual(tuple(p.shape), (9, 7))
        grads = torch.autograd.grad(p.sum(), tuple(output.values()))
        self.assertTrue(all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads))

    def test_extreme_confidence_retains_corrective_band_gradients(self):
        label = torch.zeros(3, 7, 9, dtype=torch.bool)
        label[1, 2:5, 3:6] = True
        support = VolumeSupport.build(label)
        for value in (-100., 30., 100.):
            with self.subTest(value=value):
                output = {'pred_masks': torch.full((1, 1, 7, 9), value, requires_grad=True),
                          'pred_logits': torch.full((1, 1, 1), value, requires_grad=True),
                          'presence_logit_dec': torch.full((1, 1), value, requires_grad=True)}
                log_p, log_q = semantic_log_fields(output, (7, 9))
                band, _ = sampled_losses(log_p.exp(), log_p.exp(), support, 1,
                                          log_probability=log_p, log_complement=log_q)
                gradients = torch.autograd.grad(band, tuple(output.values()))
                self.assertTrue(torch.isfinite(band))
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
