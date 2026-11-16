"""Gradient routing and spatial accounting checks for mask-only BAND."""
import unittest
from types import SimpleNamespace

import torch
from scripts.colab_hfh.medsam_constraints_mask_only import conditional_logits, mask_band, install_spatial_evaluation


class MaskOnlyTests(unittest.TestCase):
    def output(self, positive):
        return dict(pred_masks=torch.randn(1,3,4,4,requires_grad=True),
                    pred_logits=torch.randn(1,3,1,requires_grad=True),
                    presence_logit_dec=torch.randn(1,1,requires_grad=True),
                    indices=(torch.tensor([0] if positive else [],dtype=torch.long),
                             torch.tensor([1] if positive else [],dtype=torch.long),None))

    def test_positive_native_match_and_no_confidence_gradient(self):
        o=self.output(True)
        x=conditional_logits(o,(4,4),True)
        self.assertTrue(torch.equal(x,o['pred_masks'][0,1]))
        grads=torch.autograd.grad(x.square().sum(),[o['pred_masks'],o['pred_logits'],o['presence_logit_dec']],allow_unused=True)
        self.assertIsNone(grads[1]);self.assertIsNone(grads[2])
        self.assertGreater(float(grads[0][0,1].abs().sum()),0)
        self.assertEqual(float(grads[0][0,[0,2]].abs().sum()),0)

    def test_empty_slice_keeps_negative_mask_gradient(self):
        o=self.output(False);x=conditional_logits(o,(4,4),False)
        support=SimpleNamespace(foreground=torch.zeros(2,4,4,dtype=torch.bool),
            inner=torch.zeros(2,4,4,dtype=torch.bool),outer=torch.ones(2,4,4,dtype=torch.bool),inner_count=1,outer_count=32)
        loss=mask_band(x,support,0)
        grads=torch.autograd.grad(loss,[o['pred_masks'],o['pred_logits'],o['presence_logit_dec']],allow_unused=True)
        self.assertGreater(float(grads[0].abs().sum()),0)
        self.assertTrue((grads[0]>=0).all())
        self.assertIsNone(grads[1]);self.assertIsNone(grads[2])

    def test_volume_normalization(self):
        x=torch.zeros(4,4,requires_grad=True)
        fg=torch.zeros(2,4,4,dtype=torch.bool);fg[:,1:3,1:3]=True
        support=SimpleNamespace(foreground=fg,inner=fg,outer=~fg,
                                inner_count=int(fg.sum()),outer_count=int((~fg).sum()))
        self.assertAlmostEqual(float(mask_band(x,support,0).detach()),float(torch.log(torch.tensor(2.))),places=6)

    def test_missing_match_fails_closed(self):
        with self.assertRaises(ValueError):conditional_logits(self.output(False),(4,4),True)

    def test_spatial_evaluator_adaptation(self):
        from scripts.cross_dataset_campaign import foundation
        matched=SimpleNamespace(REFERENCE_EVALUATE=foundation.evaluate)
        install_spatial_evaluation(matched)
        self.assertIn('spatial',matched.REFERENCE_EVALUATE.__code__.co_names)


if __name__=='__main__':unittest.main()
