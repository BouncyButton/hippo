import unittest
import json
import numpy as np
import torch

from scripts.colab_hfh.residual_band import ResidualBandLoss, geometry, selection, policy_dict


class ResidualBandTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.truth=np.zeros((64,64,64),dtype=np.uint8)
        self.truth[20:44,22:42,20:44]=1
        self.labels=torch.tensor(self.truth[None,None],dtype=torch.long)

    def logits(self,prediction):
        z=torch.zeros((1,2,64,64,64))
        z[0,1]=torch.tensor(np.where(prediction,2.,-2.))
        return z.requires_grad_(True)

    def test_background_and_perfect_predictions_are_off(self):
        for prediction in (np.zeros_like(self.truth),self.truth):
            z=self.logits(prediction); r=ResidualBandLoss()(z,self.labels)
            self.assertEqual(float(r.loss),0.)
            r.loss.backward();self.assertEqual(float(z.grad.abs().sum()),0.)

    def test_corrects_both_sides_without_pushing_correct_or_distant_voxels(self):
        pred=self.truth.copy();pred[20,25:29,25:29]=0;pred[19,30:34,30:34]=1
        z=self.logits(pred);r=ResidualBandLoss()(z,self.labels)
        self.assertTrue(r.details['ready']);self.assertEqual(r.details['selected_errors'],32)
        r.loss.backward()
        self.assertLess(float(z.grad[0,1,20,26,26]),0.)
        self.assertGreater(float(z.grad[0,1,19,31,31]),0.)
        self.assertEqual(float(z.grad[0,1,30,30,30]),0.)
        self.assertEqual(int((z.grad[0,1]!=0).sum()),32)

    def test_deep_misses_and_distant_false_positives_are_not_boundary_errors(self):
        pred=self.truth.copy();pred[30:32,30:32,30:32]=0;pred[5:7,25:27,5:7]=1
        r=ResidualBandLoss()(self.logits(pred),self.labels)
        self.assertTrue(r.details['ready']);self.assertEqual(float(r.loss),0.)
        self.assertTrue(all(x['reason']=='no_suitable_local_shell' for x in r.details['sides'].values()))

    def test_physical_shell_respects_two_mm_axis(self):
        _,_,_,outer=geometry(self.truth.tobytes())
        self.assertTrue(outer[0,19,30,30])
        self.assertFalse(outer[0,30,21,30])
        self.assertTrue(outer[1,30,21,30])

    def test_inner_shell_never_becomes_whole_foreground(self):
        truth=np.zeros_like(self.truth);truth[20:44,30:31,20:44]=1
        gt,roi,inner,outer=geometry(truth.tobytes())
        pred=gt.copy();pred[20:24,30,20:24]=0
        support,report=selection(gt,pred,roi,inner,outer)
        self.assertTrue(report['ready']);self.assertEqual(int(support.sum()),0)

    def test_calibration_envelope_bounds_actual_logit_gradient(self):
        pred=self.truth.copy();pred[20,25:29,25:29]=0;pred[19,30:34,30:34]=1
        z=self.logits(pred)
        actual=torch.autograd.grad(ResidualBandLoss()(z,self.labels).loss,z)[0]
        envelope=torch.autograd.grad(ResidualBandLoss(envelope=True)(z,self.labels).loss,z)[0]
        self.assertLessEqual(float(actual.norm()),float(envelope.norm()))
        torch.testing.assert_close(actual[actual!=0],envelope[actual!=0])

    def test_padding_is_excluded(self):
        pred=self.truth.copy();pred[:,0,:]=1
        r=ResidualBandLoss()(self.logits(pred),self.labels)
        self.assertEqual(r.details['fp'],0);self.assertEqual(float(r.loss),0.)

    def test_policy_survives_json_storage(self):
        self.assertEqual(json.loads(json.dumps(policy_dict())),policy_dict())

    def test_smallest_covering_width_and_independent_sides(self):
        pred=self.truth.copy()
        pred[20,25:29,25:29]=0
        pred[30:34,21,30:34]=1
        r=ResidualBandLoss()(self.logits(pred),self.labels)
        self.assertEqual(r.details['sides']['inner']['selected_width_mm'],1.)
        self.assertEqual(r.details['sides']['outer']['selected_width_mm'],2.)


if __name__=='__main__':unittest.main()
