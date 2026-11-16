"""Component isolation, original RNG tape, objective transparency and pairing."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.colab_hfh.medsam_augmentation_components import (
    BASE_SPEC, COMPONENTS, component_spec, transform_pair, LoggedCriterion, attach_terms)
from scripts.colab_hfh.medsam_augmentation_components_campaign import make_protocol, pilot_phase_source
from scripts.colab_hfh.medsam_augmentation_components_worker import worker_source


class ComponentTests(unittest.TestCase):
    def test_each_component_uses_original_joint_draws(self):
        image = np.arange(4096, dtype=np.uint16).reshape(64,64).astype(np.uint8)
        for seed in range(24):
            rng = np.random.default_rng(seed)
            gamma = rng.uniform(.9,1.1) if rng.random()<.5 else 1.
            gain = rng.uniform(.9,1.1) if rng.random()<.5 else 1.
            sigma = rng.uniform(0,.015) if rng.random()<.2 else 0.
            noise = [rng.normal(0.,sigma,image.shape).astype(np.float32) if sigma else 0. for _ in range(2)]
            before = np.random.get_state()
            for component in COMPONENTS:
                actual, params = transform_pair([image,image], seed, component)
                self.assertEqual((params['sampled_gamma'],params['sampled_gain'],params['sampled_noise_std']), (gamma,gain,sigma))
                for i, result in enumerate(actual):
                    x=image.astype(np.float32)/255.
                    expected = x**gamma if component=='gamma' else x*gain if component=='gain' else x+noise[i]
                    expected[image==0]=0.
                    expected=np.round(np.clip(expected,0.,1.)*255).astype(np.uint8)
                    np.testing.assert_array_equal(result,expected)
                    self.assertEqual(result.dtype,np.uint8)
            after=np.random.get_state()
            np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_identity_probability_keeps_unselected_factors_off(self):
        for component in COMPONENTS:
            spec=component_spec(component)
            for other in COMPONENTS:
                self.assertEqual(spec[other+'_probability'],BASE_SPEC[other+'_probability'] if other==component else 0.)

    def test_logged_criterion_preserves_return_graph_gradient_and_rng(self):
        class Criterion:
            loss_fns_find=[SimpleNamespace(weight_dict={'loss_ce':20.,'loss_mask':200.})]
            normalize_by_stage_num=scale_by_find_batch_size=False
            def __call__(self,x):
                v=torch.nn.functional.dropout(x,p=.1,training=True).square().mean()
                ce=v*.2;mask=v*.03;aux=v*.1
                self.result={'loss_ce':ce,'loss_ce_aux_0':aux,'loss_mask':mask,
                             'core_loss':20.*ce+20.*aux+200.*mask}
                return self.result
        c=Criterion();x=torch.randn(8,requires_grad=True)
        torch.manual_seed(73);state=torch.get_rng_state();out=c(x)
        a=torch.autograd.grad(out['core_loss'],x)[0];after=torch.get_rng_state()
        torch.set_rng_state(state);logged=LoggedCriterion(c);out2=logged(x)
        self.assertIs(out2,c.result)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        b=torch.autograd.grad(out2['core_loss'],x)[0]
        self.assertTrue(torch.equal(a,b));self.assertEqual(float(out['core_loss']),float(out2['core_loss']))
        row=logged.rows[0]
        self.assertFalse(any(v.requires_grad for v in row.values()))
        self.assertTrue(torch.isclose(row['core_loss'],row['weighted/loss_ce']+row['weighted/loss_mask']))

    def test_protocol_preserves_recipe_and_has_three_unique_jobs(self):
        p=dict(arm='dice',fold=0,seed=0,updates=125,augmentation='none',
               frozen_sample_schedule=[['case',2]],frozen_learning_rates=[.1],train_cases=['case'])
        control=dict(protocol=p,archive_sha256='abc');snapshot=deepcopy(control)
        for component in COMPONENTS:
            out=make_protocol(control,Path('/control.zip'),component)
            self.assertEqual(out['augmentation'],component_spec(component))
            for key in p.keys()-{'arm','augmentation'}:self.assertEqual(out[key],p[key])
        self.assertEqual(control,snapshot)
        compile(worker_source(),'worker','exec');compile(pilot_phase_source(),'scheduler','exec')
        self.assertIn('for component in COMPONENTS',pilot_phase_source())
        self.assertIn('medsam_augmentation_components import install',worker_source())


if __name__=='__main__':unittest.main()
