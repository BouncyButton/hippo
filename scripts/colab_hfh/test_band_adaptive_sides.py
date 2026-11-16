"""Checks for the separately labelled training-only adaptive-side pilot."""
import json
import unittest
from pathlib import Path


class AdaptiveSidesTests(unittest.TestCase):
    def test_training_difficulty_weights_are_detached_and_bounded(self):
        import torch
        from scripts.colab_hfh.bands_variants import make_band_class
        from thesis.new_constraints.bands.outer_boundary import build_boundary_bands
        labels = torch.zeros(1, 1, 10, 10, 10, dtype=torch.long)
        labels[:, :, 3:7, 3:7, 3:7] = 1
        inner, outer = build_boundary_bands(labels.bool(), steps=2)
        loss = make_band_class(.5, adaptive=True)(foreground_class_ids=(1,), complement_class_ids=(0,))
        for foreground_logit in (-8., 0., 8.):
            logits = torch.zeros(1, 2, 10, 10, 10)
            logits[:, 1] = foreground_logit
            logits.requires_grad_(True)
            result = loss(logits, labels).loss
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1]-logits[:, 0], labels[:, 0].float(), reduction='none')
            a = (bce*inner[:, 0]).sum()/(inner.sum()+1e-6)
            b = (bce*outer[:, 0]).sum()/(outer.sum()+1e-6)
            w = ((a.detach()+1e-6)/(a.detach()+b.detach()+2e-6)).clamp(.1,.9)
            expected = w*a+(1-w)*b
            self.assertFalse(w.requires_grad)
            self.assertTrue(torch.allclose(result, expected))
            self.assertTrue(torch.allclose(torch.autograd.grad(result, logits, retain_graph=True)[0],
                                          torch.autograd.grad(expected, logits)[0], atol=1e-7))
            self.assertAlmostEqual(float(w), .9 if foreground_logit < 0 else .1 if foreground_logit > 0 else .5, places=5)

    def test_variant_does_not_change_canonical_or_allow_undeclared_combinations(self):
        from scripts.colab_hfh.bands_variants import make_band_class
        from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
        make_band_class(.5, adaptive=True)
        self.assertIs(make_band_class(.5), OuterBoundaryBandLoss)
        for kwargs in (dict(inner_weight=.9),dict(inner_weight=.5,dynamic=True),
                       dict(inner_weight=.5,steps=1),dict(inner_weight=.5,focal_gamma=1)):
            with self.assertRaises(ValueError):make_band_class(**kwargs,adaptive=True)

    def test_same_nine_pairs_native_recipe_updates_and_extra_cost(self):
        from scripts.colab_hfh.constraint_schedule import make_payload, training_flops
        from scripts.colab_hfh.constraint_search import recipes
        from scripts.colab_hfh.constraint_worker import worker_source
        root=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
        base=json.loads((root/'bands_v1_payload.json').read_text())
        plan=json.loads((root/'constraint_matrix_plan.json').read_text())
        for model in ('nnunet','swinunetr'):
            adaptive = [r for r in recipes(model,plan,'dice_bands') if r.get('adaptive_sides')]
            self.assertEqual(sorted(r['target'] for r in adaptive), [.1, .3])
            for spec in adaptive:
                costs=plan['cost_references'][model]['canonical_costs']
                payload=make_payload(base,model,'dice_bands',costs,**spec)
                p=payload['protocols'][model]['protocol.json']
                s=p['models'][p['trainer_alias']]['objectives']['dice_bands']
                fixed=make_payload(base,model,'dice_bands',costs,**{**spec,'adaptive_sides':False})['protocols'][model]['protocol.json']
                old=fixed['models'][fixed['trainer_alias']]['objectives']['dice_bands']
                self.assertEqual(s['max_steps'],old['max_steps'])
                self.assertEqual(s['flops_per_update']-old['flops_per_update'],64)
                self.assertEqual(s['calibration_flops']-old['calibration_flops'],640)
                self.assertLessEqual(training_flops(s['max_steps'],s),p['budget_flops'])
                self.assertEqual(p['initial_lrs'],fixed['initial_lrs'])
                self.assertEqual(p['lr_horizon_updates'],fixed['lr_horizon_updates'])
                self.assertEqual(set(payload['folds']),{0,1,2})
                self.assertEqual(set(payload['seeds']),{0,1,2})
                self.assertEqual(payload['references'],{k:v for k,v in base['references'].items() if k.startswith(model+'_f')})
        source=worker_source()
        self.assertIn("or p.get('band_adaptive_sides', False)",source)
        compile(source,'constraint_worker','exec')


if __name__ == '__main__':
    unittest.main()
