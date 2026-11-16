import json
import unittest
from scripts.colab_hfh.prepare_bands import EXPORT
from scripts.colab_hfh.bands_strict_search import qualifies,make_payload,specifications


class StrictTests(unittest.TestCase):
    def test_positive_average_is_insufficient(self):
        rows=[dict(fold=f,seed=s,band_dice=.7,delta_dice_pp=2.) for f in range(3) for s in range(3)]
        self.assertTrue(qualifies(rows));self.assertFalse(qualifies(rows[:-1]))
        rows[0]['delta_dice_pp']=.99;self.assertFalse(qualifies(rows))
        rows[0]['delta_dice_pp']=1.;self.assertTrue(qualifies(rows))

    def test_all_candidates_match_native_recipe_and_fit_budget(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        for model in ('nnunet','swinunetr'):
            for spec in specifications(model):
                payload=make_payload(base,model,spec);p=payload['protocols'][model]['protocol.json'];s=p['models'][p['trainer_alias']]['objectives']['dice_bands']
                original=base['protocols'][model]['protocol.json']
                self.assertLessEqual(s['consumed_training_flops'],p['budget_flops'])
                self.assertEqual(p['initial_lrs'],original['initial_lrs'])
                self.assertEqual(p['lr_horizon_updates'],original['lr_horizon_updates'])
                self.assertEqual(s['max_steps'],original['models'][original['trainer_alias']]['objectives']['dice']['max_steps'])
                for key,r in payload['references'].items():self.assertEqual(r,base['references'][key])

    def test_weighted_loss_and_gradient_match_explicit_formula(self):
        import torch
        from scripts.colab_hfh.bands_variants import make_band_class
        from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss,build_boundary_bands
        self.assertIs(make_band_class(.5),OuterBoundaryBandLoss)
        torch.manual_seed(0)
        logits=torch.randn(1,2,10,10,10,requires_grad=True);labels=torch.zeros(1,1,10,10,10,dtype=torch.long);labels[:,:,3:7,3:7,3:7]=1
        loss=make_band_class(.75)(foreground_class_ids=(1,),complement_class_ids=(0,))(logits,labels).loss
        inner,outer=build_boundary_bands(labels.bool(),steps=2)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],labels[:,0].float(),reduction='none')
        expected=.75*(bce*inner[:,0]).sum()/(inner.sum()+1e-6)+.25*(bce*outer[:,0]).sum()/(outer.sum()+1e-6)
        self.assertTrue(torch.allclose(loss,expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(loss,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0]))

    def test_focal_loss_and_gradient_match_explicit_formula(self):
        import torch
        from scripts.colab_hfh.bands_variants import make_band_class
        from thesis.new_constraints.bands.outer_boundary import build_boundary_bands
        torch.manual_seed(1)
        labels=torch.zeros(1,1,10,10,10,dtype=torch.long);labels[:,:,3:7,3:7,3:7]=1
        inner,outer=build_boundary_bands(labels.bool(),steps=2)
        for gamma in (1,2):
            for weight in (.5,.9):
                logits=torch.randn(1,2,10,10,10,requires_grad=True)
                loss=make_band_class(weight,gamma)(foreground_class_ids=(1,),complement_class_ids=(0,))(logits,labels).loss
                odds=logits[:,1]-logits[:,0];target=labels[:,0].float();prob=odds.sigmoid()
                pt=target*prob+(1-target)*(1-prob)
                values=(1-pt).pow(gamma)*torch.nn.functional.binary_cross_entropy_with_logits(odds,target,reduction='none')
                expected=weight*(values*inner[:,0]).sum()/(inner.sum()+1e-6)+(1-weight)*(values*outer[:,0]).sum()/(outer.sum()+1e-6)
                self.assertTrue(torch.allclose(loss,expected))
                self.assertTrue(torch.allclose(torch.autograd.grad(loss,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0],atol=1e-7))

    def test_narrow_band_overrides_factory_steps_and_matches_native(self):
        import torch
        from scripts.colab_hfh.bands_variants import make_band_class
        from thesis.new_constraints.bands.outer_boundary import build_boundary_bands
        torch.manual_seed(3)
        labels=torch.zeros(1,1,10,10,10,dtype=torch.long);labels[:,:,3:7,3:7,3:7]=1
        logits=torch.randn(1,2,10,10,10,requires_grad=True)
        options=dict(foreground_class_ids=(1,),complement_class_ids=(0,))
        narrow=make_band_class(.5,0,1)(steps=2,**options)
        self.assertEqual(narrow.steps,1)
        actual=narrow(logits,labels).loss
        inner,outer=build_boundary_bands(labels.bool(),steps=1)
        values=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],labels[:,0].float(),reduction='none')
        expected=.5*((values*inner[:,0]).sum()/(inner.sum()+1e-6)+(values*outer[:,0]).sum()/(outer.sum()+1e-6))
        self.assertTrue(torch.allclose(actual,expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0],atol=1e-7))

    def test_momentum_reset_preserves_weights_and_matches_fresh_sgd(self):
        import torch
        from scripts.colab_hfh.bands_variants import reset_sgd_momentum
        parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
        optimizer=torch.optim.SGD([parameter],lr=.1,momentum=.95,nesterov=True)
        parameter.grad=torch.tensor([.3,-.2]);optimizer.step()
        before=parameter.detach().clone()
        self.assertEqual(reset_sgd_momentum(optimizer),1)
        self.assertTrue(torch.equal(parameter,before))
        reference=torch.nn.Parameter(before.clone())
        fresh=torch.optim.SGD([reference],lr=.1,momentum=.95,nesterov=True)
        parameter.grad=torch.tensor([-.1,.4]);reference.grad=parameter.grad.clone()
        optimizer.step();fresh.step()
        self.assertTrue(torch.equal(parameter,reference))

    def test_release_is_native_at_final_update_without_changing_prefix(self):
        from scripts.colab_hfh.bands_variants import release_factor
        self.assertEqual(release_factor(0,90,18),1.)
        self.assertEqual(release_factor(71,90,18),1.)
        self.assertEqual(release_factor(72,90,18),1.)
        self.assertAlmostEqual(release_factor(80,90,18),9/17)
        self.assertEqual(release_factor(89,90,18),0.)

    def test_initial_ledger_precedes_first_checkpoint_without_resetting_resume(self):
        import ast
        import types
        from pathlib import Path
        from unittest.mock import patch
        from scripts.colab_hfh.bands_variants import install_training_transition
        path=Path('scripts/cross_dataset_campaign/capped_core.py')
        tree=ast.parse(path.read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        module=types.ModuleType('ledger_hook_test')
        exec(compile(ast.Module(body=[run],type_ignores=[]),str(path),'exec'),vars(module))
        captured=[]
        real_compile=compile
        def capture(source,*args,**kwargs):
            captured.append(source)
            return real_compile(source,*args,**kwargs)
        with patch('builtins.compile',side_effect=capture):
            install_training_transition(module)
        patched=next(s for s in captured if isinstance(s,str) and s.startswith('def run('))
        parsed=ast.parse(patched).body[0]
        init=next(n for n in parsed.body if isinstance(n,ast.If)
                  and ast.unparse(n.test)=='not ledger_path.exists()')
        self.assertEqual(ast.unparse(init.body[0]),'base.atomic_json(ledger_path, ledger)')
        commit=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='commit')
        self.assertLess(init.lineno,commit.lineno)

    def test_side_transition_matches_fixed_loss_and_endpoints(self):
        import torch
        from scripts.colab_hfh.bands_variants import make_band_class,side_weight_for_step
        self.assertEqual(side_weight_for_step(.9,.5,45,22,90),.9)
        self.assertEqual(side_weight_for_step(.9,.5,45,89,90),.5)
        self.assertAlmostEqual(side_weight_for_step(.9,.5,45,67,90),.7)
        labels=torch.zeros(1,1,10,10,10,dtype=torch.long);labels[:,:,3:7,3:7,3:7]=1
        logits=torch.randn(1,2,10,10,10,requires_grad=True)
        options=dict(foreground_class_ids=(1,),complement_class_ids=(0,))
        dynamic=make_band_class(.9,steps=1,dynamic=True)
        for weight in (.9,.7,.5):
            dynamic.hfh_inner_weight=weight
            actual=dynamic(**options)(logits,labels).loss
            expected=make_band_class(weight,steps=1)(**options)(logits,labels).loss
            self.assertTrue(torch.allclose(actual,expected))
            self.assertTrue(torch.allclose(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits,retain_graph=True)[0]))

    def test_optimizer_diagnostic_cannot_be_a_band_winner(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        payload=make_payload(base,'nnunet',specifications('nnunet')[83])
        self.assertTrue(payload['diagnostic_control'])
        self.assertNotIn('min_pair_delta_pp',payload)
        self.assertEqual(set(payload['pairs']),{'nnunet_f2_s0','nnunet_f0_s2'})
        self.assertFalse(qualifies([dict(fold=f,seed=s,band_dice=.9,delta_dice_pp=10.) for f,s in [(2,0),(0,2)]]))
        self.assertFalse(qualifies([dict(fold=f,seed=s,band_dice=.9,delta_dice_pp=10.,control_kind='sgd_reset_without_auxiliary_gradient') for f in range(3) for s in range(3)]))
        from scripts.colab_hfh.bands_archive import valid_band_weight
        protocol=payload['protocols']['nnunet']['protocol.json']
        self.assertTrue(valid_band_weight(protocol,0.))
        self.assertFalse(valid_band_weight(protocol,.1))
        self.assertFalse(valid_band_weight({},0.))
        self.assertFalse(valid_band_weight({},float('nan')))
        self.assertTrue(valid_band_weight({},.1))


if __name__=='__main__':unittest.main()
