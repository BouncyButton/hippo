"""Loss/autograd contracts, gating, exact restart and a real Swin backward check."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.depth_supervision.utils import save_checkpoint, atomic_json
from .prior import AnatomyEncoder, PriorTrainingModel, LearnedLoss, one_hot, Corruptions, balanced_ce
from .verify import probe, summarize, counterfactual_probabilities
from . import train


def example(size=16):
    label=torch.zeros(1,size,size,size,dtype=torch.long)
    label[:,3:size-3,4:size-4,4:size-4]=1
    label[:,size//2:size-3,4:size-4,4:size-4]=2
    context=torch.ones(1,2,size,size,size)
    context[:,:,0:2]=0; context[:,:,-2:]=0
    return dict(case_name='unit',label=label,image=torch.randn(1,size,size,size),affine=np.eye(4)),context


def frozen(path):
    torch.manual_seed(2)
    save_checkpoint(path,dict(encoder=AnatomyEncoder().state_dict(),feature_variances=[1.,1.,1.]))


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'encoder.pt'; frozen(self.path)
        self.loss=LearnedLoss(self.path,torch.device('cpu'))
        self.item,self.context=example()
    def tearDown(self):self.temp.cleanup()

    def test_identity_zero_and_frozen_encoder_receives_no_gradients(self):
        target=self.item['label'][None]
        self.assertEqual(float(self.loss.features_loss(one_hot(target[:,0]),target,self.context)),0.)
        logits=torch.randn(1,3,16,16,16,requires_grad=True)
        self.loss(logits,target,self.context).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.loss.parameters()))

    def test_artificial_padding_cannot_change_loss_or_receive_gradient(self):
        logits=torch.randn(1,3,16,16,16,requires_grad=True)
        labels=self.item['label'][None]
        a=self.loss(logits,labels,self.context)
        b=logits.detach().clone(); b[:,:,0:2]=100*torch.randn_like(b[:,:,0:2]);b[:,:,-2:]=100*torch.randn_like(b[:,:,-2:])
        torch.testing.assert_close(a,self.loss(b,labels,self.context),rtol=0,atol=0)
        a.backward()
        self.assertEqual(float(logits.grad[:,:,0:2].abs().sum()),0.)

    def test_each_corruption_has_expected_class_and_foreground_effect(self):
        maker=Corruptions(self.item,self.context)
        for kind in ('missing','extra','swap'):
            for c in (1,2):
                wrong,mask=maker.make(kind,c,2.5,np.random.default_rng(3))
                self.assertTrue(mask.any())
                self.assertTrue(np.array_equal(wrong[~mask],maker.truth[~mask]))
                delta=int((wrong>0).sum()-(maker.truth>0).sum())
                self.assertEqual(delta, -int(mask.sum()) if kind=='missing' else int(mask.sum()) if kind=='extra' else 0)
                row=probe(self.loss,self.item,self.context,wrong,mask,kind,.95,None)
                self.assertTrue(row['detected'] and row['gradient_step_decreases'])

    def test_prior_encoder_gets_gradients_from_reconstruction_and_error_training(self):
        model=PriorTrainingModel()
        labels=self.item['label']
        candidate=labels.clone();candidate[:,5:8,5:8,5:8]=0
        errors=(candidate!=labels).long()
        output=model(one_hot(candidate)*.97+.01,self.context)
        loss=sum(balanced_ce(o,y,self.context[:,1:2]) for o,y in zip(output,(candidate,labels,errors)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_each_error_kind_must_pass_gate(self):
        settings=dict(helpful_fraction=.9,minimum_probe_helpful_fraction=.75,repair_path_fraction=.95,mass_direction_fraction=.95)
        rows=[dict(kind=k,voxels=10,helpful_fraction=1.,mean_true_probability_gain=.01,detected=True,
            repair_path_decreases=True,gradient_step_decreases=True,mass_direction_ok=True,finite=True) for k in ('missing','extra','swap')]
        rows[0]['mass_direction_ok']=False
        groups=summarize(rows,settings)
        self.assertFalse(groups['missing']['passed'])
        self.assertTrue(groups['extra']['passed'] and groups['swap']['passed'])

    def test_soft_swap_preserves_foreground_probability_and_third_class(self):
        truth=self.item['label'];wrong=truth.clone();mask=truth==1;wrong[mask]=2
        for confidence in (.8,.95):
            base,candidate=counterfactual_probabilities(truth,wrong,mask,confidence)
            torch.testing.assert_close(candidate[:,0],base[:,0],rtol=0,atol=0)
            torch.testing.assert_close(candidate[:,1:].sum(1),base[:,1:].sum(1),rtol=0,atol=1e-7)
            torch.testing.assert_close(candidate.sum(1),torch.ones_like(candidate[:,0]),rtol=0,atol=1e-7)
            self.assertTrue((candidate>0).all())

    def test_failed_gate_stops_before_network_construction(self):
        with patch.object(train,'verify_gate',side_effect=AssertionError('failed gate')),patch.object(train,'model_factory') as factory:
            with self.assertRaises(AssertionError):train.run(Path(self.temp.name),{},torch.device('cpu'))
            factory.assert_not_called()

    def test_exploratory_authorization_preserves_failed_audit_and_binds_hashes(self):
        from .verify import verify_gate
        from scripts.depth_supervision.utils import digest
        root=Path(self.temp.name)/'audit';(root/'encoder').mkdir(parents=True)
        (root/'encoder/frozen.pt').write_bytes(self.path.read_bytes())
        settings=dict(seed=20261009,radii_mm=[1.,2.5,4.],confidences=[.8,.95],constructions=['isolated','uncertain'],
            helpful_fraction=.9,minimum_probe_helpful_fraction=.75,repair_path_fraction=.95,mass_direction_fraction=.95)
        rows=[]
        for kind in ('missing','extra','swap'):
            for construction in settings['constructions']:
                for i in range(120):
                    rows.append(dict(kind=kind,construction=construction,voxels=10,helpful_fraction=1.,mean_true_probability_gain=.01,
                        detected=True,repair_path_decreases=True,gradient_step_decreases=True,finite=True,
                        mass_direction_ok=not (kind=='swap' and construction=='uncertain' and i<25)))
        value=dict(passed=False,loss_only=True,includes_dice_gradient=False,protocol_sha256='original',encoder_sha256=digest(self.path),
            thresholds=settings,train_cases=[str(i) for i in range(10)],validation_cases_used=[],probes=rows,groups=summarize(rows,settings))
        atomic_json(root/'gradient_gate.json',value)
        p=dict(identity_sha='original',gate=settings,splits={'5pct':{'train':value['train_cases']}})
        with self.assertRaises(AssertionError):verify_gate(root,p)
        p['identity_sha']='exploratory'
        p['exploratory_audit_acceptance']=dict(user_authorized=True,scope='one_5pct_swin_run',audit_protocol_sha256='original',
            audit_sha256=digest(root/'gradient_gate.json'),encoder_sha256=digest(self.path),accepted_failed_groups=['swap_uncertain'])
        before=digest(root/'gradient_gate.json')
        self.assertFalse(verify_gate(root,p)['passed'])
        self.assertEqual(digest(root/'gradient_gate.json'),before)
        p['exploratory_audit_acceptance']['audit_sha256']='changed'
        with self.assertRaises(AssertionError):verify_gate(root,p)
        p['exploratory_audit_acceptance']['audit_sha256']=before
        p['exploratory_audit_acceptance']['encoder_sha256']='changed'
        with self.assertRaises(AssertionError):verify_gate(root,p)
        p['exploratory_audit_acceptance']['encoder_sha256']=digest(self.path)
        p['exploratory_audit_acceptance']['user_authorized']=False
        with self.assertRaises(AssertionError):verify_gate(root,p)

    def test_gradient_refinement_updates_encoder_and_preserves_initial_checkpoint(self):
        from . import refine
        from scripts.depth_supervision.utils import digest
        root=Path(self.temp.name)/'refinement';(root/'encoder').mkdir(parents=True)
        (root/'encoder/frozen.pt').write_bytes(self.path.read_bytes())
        model=PriorTrainingModel();model.encoder.load_state_dict(self.loss.encoder.state_dict())
        ident={'unit':'initial'}
        save_checkpoint(root/'encoder/latest.pt',dict(step=1000,identity=ident,model=model.state_dict()))
        class Items:
            def __init__(self):
                self.items=[]
                for i in range(10):
                    item,_=example(12);item['case_name']=str(i);self.items.append(item)
            def __getitem__(self,i):return self.items[i]
        class Context:
            def get(self,item):return torch.ones(1,2,12,12,12)
        p=dict(identity_sha='refinement-unit',reused_encoder={'sha256':digest(self.path),'identity':ident},
            encoder={'steps':1000},splits={'5pct':{'train':[str(i) for i in range(10)]}},
            refinement=dict(seed=17,steps=2,lr=.0003,representation_weight=.1))
        with patch.object(refine,'load_training',return_value=(Items(),Context())):
            refine.run_refinement(root,p,torch.device('cpu'))
        self.assertEqual(digest(root/'encoder/initial_frozen.pt'),digest(self.path))
        self.assertNotEqual(digest(root/'encoder/frozen.pt'),digest(self.path))
        record=json.loads((root/'encoder/refinement_completion.json').read_text())
        self.assertEqual(record['steps'],2)
        self.assertEqual(record['validation_cases_used'],[])

    def test_probability_step_objective_penalizes_shrinking_swap_and_backpropagates(self):
        from .mass_refine import repair_objective
        logits=torch.tensor([.025,.025,.95]).log().reshape(1,3,1,1,1).requires_grad_()
        target=torch.tensor([0.,1.,0.]).reshape(1,3,1,1,1)
        mask=torch.ones(1,1,1,1,dtype=torch.bool);context=torch.ones(1,2,1,1,1)
        # Lowering only the confident wrong class diverts probability to BG.
        bad=torch.tensor([0.,0.,1.]).reshape_as(logits).requires_grad_()
        a,penalty,mass,gain=repair_objective(logits,bad,target,mask,context,'swap',.1)
        self.assertLess(float(mass),0);self.assertGreater(float(gain),0);self.assertGreater(float(penalty),0)
        (a+2*penalty).backward();self.assertTrue(torch.isfinite(bad.grad).all());self.assertGreater(float(bad.grad.abs().sum()),0)
        # Raising only the correct class restores tissue rather than deleting it.
        good=torch.tensor([0.,-1.,0.]).reshape_as(logits)
        _,penalty,mass,gain=repair_objective(logits.detach(),good,target,mask,context,'swap',.1)
        self.assertGreater(float(mass),0);self.assertGreater(float(gain),0);self.assertEqual(float(penalty),0)

    def test_mass_refinement_reuses_checkpoint_and_has_second_order_gradients(self):
        from . import mass_refine
        from scripts.depth_supervision.utils import digest
        root=Path(self.temp.name)/'mass';(root/'encoder').mkdir(parents=True)
        (root/'encoder/frozen.pt').write_bytes(self.path.read_bytes())
        model=PriorTrainingModel();model.encoder.load_state_dict(self.loss.encoder.state_dict())
        identity={'unit':'refined'}
        save_checkpoint(root/'encoder/refinement_latest.pt',dict(step=300,identity=identity,model=model.state_dict()))
        class Items:
            def __init__(self):self.items=[example(12)[0] for _ in range(10)]
            def __getitem__(self,i):return self.items[i]
        class Context:
            def get(self,item):return torch.ones(1,2,12,12,12)
        p=dict(identity_sha='mass-unit',reused_refinement={'encoder_sha256':digest(self.path),'identity':identity,'steps':300},
            splits={'5pct':{'train':[str(i) for i in range(10)]}},
            mass_refinement=dict(seed=29,steps=6,lr=.0003,representation_weight=.1,mass_weight=2.,swap_margin=.1))
        with patch.object(mass_refine,'load_training',return_value=(Items(),Context())):
            mass_refine.run_mass_refinement(root,p,torch.device('cpu'))
        self.assertEqual(digest(root/'encoder/before_mass_refinement.pt'),digest(self.path))
        record=json.loads((root/'encoder/mass_refinement_completion.json').read_text())
        self.assertEqual(record['steps'],6);self.assertFalse(record['validation_cases_used'])
        self.assertNotEqual(record['encoder_sha256'],digest(self.path))

    def test_exact_resume_before_and_after_coefficient_calibration(self):
        def patients():
            generator=torch.Generator().manual_seed(78)
            class Items:
                def __init__(self):
                    self.items=[]
                    for i in range(2):
                        item,context=example(12)
                        item['image']=torch.randn(1,12,12,12,generator=generator);item['case_name']=str(i)
                        self.items.append(item)
                def __len__(self):return len(self.items)
                def __getitem__(self,i):return self.items[i]
            return Items(),Items()
        class Context:
            def __init__(self,*args):pass
            def get(self,item):return torch.ones(1,2,*item['image'].shape[1:])
        def evaluate(model,items,device,**kwargs):
            model.eval()
            with torch.no_grad():score=float(model(items[0]['image'][None]).softmax(1)[:,1].mean())
            if kwargs.get('predictions_path'):
                with open(kwargs['predictions_path'],'wb') as f:np.savez(f,prediction=np.zeros((1,),np.uint8))
            return dict(summary=dict(macro_dice=score,foreground_dice=score,missed=1,extra=2,swaps=3),cases=[])
        p=dict(identity_sha='unit',seed=0,learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,max_epochs=5,
            early_stopping=dict(min_epochs=5,patience=8,min_delta=.0005),geometry={},
            splits={'5pct':dict(train=['0','1'],val=['2','3'])},calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5))
        from scripts.depth_supervision.utils import digest
        roots=[]
        for name in ('full','before','after'):
            root=Path(self.temp.name)/name;(root/'encoder').mkdir(parents=True)
            (root/'encoder/frozen.pt').write_bytes(self.path.read_bytes());roots.append(root)
        with patch.object(train,'verify_gate',return_value={'encoder_sha256':digest(self.path)}), \
             patch.object(train,'load_data',side_effect=lambda p:patients()), \
             patch.object(train,'model_factory',side_effect=lambda device:torch.nn.Conv3d(1,3,1)), \
             patch.object(train,'ContextCache',Context),patch.object(train,'evaluate',side_effect=evaluate), \
             patch.object(train,'GeometryCache',return_value=None):
            train.run(roots[0],p,torch.device('cpu'))
            for root,stop in zip(roots[1:],(2,3)):
                train.run(root,p,torch.device('cpu'),stop_after=stop)
                train.run(root,p,torch.device('cpu'))
        original=torch.load(roots[0]/'runs/5pct_dice_learned_s0/best.pt',weights_only=False)
        for root in roots[1:]:
            actual=torch.load(root/'runs/5pct_dice_learned_s0/best.pt',weights_only=False)
            self.assertEqual(original['epoch'],actual['epoch'])
            for k,v in original['model'].items():torch.testing.assert_close(v,actual['model'][k],rtol=0,atol=0)
            import csv
            read=lambda r:list(csv.DictReader((r/'runs/5pct_dice_learned_s0/metrics.csv').open()))
            for a,b in zip(read(roots[0]),read(root)):
                for k in a:
                    if k!='epoch_seconds':self.assertEqual(a[k],b[k])


def swin_smoke():
    from scripts.depth_supervision.train import model_factory
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'frozen.pt';frozen(path)
        criterion=LearnedLoss(path,torch.device('cpu'))
        torch.manual_seed(0);model=model_factory(torch.device('cpu'))
        image=torch.randn(1,1,64,64,64)
        target=torch.zeros(1,1,64,64,64,dtype=torch.long);target[:,:,20:32,20:40,20:40]=1;target[:,:,32:44,20:40,20:40]=2
        context=torch.ones(1,2,64,64,64)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
        value=criterion(model(image),target,context)
        value.backward()
        norm=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
        assert np.isfinite(norm) and norm>0
        optimizer.step()
        assert all(p.grad is None for p in criterion.parameters())
        return dict(passed=True,shape=[64,64,64],loss_only_native_gradient_norm=norm,encoder_frozen=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--swin-smoke',action='store_true');a=parser.parse_args()
    torch.set_num_threads(4)
    if a.swin_smoke:result=swin_smoke()
    else:
        runner=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        result=dict(passed=runner.wasSuccessful(),tests=runner.testsRun)
    atomic_json(a.output,result)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
