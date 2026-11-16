"""CPU contract checks; these do not establish real-model effectiveness."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from scripts.colab_hfh.medsam_joint_constraints import (
    DEFAULT, GradientAccumulator, active_terms, auxiliary_fields,
    calibrated_weights, calibration_summary, guarded_training_source,
)
from scripts.colab_hfh.medsam_joint_constraints_worker import worker_source
from scripts.medsam3_supervised_constraints import semantic_log_fields, sampled_losses, VolumeSupport
from scripts.depth_supervision.geometry import signed_distance


class JointConstraintTests(unittest.TestCase):
    def outputs(self, shape=(9,9)):
        torch.manual_seed(37)
        return dict(pred_masks=torch.randn(1,3,*shape,requires_grad=True),
                    pred_logits=torch.randn(1,3,1,requires_grad=True),
                    presence_logit_dec=torch.randn(1,1,requires_grad=True))

    def test_forward_and_winners_unchanged_direct_confidence_gradients_removed(self):
        out=self.outputs()
        ref=semantic_log_fields(out,(9,9)); routed=auxiliary_fields(out,(9,9))
        for a,b in zip(ref,routed):self.assertTrue(torch.equal(a,b))
        gm,gs,gp=torch.autograd.grad(-routed[0].mean(),list(out.values()),allow_unused=True,retain_graph=True)
        self.assertGreater(float(gm.abs().sum()),0)
        self.assertIsNone(gs);self.assertIsNone(gp)
        # The original native/output path still retains its confidence graph.
        gg=torch.autograd.grad(-ref[0].mean(),list(out.values()),allow_unused=True)
        self.assertGreater(float(gg[1].abs().sum()),0);self.assertGreater(float(gg[2].abs().sum()),0)

    def test_all_four_losses_keep_forward_values_and_route_gradients(self):
        fg=torch.zeros(5,9,9,dtype=torch.bool);fg[1:4,3:6,3:6]=True
        support=VolumeSupport.build(fg)
        phi=torch.from_numpy(signed_distance(fg.numpy(),(2.,1.,1.),8.)/8.)
        weights=torch.ones_like(phi)/phi.numel()
        a,b=self.outputs(),self.outputs()
        def losses(field):
            lp,lq=field(a,(9,9));next_lp,_=field(b,(9,9))
            bands,edge=sampled_losses(lp.exp(),next_lp.exp(),support,2,log_probability=lp,log_complement=lq)
            sdf=((lp.exp()-fg[2].float())*phi[2]).mean()
            depth=-torch.where(fg[2],lp,lq).mul(weights[2]).sum()*len(fg)
            return bands,edge,sdf,depth
        original=losses(semantic_log_fields);routed=losses(auxiliary_fields)
        for i,(x,y) in enumerate(zip(original,routed)):
            self.assertTrue(torch.equal(x,y),str(i))
            gm,gs,gp=torch.autograd.grad(y,list(a.values()),allow_unused=True,retain_graph=True)
            self.assertTrue(torch.isfinite(gm).all());self.assertGreater(float(gm.abs().sum()),0)
            self.assertIsNone(gs);self.assertIsNone(gp)

    def test_sdf_covers_distant_fp_that_band_does_not(self):
        fg=torch.zeros(7,13,13,dtype=torch.bool);fg[2:5,5:8,5:8]=True
        support=VolumeSupport.build(fg);z=3
        logits=torch.zeros(13,13,requires_grad=True)
        lp=torch.nn.functional.logsigmoid(logits);lq=torch.nn.functional.logsigmoid(-logits)
        band,_=sampled_losses(lp.exp(),lp.exp(),support,z,log_probability=lp,log_complement=lq)
        phi=torch.from_numpy(signed_distance(fg.numpy(),(2.,1.,1.),8.)/8.)
        sdf=((lp.exp()-fg[z].float())*phi[z]).mean()
        gb=torch.autograd.grad(band,logits,retain_graph=True)[0]
        gs=torch.autograd.grad(sdf,logits)[0]
        self.assertEqual(float(gb[0,0]),0)
        self.assertGreater(float(gs[0,0]),0)  # descent suppresses distant FP
        self.assertLess(float(gs[6,6]),0)     # descent recovers foreground

    def test_edge_does_not_wrap_anatomical_faces(self):
        fg=torch.zeros(3,7,7,dtype=torch.bool);fg[:,2:5,2:5]=True
        support=VolumeSupport.build(fg)
        p=torch.full((7,7),.4);q=torch.full((7,7),.9)
        a=sampled_losses(p,p,support,2)[1];b=sampled_losses(p,q,support,2)[1]
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(sampled_losses(p,p,support,0)[1],sampled_losses(p,q,support,0)[1]))

    def test_joint_calibration_uses_cross_terms_and_shared_budget(self):
        g=np.array([[np.outer([10.,1.,1.,1.,1.],[10.,1.,1.,1.,1.])]*200])[0]
        w=calibrated_weights(g,DEFAULT)
        np.testing.assert_allclose(w,[.025]*4)
        joint=np.sqrt(w@g[0,1:,1:]@w/g[0,0,0])
        self.assertAlmostEqual(joint,.01)
        # Independent full budgets would yield 4%, not the declared 1%.
        self.assertLess(joint,.04)

    def test_zero_gradients_preserved_and_degenerate_terms_rejected(self):
        g=np.zeros((200,2,2));g[:,0,0]=100;g[150:,1,1]=1
        self.assertAlmostEqual(calibrated_weights(g,DEFAULT)[0],.5)
        with self.assertRaises(ValueError):calibrated_weights(np.zeros((200,2,2)),DEFAULT)
        invalid=g.copy();invalid[:,1,1]=-1
        with self.assertRaises(ValueError):calibrated_weights(invalid,DEFAULT)
        invalid=g.copy();invalid[:,0,1]=invalid[:,1,0]=1000
        with self.assertRaises(ValueError):calibrated_weights(invalid,DEFAULT)

    def test_subject_stability_and_category_statistics(self):
        rows=[]
        for s in range(5):
            for i in range(40):
                rows.append(dict(subject=str(s),slab_category='empty' if i<20 else 'foreground',
                    gram=np.diag([100.,0. if i<20 else (s+1)**2]).tolist()))
        r=calibration_summary(rows,('bands',),DEFAULT)
        self.assertEqual(r['statistics']['terms']['bands']['zero_gradients'],100)
        self.assertEqual(len(r['stability']['leave_one_subject_out']),5)
        self.assertFalse(r['stability']['used_to_select_weights'])
        self.assertEqual(r,calibration_summary(deepcopy(rows),('bands',),DEFAULT))

    def test_cap_applies_to_accumulated_joint_gradient_per_block(self):
        a=torch.nn.Parameter(torch.tensor([1.,1.]));b=torch.nn.Parameter(torch.tensor([1.]))
        mix=GradientAccumulator([a,b],['block.lora_A','other.lora_B'],.05)
        # Native cancels in the second block across microbatches; its auxiliary
        # must therefore be zero even though each micro has a native gradient.
        mix.add(a.sum()/2+b.sum(),100*a.sum()+100*b.sum())
        mix.add(a.sum()/2-b.sum(),100*a.sum()+100*b.sum())
        report=mix.finish()
        torch.testing.assert_close(a.grad,torch.tensor([1.05,1.05]))
        torch.testing.assert_close(b.grad,torch.tensor([0.]))
        self.assertLessEqual(report['applied_joint_ratio'],.05+1e-12)
        self.assertEqual(report['blocks']['other']['scale'],0.)

    def test_a_b_share_block_and_native_gradient_remains_when_auxiliary_zero(self):
        a=torch.nn.Parameter(torch.tensor([2.]));b=torch.nn.Parameter(torch.tensor([3.]))
        mix=GradientAccumulator([a,b],['mod.lora_A','mod.lora_B'],.05)
        mix.add(a.square().sum()+b.square().sum(),0.*(a.sum()+b.sum()))
        report=mix.finish()
        self.assertEqual(len(report['blocks']),1)
        self.assertTrue(torch.equal(a.grad,torch.tensor([4.])))
        self.assertTrue(torch.equal(b.grad,torch.tensor([6.])))

    def test_finite_gradient_gate(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        with self.assertRaises(FloatingPointError):
            GradientAccumulator([p],['m.lora_A'],.05).add(p.sum(),p.sum()*float('nan'))

    def test_training_bridge_preserves_checkpoint_and_scalar_logging(self):
        import inspect
        from scripts.cross_dataset_campaign.foundation import train
        from scripts.colab_hfh.medsam_intensity_augmentation import training_source
        source=guarded_training_source(inspect.getsource(train))
        compile(training_source(source),'<joint-training>','exec')
        self.assertIn("optimizer.load_state_dict(state['optimizer'])",source)
        self.assertIn('optimizer.step()',source)
        self.assertIn('gradient_control=mix_report',source)
        compile(worker_source(),'<joint-worker>','exec')
        self.assertEqual(active_terms('dice_bands_edge_sdf_depth'),('bands','edge','sdf','depth'))
        with self.assertRaises(ValueError):active_terms('dice')

    def test_interrupted_optimizer_resume_matches_uninterrupted(self):
        import hashlib
        import inspect
        from pathlib import Path
        import tempfile
        from scripts.cross_dataset_campaign import foundation as engine
        from scripts.colab_hfh.medsam_joint_constraints import gradient_accumulator
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A=torch.nn.Parameter(torch.tensor([.3,-.2]))
                self.base=torch.nn.Parameter(torch.tensor([1.]),requires_grad=False)
        class Slabs:
            def schedule(self,p):return [('train',z) for z in range(8)]
        def losses(model,criterion,slabs,case,z):
            x=model.lora_A
            return dict(native=(x-torch.tensor([1.,-1.])).square().sum(),
                bands=(x[0]+x[1]-.1).square(),edge=(x[0]-x[1]-.3).square(),
                sdf=x.sigmoid().sum(),depth=(x-.5).square().sum())
        def digest(model,adapter):
            data=b''.join(p.detach().numpy().tobytes() for n,p in model.named_parameters()
                          if n.startswith('lora_')==adapter)
            return hashlib.sha256(data).hexdigest()
        namespace={**engine.__dict__,'forward_slab':losses,'parameter_digest':digest,
                   'is_adapter':lambda n:n.startswith('lora_'),'hfh_gradient_accumulator':gradient_accumulator}
        exec(compile(guarded_training_source(inspect.getsource(engine.train)),'<test-training>','exec'),namespace)
        protocol=dict(updates=4,accumulation=2,seed=0,warmup_updates=1,learning_rate=.001,
            final_learning_rate=.0001,constraint_warmup_updates=2,joint_constraints=DEFAULT)
        calibration=dict(weights={k:.1 for k in ('bands','edge','sdf','depth')})
        with tempfile.TemporaryDirectory() as tmp:
            result=[]
            for interrupted in (False,True):
                path=Path(tmp)/str(interrupted);path.mkdir();engine.write(path/'protocol.json',protocol)
                calls=[0]
                def stopping():
                    calls[0]+=1
                    return interrupted and calls[0]==3
                namespace['stopping']=stopping
                model=Model()
                if interrupted:
                    with self.assertRaises(SystemExit):
                        namespace['train'](model,None,[model.lora_A],Slabs(),protocol,calibration,'all',path)
                    model=Model();namespace['stopping']=lambda:False
                namespace['train'](model,None,[model.lora_A],Slabs(),protocol,calibration,'all',path)
                result.append(torch.load(path/'checkpoint_latest.pt',weights_only=False))
            self.assertEqual(result[0]['history'],result[1]['history'])
            self.assertTrue(torch.equal(result[0]['adapter']['lora_A'],result[1]['adapter']['lora_A']))


if __name__=='__main__':unittest.main()
