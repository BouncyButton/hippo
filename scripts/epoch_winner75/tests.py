"""Schedule/resume regression checks and real-network GPU preflight."""
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.capped_winner.objectives import ARM, calibration_case, select_weights, training_loss
from scripts.capped_winner.geometry import make_targets
from scripts.depth_supervision.utils import atomic_json, seed_all, stopping_state
from . import train as trainer


def tiny_loss(outputs, labels, arm, weights, scale, **kwargs):
    native = torch.nn.functional.cross_entropy(outputs, labels[:,0].long())
    return native + (scale*.1*outputs.square().mean() if arm==ARM else 0.)


def tiny_evaluate(model, dataset, device, amp):
    model.eval()
    with torch.no_grad():
        score=float(model(dataset[0]['image'][None]).softmax(1)[:,1].mean())
    return dict(summary=dict(macro_hard_dice=score))


class EpochPolicyTests(unittest.TestCase):
    def test_floor_and_patience(self):
        flat=[dict(epoch=i,validation_dice=.7) for i in range(1,26)]
        self.assertFalse(stopping_state(flat[:24])['triggered'])
        self.assertTrue(stopping_state(flat)['triggered'])
        flat[-1]['validation_dice']=.7006
        self.assertFalse(stopping_state(flat)['triggered'])
        self.assertEqual(stopping_state(flat)['reference_epoch'],25)

    def test_lr_policy(self):
        param=torch.nn.Parameter(torch.ones(1))
        opt=torch.optim.SGD([param],lr=.01)
        schedule=torch.optim.lr_scheduler.StepLR(opt,20,.5)
        rates=[]
        for _ in range(75):
            rates.append(opt.param_groups[0]['lr']);opt.step();schedule.step()
        self.assertEqual(rates,[.01]*20+[.005]*20+[.0025]*20+[.00125]*15)

    def test_paired_prefix_and_exact_resume(self):
        p=dict(binding='test',max_epochs=8,early_stopping=dict(patience=8,min_epochs=25,min_delta=.0005),
               learning_rate_policy='StepLR20',native_geometry={'case':dict(model_affine=[],native_shape=[])},
               depth_geometry={})
        data=[dict(case_name='case',image=torch.ones(1,4,4,4),label=torch.ones(1,4,4,4,dtype=torch.long))]
        def run(path,arm,stop_after=None):
            seed_all(0)
            model=torch.nn.Sequential(torch.nn.Conv3d(1,3,1),torch.nn.Dropout3d(.2))
            opt=torch.optim.AdamW(model.parameters(),lr=.01,foreach=False)
            entry=dict(id=arm,model='basic_unet16',objective=arm,seed=0,fold=0)
            return trainer.run_epochs(p,entry,model,None,opt,data,data,path,torch.device('cpu'),
                                      evaluator=tiny_evaluate,stop_after=stop_after)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(trainer,'training_loss',tiny_loss), \
             patch.object(trainer,'make_targets',return_value={}), \
             patch.object(trainer,'calibrate',return_value=dict(weights={'test':1.})):
            root=Path(directory)
            baseline=run(root/'baseline','dice')
            continuous=run(root/'continuous',ARM)
            run(root/'resumed',ARM,stop_after=6)
            resumed=run(root/'resumed',ARM)
            self.assertEqual(baseline['initialization_sha256'],continuous['initialization_sha256'])
            self.assertEqual(baseline['warmup_sha256'],continuous['warmup_sha256'])
            a=torch.load(root/'continuous/latest.pt',weights_only=False)
            b=torch.load(root/'resumed/latest.pt',weights_only=False)
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]))
            self.assertEqual(continuous['best_dice'],resumed['best_dice'])
            self.assertEqual(continuous['final_dice'],resumed['final_dice'])
            self.assertEqual([r['validation_dice'] for r in a['history']],
                             [r['validation_dice'] for r in b['history']])
            # Selected scores and last-epoch scores are tracked separately.
            self.assertEqual(continuous['best_dice'],max(r['validation_dice'] for r in a['history']))


def gpu_preflight(p,pkl,output):
    """One real update with each objective/model on an authorized training case."""
    from scripts.campaign90 import build_model, MODELS
    from scripts.depth_supervision.train import Patients, autocast
    from scripts.capped_winner.core import model_digest
    torch.set_num_threads(4)
    df=trainer.base._load_pkl_dataframe(pkl)
    patients=Patients(df,p['split']['train'],p['native_geometry'])
    del df
    device=torch.device('cuda')
    # Verify targets for all ten training volumes before the long run.
    for item in patients:
        geo=p['native_geometry'][item['case_name']]
        make_targets(item['label'][None],geo['model_affine'],geo['model_affine'],
                     geo['native_shape'],p['depth_geometry'],torch.device('cpu'))
    item=patients[0];labels=item['label'][None].to(device)
    geo=p['native_geometry'][item['case_name']]
    target=make_targets(labels,geo['model_affine'],geo['model_affine'],
                        geo['native_shape'],p['depth_geometry'],device)
    results=[]
    for identity in ['nnunet','unetr_vit_b16','basic_unet16']:
        seed_all(0)
        spec=MODELS[identity]
        model=build_model(identity,p['nnunet_architecture']).to(device)
        assert sum(v.numel() for v in model.parameters())==spec['parameters']
        options=dict(p,initial_lrs={spec['alias']:spec['initial_lr']})
        criterion,opt=trainer.base.build_loss_and_optimizer(spec['alias'],model,options)
        criterion=criterion if identity=='nnunet' else None
        anchor=model_digest(model)
        for arm in ['dice',ARM]:
            opt.zero_grad(set_to_none=True)
            model.train()
            with autocast(device):raw=model(item['image'][None].to(device))
            if arm==ARM:
                record=calibration_case(raw,labels,ARM,criterion=criterion,depth_weights=target)
                weights,_=select_weights([record],ARM)
                assert all(torch.isfinite(torch.tensor(v)) and v>=0 for v in weights.values())
            else:weights={}
            loss=training_loss(raw,labels,arm,weights,1.,criterion=criterion,depth_weights=target)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            if identity=='nnunet':torch.nn.utils.clip_grad_norm_(model.parameters(),12.,foreach=False)
            opt.step()
        assert model_digest(model)!=anchor
        results.append(dict(model=identity,parameters=spec['parameters'],baseline_update=True,
                            calibrated_constraint_update=True,finite_gradients=True))
        print(__import__('json').dumps(dict(stage='gpu_preflight',**results[-1])),flush=True)
        del model,opt,raw,criterion,loss
        gc.collect();torch.cuda.empty_cache()
    atomic_json(output,dict(status='passed',training_geometry_cases=10,models=results))


if __name__=='__main__':
    unittest.main()
