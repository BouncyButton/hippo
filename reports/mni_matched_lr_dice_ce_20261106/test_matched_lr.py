"""Check native baseline loss dispatch, budget accounting and resumable SGD."""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import torch
from scripts.unetr_subfields import matched_lr_runtime as runtime
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest

PARENT=Path(__file__).resolve().parents[2]/'experiments/mni_matched_lr_dice_ce_20261106'

class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv3d(1,4,1)
    def forward(self,x):
        z=self.conv(x)
        return [z]+[torch.nn.functional.avg_pool3d(z,kernel_size=2**i) for i in (1,2,3)]

class BaselineTests(unittest.TestCase):
    def test_three_native_baselines_budget_and_sgd_resume(self):
        torch.set_num_threads(2)
        for identity,spec in runtime.MODELS.items():
            with self.subTest(model=identity),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);folder=root/'protocol';shutil.copytree(PARENT/'protocols'/identity,folder)
                p=read(folder/'protocol.json');alias=spec['alias'];split=read(folder/'splits_10patients.json')[0]
                write(folder/'splits_10patients.json',[dict(train=split['train'][:1],val=split['val'][:1])])
                schedule=deepcopy(p['models'][alias]['objectives']['dice'])
                for key in ('baseline_flops_per_update','early_flops_per_update','flops_per_update'):schedule[key]=17
                schedule['max_steps']=2
                p.update(dataset='MNI',model_identity=identity,trainer_alias=alias,initial_lrs={alias:spec['lr']},
                    requested_arm_order=['dice'],augmentation='none',training_loss=spec['loss'],
                    models={alias:dict(parameters=8,objectives={'dice':schedule})},budget_flops=34,
                    lr_horizon_updates=2,validation_target_flops=[17,34],train_count=1,folds=[0],seeds=[0],
                    splits_sha256=digest(folder/'splits_10patients.json'))
                write(folder/'protocol.json',p)
                args=argparse.Namespace(model=alias,objective='dice',fold=0,seed=0,pkl=PARENT/'data/MNI.pkl',
                    protocol_dir=folder,output=root/'resumed',device='cpu',threads=2,resume=True,smoke_steps=0)
                factory=lambda *_:TinyNative() if identity=='nnunet' else torch.nn.Conv3d(1,4,1)
                with patch.object(runtime,'build_model',factory):
                    if True:
                        with patch.object(runtime.core.trainer,'stopping',side_effect=[False,True]):
                            with self.assertRaises(SystemExit) as stopped:runtime.train(args)
                            self.assertEqual(stopped.exception.code,75)
                    with patch.object(runtime.core.trainer,'stopping',return_value=False):
                        runtime.train(args)
                        if True:
                            direct=deepcopy(args);direct.output=root/'direct';runtime.train(direct)
                            a=torch.load(args.output/'checkpoint_latest.pt',map_location='cpu')['model'];b=torch.load(direct.output/'checkpoint_latest.pt',map_location='cpu')['model']
                            for key in a:self.assertTrue(torch.equal(a[key],b[key]))
                done=read(args.output/'completion.json');self.assertEqual(done['updates'],2);self.assertEqual(done['consumed_training_flops'],34)
                self.assertEqual(done['discarded_attempts'],0);self.assertEqual(read(args.output/'calibration.json')['status'],'not_required')
                cfg=read(args.output/'config.json');self.assertEqual(cfg['augmentation'],'none')
                if identity=='nnunet':self.assertIn('Dice+CE',cfg['native_loss']);self.assertEqual(cfg['nnunet_sgd_momentum'],.95)

    def test_swin_loss_and_gradients_are_dice_plus_ce(self):
        from monai.losses import DiceCELoss
        protocol=read(PARENT/'protocols/swinunetr/protocol.json');trainer=runtime.bind(protocol)
        torch.manual_seed(3);z=torch.randn(1,4,4,4,4,requires_grad=True);y=torch.randint(0,4,(1,1,4,4,4))
        loss=trainer.training_loss(z,y,'dice',{},.2)
        reference=DiceCELoss(to_onehot_y=True,softmax=True,include_background=True,lambda_dice=1.,lambda_ce=1.)(z,y)
        self.assertTrue(torch.allclose(loss,reference))
        a=torch.autograd.grad(loss,z,retain_graph=True)[0];b=torch.autograd.grad(reference,z)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-7))

    def test_schedule_uses_actual_budget_and_endpoints(self):
        import math
        for identity in runtime.MODELS:
            p=read(PARENT/'protocols'/identity/'protocol.json');runtime.bind(p)
            n=p['models'][identity]['objectives']['dice']['max_steps'];initial=p['initial_lrs'][identity]
            values=[runtime.core.base.learning_rate(identity,i,n,initial) for i in range(n)]
            self.assertEqual(p['lr_horizon_updates'],n)
            if identity=='nnunet':
                self.assertEqual(values[0],initial)
                self.assertAlmostEqual(values[-1],initial*(1/n)**.9)
                self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
                self.assertEqual(runtime.core.base.learning_rate(identity,n,n,initial),0)
            else:
                warmup=math.ceil(.05*n);self.assertAlmostEqual(values[0],initial/warmup)
                self.assertAlmostEqual(values[warmup-1],initial);self.assertAlmostEqual(values[-1],initial*.01)
                self.assertTrue(all(a>=b for a,b in zip(values[warmup:],values[warmup+1:])))
            bad=deepcopy(p);bad['lr_horizon_updates']=n+1
            with self.assertRaises(ValueError):runtime.bind(bad)

if __name__=='__main__':unittest.main()
