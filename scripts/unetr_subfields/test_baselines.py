"""Check native baseline loss dispatch, budget accounting and resumable SGD."""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import torch
from scripts.unetr_subfields import baseline_runtime as runtime
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest

PARENT=Path(__file__).resolve().parents[2]/'experiments/unetr_subfields_geo_recipe_seed_search_20261106'

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
                root=Path(directory);folder=root/'protocol';shutil.copytree(PARENT/'protocols/MNI',folder)
                p=read(folder/'protocol.json');alias=spec['alias'];split=read(folder/'splits_10patients.json')[0]
                write(folder/'splits_10patients.json',[dict(train=split['train'][:1],val=split['val'][:1])])
                schedule=deepcopy(p['models']['unet']['objectives']['dice_ce'])
                for key in ('baseline_flops_per_update','early_flops_per_update','flops_per_update'):schedule[key]=17
                schedule['max_steps']=2
                p.update(dataset='MNI',model_identity=identity,trainer_alias=alias,initial_lrs={alias:spec['lr']},
                    requested_arm_order=['dice'],augmentation='none',training_loss=spec['loss'],
                    models={alias:dict(parameters=8,objectives={'dice':schedule})},budget_flops=34,
                    validation_target_flops=[17,34],train_count=1,folds=[0],seeds=[0],
                    splits_sha256=digest(folder/'splits_10patients.json'))
                write(folder/'protocol.json',p)
                args=argparse.Namespace(model=alias,objective='dice',fold=0,seed=0,pkl=PARENT/'data/MNI.pkl',
                    protocol_dir=folder,output=root/'resumed',device='cpu',threads=2,resume=True,smoke_steps=0)
                factory=lambda *_:TinyNative() if identity=='nnunet' else torch.nn.Conv3d(1,4,1)
                with patch.object(runtime,'build_model',factory):
                    if identity=='nnunet':
                        with patch.object(runtime.core.trainer,'stopping',side_effect=[False,True]):
                            with self.assertRaises(SystemExit) as stopped:runtime.train(args)
                            self.assertEqual(stopped.exception.code,75)
                    with patch.object(runtime.core.trainer,'stopping',return_value=False):
                        runtime.train(args)
                        if identity=='nnunet':
                            direct=deepcopy(args);direct.output=root/'direct';runtime.train(direct)
                            a=torch.load(args.output/'checkpoint_latest.pt',map_location='cpu')['model'];b=torch.load(direct.output/'checkpoint_latest.pt',map_location='cpu')['model']
                            for key in a:self.assertTrue(torch.equal(a[key],b[key]))
                done=read(args.output/'completion.json');self.assertEqual(done['updates'],2);self.assertEqual(done['consumed_training_flops'],34)
                self.assertEqual(done['discarded_attempts'],0);self.assertEqual(read(args.output/'calibration.json')['status'],'not_required')
                cfg=read(args.output/'config.json');self.assertEqual(cfg['augmentation'],'none')
                if identity=='nnunet':self.assertIn('Dice+CE',cfg['native_loss']);self.assertEqual(cfg['nnunet_sgd_momentum'],.95)

if __name__=='__main__':unittest.main()
