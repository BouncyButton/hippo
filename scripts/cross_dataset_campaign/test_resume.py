"""An actual interrupted optimizer trajectory must match uninterrupted training."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.cross_dataset_campaign import capped_core as core
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.conventional import evaluate


class Resume(unittest.TestCase):
    def test_foundation_resume_preserves_adapter_optimizer_and_sampling(self):
        from scripts.cross_dataset_campaign import foundation as f
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__();self.block=torch.nn.Module()
                self.block.register_parameter('lora_A',torch.nn.Parameter(torch.tensor([.3,.7])))
                self.block.register_parameter('lora_B',torch.nn.Parameter(torch.tensor([.1,.2])))
                self.register_parameter('frozen',torch.nn.Parameter(torch.tensor([2.]),requires_grad=False))
        class Slabs:
            indices={'case':[0,1,2]}
            def schedule(self,p):return [('case',i%3) for i in range(p['updates']*p['accumulation'])]
        calls={'count':0,'pause':False}
        def forward(model,criterion,slabs,case,anchor):
            calls['count']+=1
            noise=torch.rand(2)
            loss=((model.block.lora_A*model.block.lora_B-noise-.1*anchor)**2).mean()
            return dict(native=loss,bands=loss*.3,edge=loss*.2,sdf=loss*.1,depth=loss*.4)
        p=dict(seed=0,updates=6,accumulation=2,learning_rate=.01,final_learning_rate=.001,warmup_updates=2,constraint_warmup_updates=2)
        with tempfile.TemporaryDirectory() as temporary,patch.object(f,'forward_slab',forward),patch.object(f,'stopping',lambda:calls['pause'] and calls['count']==6):
            root=Path(temporary)
            for name in ('full','resumed'):
                write(root/name/'protocol.json',p)
            model=Adapter()
            f.train(model,None,[v for v in model.parameters() if v.requires_grad],Slabs(),p,dict(weights={'bands':.1}),'dice_bands',root/'full')
            model=Adapter();calls.update(count=0,pause=True)
            with self.assertRaises(SystemExit) as result:
                f.train(model,None,[v for v in model.parameters() if v.requires_grad],Slabs(),p,dict(weights={'bands':.1}),'dice_bands',root/'resumed')
            self.assertEqual(result.exception.code,75)
            model=Adapter();calls['pause']=False
            f.train(model,None,[v for v in model.parameters() if v.requires_grad],Slabs(),p,dict(weights={'bands':.1}),'dice_bands',root/'resumed')
            full=torch.load(root/'full/checkpoint_latest.pt',weights_only=False)
            resumed=torch.load(root/'resumed/checkpoint_latest.pt',weights_only=False)
            for name in full['adapter']:self.assertTrue(torch.equal(full['adapter'][name],resumed['adapter'][name]))
            self.assertEqual(full['history'],resumed['history'])
            for key,state in full['optimizer']['state'].items():
                for name,value in state.items():self.assertTrue(torch.equal(value,resumed['optimizer']['state'][key][name]))

    def test_checkpoint_boundary_preserves_optimizer_rng_and_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);protocol_dir=root/'protocol';protocol_dir.mkdir()
            data=root/'data';data.write_bytes(b'frozen fixture')
            cases=[f'case{i}' for i in range(10)]
            write(protocol_dir/'splits_10patients.json',[dict(train=cases,val=['validation'])])
            write(protocol_dir/'nnunet_architecture.json',{})
            schedule=dict(max_steps=6,prefix_updates=0,sdf_prefix_updates=0,
                baseline_flops_per_update=10,early_flops_per_update=10,flops_per_update=10,
                calibration_flops=0,constraint_warmup_updates=1,sdf_warmup_updates=1)
            p=dict(models={'unet':dict(parameters=4,objectives={'dice':schedule})},budget_flops=60,
                budget_scope='optimization_only',seeds=[0],folds=[0],augmentation='none',train_count=10,
                splits_sha256=digest(protocol_dir/'splits_10patients.json'),
                nnunet_architecture_sha256=digest(protocol_dir/'nnunet_architecture.json'),pkl_sha256=digest(data),
                model_identity='tiny',lr_horizon_updates=6,precision='cpu',initial_lrs={'unet':.01},
                constraint_warmup_updates=1,checkpoint_selection='final',validation_milestones=2,
                native_geometry={name:dict(model_affine=torch.eye(4).tolist()) for name in cases})
            write(protocol_dir/'protocol.json',p)
            class Patients:
                def __init__(self,frame,names):
                    generator=torch.Generator().manual_seed(99)
                    self.items=[dict(image=torch.randn(1,8,8,8,generator=generator),
                        label=torch.randint(0,2,(1,8,8,8),generator=generator),case_name=name) for name in names]
                def __len__(self):return len(self.items)
                def __getitem__(self,index):return self.items[index]
            calls={'steps':0,'pause':False}
            class Adam(torch.optim.AdamW):
                def step(self,*args,**kwargs):
                    value=super().step(*args,**kwargs);calls['steps']+=1;return value
            def loss_optimizer(name,model,protocol):
                return None,Adam(model.parameters(),lr=.01)
            def model(name,architecture):
                return torch.nn.Sequential(torch.nn.Conv3d(1,2,1),torch.nn.Dropout3d(.25))
            with ExitStack() as stack:
                for name,value in dict(build_model=model,build_loss_and_optimizer=loss_optimizer,
                    PreparedPatients=Patients,_load_pkl_dataframe=lambda p:None,
                    evaluate=lambda m,d,v,a:evaluate(m,d,v,a,['background','foreground']),
                    validation_steps=lambda maximum,count:[maximum]).items():
                    stack.enter_context(patch.object(core.base,name,value))
                stack.enter_context(patch.object(core,'stopping',lambda:calls['pause'] and calls['steps']==3))
                arguments=dict(protocol_dir=protocol_dir,model='unet',objective='dice',fold=0,seed=0,
                    pkl=data,device='cpu',threads=1,resume=True,smoke_steps=6)
                core.run(SimpleNamespace(output=root/'full',**arguments))
                calls.update(steps=0,pause=True)
                with self.assertRaises(SystemExit) as result:
                    core.run(SimpleNamespace(output=root/'resumed',**arguments))
                self.assertEqual(result.exception.code,75)
                checkpoint=torch.load(root/'resumed/checkpoint_latest.pt',weights_only=False)
                self.assertEqual(checkpoint['step'],3)
                self.assertEqual(checkpoint['charged_attempts'],3)
                calls['pause']=False
                core.run(SimpleNamespace(output=root/'resumed',**arguments))
            full=torch.load(root/'full/checkpoint_latest.pt',weights_only=False)
            resumed=torch.load(root/'resumed/checkpoint_latest.pt',weights_only=False)
            self.assertEqual(full['permutation'],resumed['permutation'])
            self.assertEqual(full['cursor'],resumed['cursor'])
            for name in full['model']:
                self.assertTrue(torch.equal(full['model'][name],resumed['model'][name]),name)
            for key,state in full['optimizer']['state'].items():
                for name,value in state.items():
                    self.assertTrue(torch.equal(value,resumed['optimizer']['state'][key][name]))
            self.assertEqual(read(root/'resumed/completion.json')['discarded_attempts'],0)


if __name__=='__main__':unittest.main()
