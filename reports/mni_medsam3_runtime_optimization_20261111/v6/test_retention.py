import copy
import sys
import types
import unittest
from unittest.mock import patch
import torch
import torch.utils.checkpoint as checkpoint
from selective_retention import retention


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(4,4)
        self.dropout=torch.nn.Dropout(.1)
    def forward(self,x):
        return x+self.dropout(self.linear(x))*.05


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Module()
        self.backbone.language_backbone=torch.nn.ModuleList([Block() for _ in range(3)])
        self.backbone.vision_backbone=torch.nn.Module()
        self.backbone.vision_backbone.trunk=torch.nn.Module()
        self.backbone.vision_backbone.trunk.blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        self.backbone.vision_backbone.trunk.use_act_checkpoint=True
    def forward(self,x):
        text=sys.modules['sam3.model.text_encoder_ve']
        for block in self.backbone.language_backbone:
            x=text.checkpoint(block,x,use_reentrant=False)
        for block in self.backbone.vision_backbone.trunk.blocks:
            x=checkpoint.checkpoint(block,x,use_reentrant=False)
        return x


class RetentionTests(unittest.TestCase):
    def modules(self):
        text=types.ModuleType('sam3.model.text_encoder_ve');text.checkpoint=checkpoint.checkpoint
        package=types.ModuleType('sam3');package.__path__=[]
        model=types.ModuleType('sam3.model');model.__path__=[];model.text_encoder_ve=text
        package.model=model
        return {'sam3':package,'sam3.model':model,'sam3.model.text_encoder_ve':text}

    def test_dropout_losses_gradients_optimizer_and_rng_are_exact(self):
        torch.manual_seed(91);initial=Model().state_dict();results=[]
        with patch.dict(sys.modules,self.modules()):
            for optimized in (False,True):
                model=Model();model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
                torch.manual_seed(515)
                from contextlib import nullcontext
                with retention(model,'text_vision8') if optimized else nullcontext():
                    losses=[];gradients=[]
                    for _ in range(3):
                        opt.zero_grad(set_to_none=True)
                        out=model(torch.arange(8.).reshape(2,4)/10)
                        loss=out.square().mean()+out.sin().mean()*.1
                        loss.backward();losses.append(loss.detach().clone())
                        gradients.append([p.grad.clone() for p in model.parameters()])
                        opt.step()
                self.assertTrue(model.backbone.vision_backbone.trunk.use_act_checkpoint)
                results.append((losses,gradients,copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state()))
        def equal(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        self.assertTrue(equal(*results))

    def test_selected_calls_are_bypassed_and_context_restored(self):
        original=checkpoint.checkpoint
        modules=self.modules();model=Model()
        with patch.dict(sys.modules,modules):
            with patch.object(checkpoint,'checkpoint',wraps=original) as counted:
                # 32 visual blocks, eight retained; language blocks are retained.
                with retention(model,'text_vision8'):
                    model(torch.ones(1,4)).sum().backward()
                self.assertEqual(counted.call_count,24)
            self.assertIs(checkpoint.checkpoint,original)
            self.assertIs(modules['sam3.model.text_encoder_ve'].checkpoint,original)
            with self.assertRaisesRegex(ValueError,'intentional'):
                with retention(model,'text_vision8'):
                    raise ValueError('intentional')
            self.assertIs(checkpoint.checkpoint,original)
            self.assertIs(modules['sam3.model.text_encoder_ve'].checkpoint,original)


if __name__=='__main__':unittest.main()
