import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from scripts.colab_hfh.constraint_online_sdf import OnlineSDF, coefficient, measure, probe_allowance, validate_trace


def trace_probe_cost(logits, labels, sdf):
    """Conservative elementary-operation accounting; unknown operators fail."""
    zero = {'_to_copy','detach','zeros','scatter_','arange','slice','ones_like','expand',
            'unsqueeze','slice_backward','_local_scalar_dense','empty','empty_like','view'}
    pointwise = {'mul':1,'rsub':1,'add':1,'div':16,'eq':1,'sub':1,'neg':1,'pow':2,'sqrt':16}
    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.total=0;self.operations={}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result=func(*args,**(kwargs or {}));name=str(func).split('.')[1]
            if name in zero:
                cost=0
            elif name in pointwise:
                cost=result.numel()*pointwise[name]
            elif name in ('sum','mean'):
                cost=args[0].numel()+result.numel()
            elif name=='_softmax':
                # Includes generous allowances for exp/division/reductions.
                cost=64*args[0].numel()
            elif name=='_softmax_backward_data':
                cost=16*args[0].numel()
            else:
                raise AssertionError('Unaccounted online SDF probe operator: '+str(func))
            self.total+=cost;self.operations[str(func)]=self.operations.get(str(func),0)+cost
            return result
    count=Count()
    with count:measure(logits,labels,sdf)
    return dict(upper_bound_flops=count.total,operations=count.operations,
                charged_allowance_flops=probe_allowance(),
                interpretation='Conservative arithmetic upper bound from dispatched operators; no model forward/backward in this detached probe.')


class OnlineSDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def inputs(size=8):
        generator=torch.Generator().manual_seed(17)
        logits=torch.randn((1,2,size,size,size),generator=generator,requires_grad=True)
        labels=torch.zeros((1,1,size,size,size),dtype=torch.long)
        labels[:,:,size//4:3*size//4,size//4:3*size//4,size//4:3*size//4]=1
        return logits,labels,1-2*labels.float()

    def test_target_and_upper_multiplier(self):
        self.assertEqual(coefficient(2,4,1,.1),(0.05,.1,False))
        weight,ratio,clipped=coefficient(2,.00001,1,.1)
        self.assertEqual(weight,64);self.assertLess(ratio,.1);self.assertTrue(clipped)
        for value in (0,-1,float('nan')):
            with self.assertRaises(ValueError):coefficient(value,1,1,.1)

    def test_probe_preserves_model_graph_rng_and_inputs(self):
        z,y,sdf=self.inputs();rng=torch.get_rng_state().clone();before=z.detach().clone()
        native,auxiliary=measure(z,y,sdf)
        self.assertGreater(native,0);self.assertGreater(auxiliary,0)
        self.assertIsNone(z.grad);self.assertTrue(torch.equal(z,before))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        z.square().sum().backward();self.assertTrue(torch.equal(z.grad,2*z.detach()))

    def test_wrapper_preserves_native_path_and_logs_actual_strength(self):
        from scripts.adni_constraints.objectives import training_loss
        from functools import partial
        original=partial(training_loss,policy='delayed_5_of_9')
        z,y,sdf=self.inputs(64)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'budget_ledger.json').write_text('{"charged_attempts":115}')
            wrapper=OnlineSDF(original,dict(model_identity='swinunetr',active_objective='dice_sdf',
                online_sdf_gradient_ratio=.1,online_sdf_max_multiplier=64.),path)
            self.assertTrue(torch.equal(wrapper(z,y,'dice',{'sdf':1.},.2),
                                        original(z,y,'dice',{'sdf':1.},.2)))
            self.assertFalse((path/'online_sdf_strength.jsonl').exists())
            loss=wrapper(z,y,'dice_sdf',{'sdf':1.},1.,sdf,sdf_scale=.5)
            row=json.loads((path/'online_sdf_strength.jsonl').read_text())
            self.assertAlmostEqual(row['achieved_unramped_ratio'],.1)
            self.assertAlmostEqual(row['achieved_ramped_ratio'],.05)
            expected=original(z,y,'dice_sdf',{'sdf':row['effective_weight']},1.,sdf,sdf_scale=.5)
            self.assertTrue(torch.equal(loss,expected));loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())
            protocol=dict(online_sdf_gradient_ratio=.1,online_sdf_max_multiplier=64.)
            schedule=dict(sdf_prefix_updates=114,max_steps=115,sdf_warmup_updates=2,
                          online_sdf_probe_flops_per_active_update=probe_allowance())
            result=validate_trace(path,protocol,schedule,1.)
            self.assertEqual(result['probe_optimization_flops_allowance'],probe_allowance())
            row['effective_weight']*=2
            (path/'online_sdf_strength.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):validate_trace(path,protocol,schedule,1.)

    def test_full_grid_probe_fits_declared_arithmetic_allowance(self):
        previous=torch.get_num_threads();torch.set_num_threads(2)
        try:report=trace_probe_cost(*self.inputs(64))
        finally:torch.set_num_threads(previous)
        self.assertLess(report['upper_bound_flops']+65536,probe_allowance())


if __name__=='__main__':unittest.main()
