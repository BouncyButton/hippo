"""CPU checks for non-invasive probes, native-only updates, and resumable diagnosis."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # Load registrations before temporary sys.modules patches.

from scripts.colab_sam3_geofwi import diagnose as d, runtime as rt, parallel
from scripts.medsam3_supervised_constraints import semantic_log_fields, VolumeSupport, sampled_losses
from scripts.cross_dataset_campaign import common


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A=torch.nn.Parameter(torch.tensor([.2,.3,.4]))
        self.base=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        self.register_buffer('visits',torch.zeros(1))


def reset(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def digest(model,adapter):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for n,p in model.named_parameters()
                                  if n.startswith('lora')==adapter)).hexdigest()


def setup_toy():
    truth=torch.zeros((3,7,9),dtype=torch.bool);truth[:,2:5,3:6]=True
    slabs=SimpleNamespace(indices={f'case{i}':[0,1,2] for i in range(5)},
        samples=[(None,t.numpy()) for t in truth],inventory=['five_volumes'])
    support=VolumeSupport.build(truth)
    original=SimpleNamespace(semantic_log_fields=semantic_log_fields,reset_seed=reset,
        parameter_digest=digest,is_adapter=lambda n:n.startswith('lora'))
    def forward(model,criterion,slabs,case,anchor):
        model.visits.add_(1);planes=[];native=[]
        for z in (anchor,(anchor+1)%3):
            x=model.lora_A
            noise=torch.rand(1,1,7,9)+random.random()+np.random.rand()
            output={'pred_masks':x[0]+noise,'pred_logits':x[1].reshape(1,1,1),
                    'presence_logit_dec':x[2].reshape(1,1)}
            planes.append(original.semantic_log_fields(output,(7,9)))
            native.append(sum(value.square().mean() for value in output.values()))
        bands,edge=sampled_losses(planes[0][0].exp(),planes[1][0].exp(),support,anchor,
                                 log_probability=planes[0][0],log_complement=planes[0][1])
        return dict(native=torch.stack(native).mean(),bands=bands,edge=edge)
    original.forward_slab=forward
    original.lr_for_update=lambda i,h,w,a,b:a*(i+1)/w
    def save(path,value):
        tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    original.save_checkpoint=save
    slabs.schedule=lambda p:[(f'case{i%5}',i%3) for i in range(p['updates']*p['accumulation'])]
    return original,slabs


class DiagnosticTests(unittest.TestCase):
    def test_probe_preserves_rng_parameters_buffers_and_existing_gradients(self):
        original,slabs=setup_toy();model=Tiny();reset(42)
        model.lora_A.grad=torch.ones_like(model.lora_A)
        state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        before=model.lora_A.detach().clone()
        row=d.measure(original,model,None,[model.lora_A],slabs,'case0',1,3,dict(bands=2.,edge=10.),0)
        torch.testing.assert_close(model.lora_A,before,rtol=0,atol=0)
        torch.testing.assert_close(model.lora_A.grad,torch.ones_like(model.lora_A))
        self.assertEqual(float(model.visits),0.)
        self.assertEqual(random.getstate(),state[0]);np.testing.assert_equal(np.random.get_state(),state[1])
        self.assertTrue(torch.equal(torch.get_rng_state(),state[2]))
        self.assertEqual(len(row['head_gradients']['edge']),6)
        self.assertEqual(row['confidence'][0]['foreground_pixels'],9)
        self.assertTrue(np.isfinite(row['gram']).all())

    def test_ten_update_end_to_end_resume_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            base=Path(tmp);outcomes=[]
            for label,interrupt in [('whole',False),('resumed',True)]:
                root=base/label/'bundle';prior=base/label/'prior';durable=base/label/'diagnostic'
                root.mkdir(parents=True)
                original,slabs=setup_toy();prototype=Tiny()
                p=dict(train_cases=list(slabs.indices),seed=0,calibration_seed=4,updates=25,
                       accumulation=2,warmup_updates=20,learning_rate=.01,final_learning_rate=.001)
                rows=[dict(case=c,anchor=a) for c in p['train_cases'] for a in [0,1,2,2]]
                shared=dict(rows=rows,inventory=slabs.inventory,
                    initialization=dict(frozen=digest(prototype,False),adapter=digest(prototype,True)))
                rt.write(prior/'environment_binding.json',{})
                rt.write(prior/'calibration/fold0/calibration.json',shared)
                rt.write(prior/'runs/edge/calibration.json',dict(weights=dict(bands=2.,edge=10.),
                    shared_calibration_sha256=rt.digest(prior/'calibration/fold0/calibration.json')))
                task=dict(id='edge',scientific_arm='dice_bands_edge')
                loads=[]
                def load(*args):
                    model=Tiny();loads.append(model)
                    return original,model,None,[model.lora_A],None,None,slabs
                def state():return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
                def restore(s):torch.set_rng_state(s['torch']);random.setstate(s['python']);np.random.set_state(s['numpy'])
                engine=SimpleNamespace(load=load,rng_state=state,set_rng=restore)
                stop=[interrupt]
                def stopping():
                    path=durable/'training_history.json'
                    return stop[0] and path.exists() and len(rt.read(path))>=3
                argv=['diagnose','--bundle',str(root),'--source-results',str(prior),'--durable',str(durable)]
                with patch.dict(sys.modules,{'runtime':rt,'engine':engine,'parallel':parallel}),\
                     patch.object(rt,'activate'),patch.object(rt,'verify_bundle',return_value={'tasks':[task]}),\
                     patch.object(rt,'binding',return_value={}),patch.object(rt,'fold_calibration',return_value=shared),\
                     patch.object(rt,'completed',return_value=True),patch.object(rt,'protocol',side_effect=lambda *_:dict(p)),\
                     patch.object(common,'install_signals'),patch.object(common,'stopping',side_effect=stopping),\
                     patch.object(sys,'argv',argv):
                    d.main()
                    if interrupt:
                        self.assertEqual(rt.read(durable/'checkpoint.json')['step'],3)
                        stop[0]=False;d.main()
                    self.assertEqual(rt.read(durable/'completion.json')['validation_volumes_evaluated'],0)
                    self.assertEqual(len(rt.read(durable/'measurements.json')),15)
                    self.assertEqual(len(rt.read(durable/'training_history.json')),10)
                    self.assertFalse((durable/'checkpoint.pt').exists())
                    count=len(loads);d.main();self.assertEqual(len(loads),count)
                outcomes.append((rt.read(durable/'training_history.json'),rt.read(durable/'completion.json')['final_adapter_sha256']))
            self.assertEqual(outcomes[0],outcomes[1])

    def test_fixed_weight_ratio_and_five_slab_summary(self):
        gram=[[4.,0.,0.],[0.,9.,0.],[0.,0.,16.]]
        value=d.ratio_metrics(gram,dict(bands=2.,edge=3.))
        self.assertAlmostEqual(value['edge_over_native_bands'],12/np.sqrt(40))
        self.assertEqual(value['cosine_edge_native_bands'],0.)

    def test_notebook_cannot_launch_old_sweep_or_full_preflight(self):
        import ast
        from scripts.colab_sam3_geofwi.prepare_diagnosis import EXPORT
        nb=rt.read(EXPORT/'SAM3_GeoFWI3D_edge_diagnosis.ipynb')
        source='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        self.assertNotIn("campaign('preflight')",source)
        self.assertNotIn("BUNDLE/'parallel.py'",source)
        self.assertIn('DIAGNOSTIC_SCRIPT',source)
        self.assertIn('diagnostics_edge_10updates_v1',source)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])


if __name__=='__main__':unittest.main()
