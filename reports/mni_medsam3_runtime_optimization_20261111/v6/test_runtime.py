import ast
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from scripts.cross_dataset_campaign import foundation
from scripts.cross_dataset_campaign.foundation_fast import gram_matrix
from scripts.cross_dataset_campaign.common import ARMS,read,write
import mni_runtime


class RuntimeTests(unittest.TestCase):
    def test_prefer_mixed_only_when_resources_are_available(self):
        text='NodeName=test CPUTot=24 CPUAlloc=0 RealMemory=500000 AllocMem=0 State=IDLE Gres=gpu:7g.80gb:1(S:0),gpu:4g.40gb:3(S:0-1) GresUsed=gpu:7g.80gb:0(IDX:N/A),gpu:4g.40gb:0(IDX:N/A)'
        self.assertEqual(mni_runtime.preferred_profile(text),'mixed')
        self.assertEqual(mni_runtime.preferred_profile(text.replace('gpu:7g.80gb:0','gpu:7g.80gb:1')),'balanced')
        self.assertEqual(mni_runtime.preferred_profile(text.replace('State=IDLE','State=DOWN')),'balanced')
        self.assertEqual(mni_runtime.preferred_profile(text.replace('CPUAlloc=0','CPUAlloc=20')),'balanced')

    def test_worker_types_are_isolated_and_gpu_count_is_one(self):
        command=['srun','--gres=gpu:4g.40gb:1','--gpus-per-task=4g.40gb:1','--mem=40G']
        self.assertEqual(mni_runtime.lane_command(command,1,'mixed'),command)
        changed=mni_runtime.lane_command(command,0,'mixed')
        self.assertIn('--gres=gpu:7g.80gb:1',changed)
        self.assertIn('--gpus-per-task=7g.80gb:1',changed)
        self.assertEqual(mni_runtime.lane_command(command,0,'balanced'),command)

    def test_only_gram_assignment_changes(self):
        altered=mni_runtime.replace_gram(foundation.calibrate,gram_matrix)
        self.assertEqual(inspect.signature(altered),inspect.signature(foundation.calibrate))
        self.assertIn('_exact_batched_gram',altered.__globals__)

    def test_all_arms_calibrate_exactly_with_identical_rows_and_weights(self):
        torch.manual_seed(42)
        parameters=[torch.randn(3,4,dtype=torch.float64,requires_grad=True),torch.randn(5,dtype=torch.float64,requires_grad=True)]
        def forward(*args):
            x,y=parameters
            return dict(native=(x.square().sum()+y.square().sum()),bands=x.sin().sum()+y.cos().sum(),
                edge=(x.cos().sum()+y.sin().sum()),sdf=x.sum()*.1+y.sum()*.3,depth=x.pow(3).sum()+y.pow(3).sum())
        slabs=SimpleNamespace(calibration_schedule=lambda count:[('case',i) for i in range(8)])
        protocol=dict(calibration_seed=8,calibration_slabs_per_volume=8,gradient_ratio_target=.1,gradient_ratio_p95_cap=.5)
        with tempfile.TemporaryDirectory() as temporary,patch.object(foundation,'forward_slab',forward),patch.object(foundation,'stopping',return_value=False),patch.object(foundation,'reset_seed'):
            optimized=mni_runtime.replace_gram(foundation.calibrate,gram_matrix)
            for arm in ARMS:
                a=Path(temporary)/arm/'reference';b=Path(temporary)/arm/'batched';a.mkdir(parents=True);b.mkdir(parents=True)
                expected=foundation.calibrate(None,None,parameters,slabs,protocol,arm,a)
                actual=optimized(None,None,parameters,slabs,protocol,arm,b)
                self.assertEqual(actual,expected,arm)
                if (a/'calibration_partial.json').exists():
                    self.assertEqual(read(a/'calibration_partial.json'),read(b/'calibration_partial.json'))
                    partial=read(a/'calibration_partial.json');partial['rows']=partial['rows'][:4]
                    write(b/'calibration_partial.json',partial)
                    self.assertEqual(optimized(None,None,parameters,slabs,protocol,arm,b),expected)

    def test_gram_preserves_missing_gradients_and_order(self):
        torch.manual_seed(2)
        gradients=[tuple(torch.randn(3,7,dtype=torch.float64) if (i+j)%3 else None for j in range(9)) for i in range(5)]
        self.assertEqual(gram_matrix(gradients),mni_runtime.reference_gram(gradients))
        self.assertEqual(gram_matrix([(None,),(None,)]),[[0,0],[0,0]])

    def test_unrecognized_reference_is_rejected(self):
        def different():
            gram=[]
            return gram
        # Nested function text is intentionally outside the permitted shape.
        with self.assertRaises((ValueError,IndentationError)):mni_runtime.replace_gram(different,gram_matrix)


if __name__=='__main__':unittest.main()
