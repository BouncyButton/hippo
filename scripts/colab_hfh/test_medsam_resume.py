import hashlib
import json
import random
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

from scripts.colab_hfh.medsam_constraints_resume import install


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([1.]))
        self.base = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)


class ResumeTests(unittest.TestCase):
    def test_completed_checkpoint_performs_no_training_and_keeps_bytes(self):
        with TemporaryDirectory() as directory:
            output=Path(directory);model=Tiny();reference=Mock(side_effect=AssertionError('Must not train'))
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            pd=lambda m,adapter:hashlib.sha256((m.adapter if adapter else m.base).detach().numpy().tobytes()).hexdigest()
            write=lambda p,v:p.write_text(json.dumps(v))
            original=SimpleNamespace(train=reference,digest=digest,parameter_digest=pd,write=write)
            p=dict(updates=200);write(output/'protocol.json',p);write(output/'runtime_validation.json',{})
            checkpoint=output/'checkpoint_latest.pt'
            torch.save(dict(step=200,history=[dict(lr=.1)]*200,protocol_sha256=digest(output/'protocol.json'),
                frozen_before=pd(model,False),adapter_before=pd(model,True),adapter={'adapter':torch.tensor([2.])},
                elapsed_seconds=12.,torch_rng=torch.get_rng_state(),cuda_rng=[],
                python_rng=random.getstate(),numpy_rng=np.random.get_state()),checkpoint)
            before=digest(checkpoint);install(original)
            original.train(model,None,list(model.parameters()),SimpleNamespace(schedule=lambda _:[]),p,{'weights':{}},'dice',output)
            reference.assert_not_called();self.assertEqual(digest(checkpoint),before)
            self.assertEqual(model.adapter.item(),2.)
            receipt=json.loads((output/'recovery_evaluation_only.json').read_text())
            self.assertEqual(receipt['optimizer_updates_performed'],0)
            self.assertEqual(receipt['restored_updates'],200)

    def test_partial_checkpoint_keeps_normal_resume_path(self):
        with TemporaryDirectory() as directory:
            output=Path(directory);(output/'runtime_validation.json').write_text('{}')
            torch.save({'step':42},output/'checkpoint_latest.pt')
            reference=Mock(return_value='resumed');original=SimpleNamespace(train=reference);install(original)
            self.assertEqual(original.train(None,None,[],None,{'updates':200},{},'dice',output),'resumed')
            reference.assert_called_once()


if __name__=='__main__':unittest.main()
