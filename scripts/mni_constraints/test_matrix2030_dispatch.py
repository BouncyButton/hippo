import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
from scripts.mni_constraints import matrix2030_dispatch as dispatch


class Clock:
    def __init__(self):self.now=0.
    def time(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class DispatchTests(unittest.TestCase):
    def test_registered_stage_runs_once_then_hands_off_at_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir();stage=root/'mni_nnunet_swin_refine2030_round_test';stage.mkdir()
            (stage/'manifest.json').write_text('{}');sha=hashlib.sha256(b'{}').hexdigest()
            (stage/'preflight.json').write_text(json.dumps(dict(status='passed',manifest_sha256=sha)))
            (stage/'plan.json').write_text(json.dumps(dict(deadline=1000.,gpu_workers=2,budget_flops=53)))
            entry=dict(stage=str(stage),module='scripts.mni_constraints.nnunet_swin_refine2030_round3',manifest_sha256=sha)
            (bundle/'control.json').write_text(json.dumps(dict(pause=False,refinement_queue=[entry])))
            spec=dict(parent=str(root/'parent'),start_at=1000.,python=sys.executable,budget_flops=53)
            clock=Clock();process=Mock();process.poll.return_value=0;process.returncode=0
            with patch.object(dispatch,'time',clock),patch.object(dispatch,'STOP',False),patch.object(dispatch.signal,'signal'), \
                 patch.object(dispatch.subprocess,'Popen',return_value=process) as launch,patch.dict(os.environ,{'SLURM_JOB_ID':'test','CUDA_VISIBLE_DEVICES':'0,1'}):
                self.assertTrue(dispatch.refine_until_handoff(bundle,spec));launch.assert_called_once()
                self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                self.assertEqual(clock.now,1000.)
            self.assertTrue((bundle/f'refinement_exit_{stage.name}.json').exists())

    def test_pause_prevents_refinement_and_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'control.json').write_text(json.dumps(dict(pause=True)))
            with patch.object(dispatch,'time',Clock()),patch.object(dispatch,'STOP',False),patch.object(dispatch.signal,'signal'),patch.object(dispatch.subprocess,'Popen') as launch:
                self.assertFalse(dispatch.refine_until_handoff(p,dict(start_at=1000.)));launch.assert_not_called()


if __name__=='__main__':unittest.main()
