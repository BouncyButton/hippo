"""Regression checks for full-suite reconciliation and safe unattended execution."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.geofwi_constraints import test_controller as suite
from scripts.geofwi_constraints.prepare_reweighted import STAGE
sys.modules.pop('baseline_runtime',None)
sys.path.insert(0,str(STAGE))
spec=importlib.util.spec_from_file_location('reweighted_controller',STAGE/'campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
suite.c,suite.STAGE=c,STAGE


class ReweightedTests(suite.ControllerTests):
    def test_full_suite_has_no_reruns_or_baselines(self):
        plan=c.verify(STAGE,data=False)
        self.assertEqual(len(plan['tasks']),381)
        self.assertEqual(len(plan['reused_rows']),87)
        self.assertFalse({c.key(t) for t in plan['tasks']} & {c.key(t) for t in plan['reused_rows']})
        self.assertTrue(all(t['arm']!='dice' for t in plan['tasks']))
        for lane in (plan['tasks'][::2],plan['tasks'][1::2]):
            seen_new=False
            for t in lane:
                if t['model'] in ('unetr_vit_b16','nnunet'):seen_new=True
                else:self.assertFalse(seen_new)

    def test_incorrect_new_strength_rejected(self):
        original=c.read
        task=next(t for t in original(STAGE/'plan.json')['tasks'] if t['model']=='nnunet')
        path=STAGE/'protocols'/task['id']/'protocol.json'
        bad=deepcopy(original(path));bad['calibration_target_ratio']=.6
        with patch.object(c,'read',side_effect=lambda p:bad if p==path else original(p)),self.assertRaises(AssertionError):
            c.verify(STAGE,data=False)

    def test_stop_marker_prevents_continuation(self):
        (self.stage/'STOP').touch()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'42'}), \
             patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')), \
             patch.object(c.runtime,'run',return_value=75),patch.object(c,'submit') as submit:
            self.assertEqual(c.run(self.stage),75)
            submit.assert_not_called()

    def test_completion_independent_of_threshold(self):
        for name in ('plan.json','baseline_results.json'):
            c.write(self.stage/name,c.read(STAGE/name))
        for t in c.read(self.stage/'plan.json')['tasks']:
            c.write(self.stage/'runs'/t['id']/'cleanup.json',dict(status='complete'))
        with patch.object(c.runtime,'validate_run',side_effect=lambda stage,t:dict(t,final_dice=.49)):
            result=c.summarize(self.stage,'running')
        self.assertTrue(result['complete_matrix'])
        self.assertFalse(result['all_above_50'])
        self.assertEqual(len(result['rows']),468)


if __name__=='__main__':
    if not (STAGE/'manifest.json').exists():c.write(STAGE/'manifest.json',{})
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReweightedTests))
    c.write(STAGE/'controller_checks.json',dict(status='passed' if result.wasSuccessful() else 'failed',
          tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    sys.exit(0 if result.wasSuccessful() else 1)
