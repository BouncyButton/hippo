"""Verify study separation and inherited unattended job lifecycle behavior."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.geofwi_constraints import test_controller as suite
from scripts.geofwi_constraints.prepare_extension import STAGE

sys.modules.pop('baseline_runtime', None)
sys.modules.pop('parent_controller', None)
sys.path.insert(0, str(STAGE))
spec = importlib.util.spec_from_file_location('extension', STAGE / 'campaign.py')
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
c = ext.parent
suite.c, suite.STAGE = c, STAGE


class ExtensionTests(suite.ControllerTests):
    def test_stop_marker_prevents_continuation(self):
        (self.stage / 'STOP').touch()
        with patch.dict(os.environ, {'SLURM_JOB_ID': '42'}), \
             patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')), \
             patch.object(c.runtime, 'run', return_value=75), patch.object(c, 'submit') as submit:
            self.assertEqual(c.run(self.stage), 75)
            submit.assert_not_called()

    def test_fixed_confirmation_and_exploration_matrix(self):
        plan = ext.verify(STAGE, data=False)
        self.assertEqual(len(plan['tasks']), 27)
        self.assertEqual(len(plan['reused_rows']), 6)
        self.assertEqual(len({ext.key(t) for t in plan['tasks'] + plan['reused_rows']}), 33)
        self.assertFalse(set(t['id'] for t in plan['tasks']) & set(t['id'] for t in plan['reused_rows']))

    def test_moving_exploration_into_confirmation_fold_is_rejected(self):
        read = ext.read
        plan = deepcopy(read(STAGE / 'plan.json'))
        next(t for t in plan['tasks'] if t['role'] == 'exploration')['fold'] = 1
        def altered(path):
            return plan if path == STAGE / 'plan.json' else read(path)
        with patch.object(ext, 'read', side_effect=altered), self.assertRaises(AssertionError):
            ext.verify(STAGE, data=False)

    def test_summary_separates_confirmation_and_exploration_without_score_gate(self):
        for name in ('plan.json', 'controls.json', 'baseline_results.json'):
            c.write(self.stage / name, c.read(STAGE / name))
        plan = c.read(self.stage / 'plan.json')
        for task in plan['tasks']:
            c.write(self.stage / 'runs' / task['id'] / 'cleanup.json', {'status': 'complete'})
        def result(stage, task):
            return dict(task, final_dice=.49 if task['role'] == 'exploration' else .6)
        with patch.object(ext, 'validate_run', side_effect=result):
            doc = ext.summarize(self.stage, 'running')
        self.assertTrue(doc['complete_matrix'])
        self.assertTrue(doc['confirmation_complete'])
        self.assertFalse(doc['all_above_50'])
        self.assertEqual(len(doc['confirmation_rows']), 18)
        self.assertEqual(len(doc['exploration_rows']), 15)
        self.assertTrue(all('delta_vs_selected_constraints_pp' not in r for r in doc['confirmation_rows']))
        self.assertTrue(all('delta_vs_selected_constraints_pp' in r for r in doc['exploration_rows']))


if __name__ == '__main__':
    # The final full hash inventory is frozen after these preparatory checks.
    if not (STAGE / 'manifest.json').exists():
        c.write(STAGE / 'manifest.json', {})
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionTests))
    c.write(STAGE / 'controller_checks.json', dict(status='passed' if result.wasSuccessful() else 'failed',
                                                 tests=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
    sys.exit(0 if result.wasSuccessful() else 1)
