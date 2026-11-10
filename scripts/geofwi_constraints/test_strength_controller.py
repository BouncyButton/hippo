"""Run the existing lifecycle tests against the new frozen controller."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "experiments/geofwi3d_constraint_strength_20261020"
sys.path.insert(0, str(ROOT))
from scripts.geofwi_constraints import test_controller as suite

sys.modules.pop("baseline_runtime", None)
sys.path.insert(0, str(STAGE))
spec = importlib.util.spec_from_file_location("strength_controller", STAGE / "campaign.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
suite.c, suite.STAGE = c, STAGE


class StrengthTests(suite.ControllerTests):
    def test_stop_marker_prevents_continuation(self):
        (self.stage / "STOP").touch()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "42"}), \
             patch.object(c.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")), \
             patch.object(c.runtime, "run", return_value=75), patch.object(c, "submit") as submit:
            self.assertEqual(c.run(self.stage), 75)
            submit.assert_not_called()

    def test_strength_is_part_of_result_identity(self):
        tasks = c.read(STAGE / "plan.json")["tasks"]
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({c.key(t) for t in tasks}), 12)
        self.assertEqual({t["model"] for t in tasks[::2]}, {"unetr_vit_b16"})
        self.assertEqual({t["model"] for t in tasks[1::2]}, {"nnunet"})


if __name__ == "__main__":
    unittest.main()
