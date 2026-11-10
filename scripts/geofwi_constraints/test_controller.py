"""Regression checks for unattended submission and verified retirement."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT/"experiments/geofwi3d_constraints_3fold_3seed_20261020"
sys.path.insert(0,str(STAGE))
spec=importlib.util.spec_from_file_location("constraints_controller",Path(__file__).with_name("campaign.py"))
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.stage=Path(self.temp.name)
        for name in ("cpu_contract_checks","cpu_audit_unetr","cpu_audit_basic"):
            c.write(self.stage/f"{name}.json",dict(status="passed"))
        c.write(self.stage/"manifest.json",{})
        self.calls=[]

    def tearDown(self):
        self.temp.cleanup()

    def output(self,cmd,**kwargs):
        self.calls.append(cmd)
        return "" if cmd[0]=="squeue" else "12345\n"

    def submit(self,continuation=False,queue=None):
        def output(cmd,**kwargs):
            if queue is not None and cmd[0]=="squeue":
                self.calls.append(cmd)
                return queue
            return self.output(cmd,**kwargs)
        with patch.object(c,"verify"), patch.object(c.runtime,"quota",return_value={"free_gib":65}), \
             patch.object(c.subprocess,"check_output",side_effect=output):
            return c.submit(self.stage,continuation)

    def test_receipt_prevents_duplicate(self):
        self.assertEqual(self.submit()["job_id"],"12345")
        self.assertEqual(self.submit()["job_id"],"12345")
        self.assertEqual(sum(cmd[0]=="sbatch" for cmd in self.calls),1)

    def test_unknown_outcome_stops(self):
        c.write(self.stage/"submissions/initial/intent.json",{"time":1})
        with self.assertRaises(RuntimeError): self.submit()
        self.assertFalse(self.calls)

    def test_existing_queued_job_prevents_submission(self):
        with self.assertRaises(AssertionError): self.submit(queue="99|PENDING|other\n")
        self.assertFalse(any(cmd[0]=="sbatch" for cmd in self.calls))

    def test_continuation_depends_on_own_allocation(self):
        c.write(self.stage/"status.json",dict(status="needs_resume"))
        with patch.dict(os.environ,{"SLURM_JOB_ID":"42"}):
            result=self.submit(True,"42|RUNNING|hippo-geofwi-constraints\n")
        self.assertIn("--dependency=afterany:42",result["command"])
        self.assertEqual(result["predecessor"],"42")

    def test_bad_status_does_not_continue(self):
        c.write(self.stage/"status.json",dict(status="failed"))
        with patch.dict(os.environ,{"SLURM_JOB_ID":"42"}), self.assertRaises(AssertionError):
            self.submit(True,"42|RUNNING|hippo-geofwi-constraints\n")
        self.assertFalse(any(cmd[0]=="sbatch" for cmd in self.calls))

    def test_mixed_failure_and_pause_cannot_restart(self):
        c.write(self.stage/"failure_42_0.json",dict(status="failed"))
        with patch.dict(os.environ,{"SLURM_JOB_ID":"42"}), \
             patch.object(c.subprocess,"run",return_value=subprocess.CompletedProcess([],0,"","")), \
             patch.object(c.runtime,"run",return_value=75), patch.object(c,"submit") as submit, \
             patch.object(c,"summarize"):
            self.assertEqual(c.run(self.stage),1)
            submit.assert_not_called()

    def test_clean_boundary_submits_continuation(self):
        with patch.dict(os.environ,{"SLURM_JOB_ID":"42"}), \
             patch.object(c.subprocess,"run",return_value=subprocess.CompletedProcess([],0,"","")), \
             patch.object(c.runtime,"run",return_value=75), patch.object(c,"submit") as submit:
            self.assertEqual(c.run(self.stage),0)
            submit.assert_called_once_with(self.stage,continuation=True)

    def test_archive_corruption_blocks_deletion(self):
        archive=c.ClusterArchive(STAGE)
        archive.stage=self.stage
        out=self.stage/"runs/task"
        destination=self.stage/"archives/task"
        c.write(out/"metrics.json",{"dice":.5})
        destination.mkdir(parents=True)
        (destination/"metrics.json").write_bytes((out/"metrics.json").read_bytes())
        checkpoint=out/"checkpoint_latest.pt"
        checkpoint.write_bytes(b"example-recovery")
        manifest=dict(metadata={"metrics.json":{"bytes":(out/"metrics.json").stat().st_size,
                                               "sha256":c.sha(out/"metrics.json")}},
                      deletion_candidates=[dict(name=checkpoint.name,bytes=checkpoint.stat().st_size,sha256=c.sha(checkpoint))])
        c.write(out/"archive_manifest.json",manifest)
        c.write(out/"archive_acknowledgment.json",dict(durable_archive=str(destination),
                                                     manifest_sha256=c.sha(out/"archive_manifest.json")))
        c.write(destination/"metrics.json",{"dice":.4})
        with self.assertRaises(ValueError): archive.retire(out)
        self.assertTrue(checkpoint.exists())
        (destination/"metrics.json").write_bytes((out/"metrics.json").read_bytes())
        self.assertEqual(archive.retire(out)["status"],"complete")
        self.assertFalse(checkpoint.exists())
        self.assertTrue((destination/"metrics.json").exists())
        self.assertEqual(archive.retire(out)["status"],"complete")


if __name__=="__main__":
    unittest.main()
