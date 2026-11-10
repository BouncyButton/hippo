"""Regression checks for ordering and the single continuation across two phases."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import followon_job as job, controller, collect
from scripts.cross_dataset_campaign.common import read, write, digest


class Followon(unittest.TestCase):
    def test_completed_predecessor_requires_verified_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);plan={'tasks':[{'id':'one'}]}
            write(stage/'status.json',{'status':'running','completed':1})
            self.assertFalse(job.predecessor_complete(stage,plan))
            write(stage/'status.json',{'status':'complete','completed':1})
            for name in ('completion','cleanup'):write(stage/f'runs/one/{name}.json',{'status':'complete'})
            write(stage/'runs/one/archive_manifest.json',{'metadata':{}})
            write(stage/'runs/one/archive_acknowledgment.json',{'manifest_sha256':'wrong'})
            with self.assertRaises(ValueError):job.predecessor_complete(stage,plan)
            write(stage/'runs/one/archive_acknowledgment.json',{'manifest_sha256':digest(stage/'runs/one/archive_manifest.json')})
            self.assertTrue(job.predecessor_complete(stage,plan))

    def test_shared_successor_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);write(stage/'manifest.json',{})
            receipt={'job_id':'2','dependency':'1','manifest_sha256':digest(stage/'manifest.json')}
            write(stage/'successor.json',receipt)
            with patch.object(controller,'queue',return_value=[['1','RUNNING','ours'],['2','PENDING','ours']]),patch.object(job,'call_submit') as submit:
                self.assertEqual(job.ensure_successor(stage,'1'),receipt);submit.assert_not_called()

    def test_archive_owner_and_dependency_pause_block_automatic_restart(self):
        stage=Path('/cluster/followon');state={'status.json':{'status':'waiting_for_archive_offline'}}
        plan={'archive_resume_module':'followon','archive_dependency_stages':['/cluster/msd']}
        for responses in ([b'{"automatic_continuation_owner":"/cluster/other"}'],
                          [b'{}',b'0',b'{"pause":true}'],[b'{}',b'0',b'{}',b'1']):
            with patch.object(collect,'remote',side_effect=responses):
                self.assertIsNone(collect.resume_after_archive(stage,plan,state))

    def test_new_phase_does_not_reset_allocation_deadline(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'SLURM_JOB_ID':'1','HIPPO_ALLOCATION_DEADLINE':'100'}):
            stage=Path(directory);write(stage/'control.json',{'pause':True})
            with patch.object(controller,'verify',return_value={'graceful_stop_seconds':500}),patch.object(controller,'install_signals'),patch.object(controller,'summarize'):
                controller.run(stage)
            self.assertEqual(float(os.environ['HIPPO_DEADLINE']),100.)

    def test_incomplete_predecessor_does_not_launch_followon(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'SLURM_JOB_ID':'1'}):
            root=Path(directory);stage=root/'new';old=root/'old'
            for folder in (stage,old):write(folder/'control.json',{'pause':False})
            write(old/'status.json',{'status':'continuing_at_time_limit'})
            plan={'graceful_stop_seconds':500}
            original=(controller.verify,controller.command_for,controller.submit,controller.cancel_successor)
            try:
                with patch.object(job,'verify',return_value=plan),patch.object(job,'check_predecessor',return_value=(old,{})),patch.object(job,'predecessor_complete',return_value=False),patch.object(controller,'run') as run,patch.object(controller,'summarize'):
                    job.run(stage)
                    run.assert_called_once_with(old)
                    with patch.object(job,'BASE_COMMAND',return_value=['python','-m','old']):
                        command=controller.command_for(old,{},None,None)
                        self.assertIn('PYTHONPATH='+str(old/'source'),command[1])
                        self.assertEqual(command[4],'scripts.cross_dataset_campaign.msd_ce_constraints')
            finally:
                controller.verify,controller.command_for,controller.submit,controller.cancel_successor=original


if __name__=='__main__':unittest.main()
