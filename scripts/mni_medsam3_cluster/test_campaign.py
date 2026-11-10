"""CPU checks for immutable recovery, fixed splits and queue admission."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign.common import read, write
from scripts.mni_medsam3_cluster import campaign, worker
from scripts.mni_medsam3_cluster.prepare import STAGE


class CampaignTests(unittest.TestCase):
    def test_frozen_scope_and_nested_support(self):
        plan = campaign.verify(STAGE)
        protocols = [read(STAGE/'protocols'/t['protocol']/'protocol.json') for t in plan['tasks']]
        self.assertEqual([p['seed'] for p in protocols], [0, 1, 2])
        self.assertEqual(protocols[0]['support_subjects'],
            ['hippocampus_mni_'+s for s in ('s23', 's03', 's25', 's11', 's14')])
        self.assertEqual(len(protocols[0]['evaluation_cases']), 10)
        self.assertTrue(all(p['updates'] == 200 for p in protocols))

    def test_async_checkpoint_is_immutable_during_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root/'local'; output.mkdir(); durable = root/'durable'
            write(output/'protocol.json', dict(seed=0)); write(output/'calibration.json', dict(weights={}))
            checkpoint = output/'checkpoint_latest.pt'; checkpoint.write_bytes(b'first')
            entered, release = threading.Event(), threading.Event()
            real_copy = worker.copy_verified
            def delayed(source, target):
                if source.name == 'mirror_snapshot.pt':
                    entered.set(); release.wait(5)
                real_copy(source, target)
            with patch.object(worker, 'copy_verified', delayed):
                mirror = worker.CheckpointMirror(output, durable, interval=0)
                try:
                    mirror.capture(checkpoint)
                    self.assertTrue(entered.wait(5))
                    replacement = output/'replacement'; replacement.write_bytes(b'second'); replacement.replace(checkpoint)
                finally:
                    release.set(); mirror.close()
            self.assertEqual((durable/'checkpoint_latest.pt').read_bytes(), b'first')
            self.assertEqual(checkpoint.read_bytes(), b'second')
            self.assertFalse((output/'mirror_snapshot.pt').exists())

    def test_two_lane_command_isolates_devices(self):
        plan = campaign.verify(STAGE)
        command = campaign.worker_command(STAGE, plan, plan['tasks'][2], Path('/tmp/base.pt'), 1)
        self.assertIn('--gpus-per-task=4g.40gb:1', command)
        self.assertIn('--ntasks=1', command)
        self.assertIn('HIPPO_GPU_WORKER=1', command)
        self.assertIn('scripts.mni_medsam3_cluster.worker', command)

    def test_full_account_queue_is_never_submitted(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder); write(stage/'control.json', dict(pause=False))
            with patch.object(campaign, 'verify'), patch.object(campaign.controller, 'queue', return_value=[['1','RUNNING','other'],['2','PENDING','other']]), patch.object(campaign, 'submit') as submit:
                def pause(_):
                    write(stage/'control.json', dict(pause=True))
                with patch.object(campaign.time, 'sleep', pause):
                    campaign.admit(stage)
                submit.assert_not_called()
                self.assertEqual(read(stage/'admission.json')['status'], 'paused')

    def test_pending_submission_is_withdrawn_if_another_campaign_wins_race(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder)
            queues = [[['1','RUNNING','other']], [['1','RUNNING','other'],['2','PENDING','other'],['3','PENDING','ours']]]
            with patch.object(campaign, 'verify', return_value={}), patch.object(campaign.controller, 'queue', side_effect=queues), patch.object(campaign.controller, 'submit', return_value=dict(job_id='3')), patch.object(campaign.subprocess, 'run') as call:
                with self.assertRaisesRegex(RuntimeError, 'Submission race'):
                    campaign.submit(stage)
                call.assert_called_once_with(['scancel','3'], check=True)

    def test_sweep_continuation_is_not_completion(self):
        from scripts.cross_dataset_campaign.common import digest
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder); write(stage/'plan.json', dict(tasks=['a','b']))
            plan = dict(wait_for_campaigns=[dict(stage=str(stage), plan_sha256=digest(stage/'plan.json'), expected_runs=2)])
            for state in ('running_offline', 'continuing_at_time_limit', 'failed', 'paused'):
                write(stage/'status.json', dict(status=state, training_completed=1, active=[]))
                self.assertFalse(campaign.dependencies_ready(plan)[0])
            write(stage/'status.json', dict(status='complete_pending_archive', training_completed=2, active=[]))
            self.assertTrue(campaign.dependencies_ready(plan)[0])
            write(stage/'plan.json', dict(tasks=['a','b','c']))
            self.assertFalse(campaign.dependencies_ready(plan)[0])


if __name__ == '__main__':
    unittest.main()
