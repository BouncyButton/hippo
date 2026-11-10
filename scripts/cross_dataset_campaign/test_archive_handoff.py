"""An offline archive releases the GPU without deleting recovery files."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import controller, collect
from scripts.cross_dataset_campaign.common import read, write


class ArchiveHandoff(unittest.TestCase):
    def test_missing_base_delivery_releases_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);task=dict(id='one',kind='foundation',dataset='ADNI')
            plan=dict(tasks=[task],graceful_stop_seconds=1000,base_sha256='unused',
                storage=dict(minimum_free_gib=12),archive_wait_timeout_seconds=300)
            write(stage/'control.json',dict(pause=False))
            with patch.dict('os.environ',{'SLURM_JOB_ID':'999999999'}), \
                 patch.object(controller,'verify',return_value=plan),patch.object(controller,'install_signals'), \
                 patch.object(controller,'submit'),patch.object(controller,'quota',return_value={'free_gib':60}), \
                 patch.object(controller,'stopping',return_value=False),patch.object(controller,'cancel_successor') as cancel, \
                 patch.object(controller,'summarize') as summary, \
                 patch.object(controller.time,'monotonic',side_effect=[0,301]):
                controller.run(stage)
            cancel.assert_called_once_with(stage,'999999999')
            self.assertEqual(summary.call_args.args[2],'waiting_for_base_offline')

    def test_offline_archive_preserves_files_and_cancels_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory)
            task=dict(id='one',kind='conventional',dataset='MSD')
            plan=dict(tasks=[task],graceful_stop_seconds=1000,
                storage=dict(minimum_free_gib=12),archive_wait_timeout_seconds=300)
            write(stage/'control.json',dict(pause=False))
            write(stage/'runs/one/completion.json',dict(status='complete'))
            checkpoint=stage/'runs/one/checkpoint_latest.pt'
            checkpoint.write_bytes(b'recovery')
            with patch.dict('os.environ',{'SLURM_JOB_ID':'999999999'}), \
                 patch.object(controller,'verify',return_value=plan), \
                 patch.object(controller,'install_signals'), \
                 patch.object(controller,'submit'), \
                 patch.object(controller,'quota',return_value={'free_gib':60}), \
                 patch.object(controller,'stopping',return_value=False), \
                 patch.object(controller,'prepare'), \
                 patch.object(controller,'retire') as retire, \
                 patch.object(controller,'cancel_successor') as cancel, \
                 patch.object(controller,'summarize') as summary, \
                 patch.object(controller.time,'monotonic',side_effect=[0,301]):
                controller.run(stage)
            self.assertEqual(checkpoint.read_bytes(),b'recovery')
            retire.assert_not_called()
            cancel.assert_called_once_with(stage,'999999999')
            self.assertEqual(summary.call_args.args[2],'waiting_for_archive_offline')

    def test_archive_resume_respects_manual_pause_and_missing_receipts(self):
        stage=Path('/mnt/beegfsstudents/home/3160552/test')
        plan={'archive_resume_module':'scripts.cross_dataset_campaign.msd_ce_constraints_job'}
        state={'status.json':{'status':'waiting_for_archive_offline'}}
        for responses in ([b'{"pause":true}'],[b'{"pause":false}',b'1']):
            with patch.object(collect,'remote',side_effect=responses) as remote:
                self.assertIsNone(collect.resume_after_archive(stage,plan,state))
                self.assertEqual(remote.call_count,len(responses))
        with patch.object(collect,'remote',side_effect=[b'{"pause":false}',b'0',b'resumed']) as remote:
            self.assertEqual(collect.resume_after_archive(stage,plan,state),'resumed')
            command=remote.call_args.args[0]
            self.assertNotIn('resume',command)
            self.assertIn('controller.submit(p)',command[4])


if __name__=='__main__':
    unittest.main()
