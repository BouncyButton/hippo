import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cluster_tail_20261022 as runner
from scripts.cross_dataset_campaign import storage, common


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)/'stage'
        self.root = self.stage/'bundle'; self.root.mkdir(parents=True)
        runner.write(self.root/'bundle_manifest.json', {})
        runner.write(self.stage/'assignment.json', dict(cluster_tasks=['task'],colab_tasks=['deferred'],
            completed_ids=['old'], deadline_epoch=0))
        runner.write(self.stage/'colab_binding.json', {'environment': 'original'})
        self.task = dict(id='task',fold=3,seed=0,arm='dice',updates=25)
        self.out = self.root/'work/task'; self.out.mkdir(parents=True)
        (self.out/'checkpoint_latest.pt').write_bytes(b'verified checkpoint')
        runner.write(self.out/'protocol.json', dict(updates=25,train_cases=['a'],evaluation_cases=['b']))
        runner.write(self.out/'completion.json', dict(status='complete',updates=25,
            checkpoint_sha256=runner.digest(self.out/'checkpoint_latest.pt')))
        runner.write(self.out/'training_summary.json', dict(updates=25,frozen_before='same',frozen_after='same'))
        runner.write(self.out/'training_history.json', [dict(update=i) for i in range(1,26)])
        runner.write(self.out/'evaluation.json', dict(records=[dict(split='training_support',case_name='a'),dict(split='validation',case_name='b')]))
        self.transport = runner.LocalTransport(self.stage,self.root,self.task,{'gpu':'cluster'})

    def ack(self, seconds):
        manifest = runner.read(self.out/'archive_manifest.json')
        archive = Path(self.temp.name)/'durable_archive'
        shutil.copytree(self.out,archive,ignore=shutil.ignore_patterns('*.pt'),dirs_exist_ok=True)
        storage.verify_archive(archive,manifest)
        runner.write(self.out/'archive_acknowledgment.json',dict(status='verified',
            manifest_sha256=runner.digest(self.out/'archive_manifest.json'),source_verified=True,
            durable_archive='/Users/test/reports/verified-run'))

    def test_archive_verified_before_checkpoint_deletion(self):
        with patch.object(runner.time,'sleep',side_effect=self.ack), patch.object(common,'stopping',return_value=False):
            self.transport.publish('completed/task',[])
        self.assertTrue((self.out/'checkpoint_latest.pt').exists())
        self.transport.state['completed']['task']=runner.read(self.out/'completion.json')
        self.transport.retire('task')
        self.assertFalse((self.out/'checkpoint_latest.pt').exists())
        self.assertEqual(runner.read(self.out/'cleanup.json')['status'],'complete')

    def test_unacknowledged_output_is_preserved(self):
        with patch.object(common,'stopping',return_value=True), self.assertRaises(SystemExit):
            self.transport.publish('completed/task',[])
        self.assertTrue((self.out/'checkpoint_latest.pt').exists())
        with self.assertRaises(ValueError): self.transport.retire('task')

    def test_launcher_failure_requires_verified_retirement(self):
        self.assertFalse(runner.verified_retirement(self.stage, 'task'))
        with patch.object(runner.time,'sleep',side_effect=self.ack), patch.object(common,'stopping',return_value=False):
            self.transport.publish('completed/task',[])
        self.transport.state['completed']['task']=runner.read(self.out/'completion.json')
        self.transport.retire('task')
        self.assertTrue(runner.verified_retirement(self.stage, 'task'))
        (self.out/'evaluation.json').write_text('{}')
        self.assertFalse(runner.verified_retirement(self.stage, 'task'))

    def test_task_ledgers_are_independent(self):
        self.transport.charge('task',{'step':7})
        second=runner.LocalTransport(self.stage,self.root,dict(self.task,id='other'),{'gpu':'cluster'})
        second.charge('other',{'step':11})
        self.assertEqual(runner.read(self.transport.state_path)['charges'],{'task':{'step':7}})

    def test_only_reconciled_colab_tasks_are_added(self):
        self.assertEqual(runner.assignments(self.stage),['task'])
        runner.write(self.stage/'colab_handoff.json',dict(additional_task_ids=['deferred'],colab_completed=[],
            assignment_sha256=runner.digest(self.stage/'assignment.json')))
        self.assertEqual(runner.assignments(self.stage),['task','deferred'])
        runner.write(self.stage/'colab_handoff.json',dict(additional_task_ids=['unknown'],colab_completed=[],
            assignment_sha256=runner.digest(self.stage/'assignment.json')))
        with self.assertRaises(ValueError): runner.assignments(self.stage)

    def test_blocked_handoff_preserves_ownership_without_execution(self):
        runner.write(self.stage/'colab_handoff.json',dict(additional_task_ids=['deferred'],colab_completed=[],
            blocked_tasks={'deferred':{'charged_step':1,'checkpoint_step':0}},
            assignment_sha256=runner.digest(self.stage/'assignment.json')))
        self.assertEqual(runner.assignments(self.stage),['task','deferred'])
        self.assertEqual(list(runner.blocked_tasks(self.stage)),['deferred'])
        value=runner.read(self.stage/'colab_handoff.json');value['blocked_tasks']={'unknown':{}}
        runner.write(self.stage/'colab_handoff.json',value)
        with self.assertRaises(ValueError):runner.blocked_tasks(self.stage)


if __name__=='__main__': unittest.main()
