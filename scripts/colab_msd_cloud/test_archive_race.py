"""Real archive checks around the live-log/worker-shutdown race; no GPU."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

sys.path.insert(0,str(Path(__file__).parent))
import hooks
import store
from scripts.cross_dataset_campaign import storage


class ArchiveRaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'run';self.output.mkdir()
        self.task=dict(id='run',kind='conventional',dataset='MSD')
        (self.output/'checkpoint_best.pt').write_bytes(b'best checkpoint')
        (self.output/'checkpoint_latest.pt').write_bytes(b'latest checkpoint')
        store.atomic(self.output/'config.json',dict(augmentation='none',budget_flops=100,
            train_cases=['train'],validation_cases=['val']))
        store.atomic(self.output/'completion.json',dict(status='complete',consumed_training_flops=99,
            discarded_attempts=0,selected_step=2,updates=2,
            checkpoint_best_sha256=store.digest(self.output/'checkpoint_best.pt'),
            checkpoint_latest_sha256=store.digest(self.output/'checkpoint_latest.pt')))
        store.atomic(self.output/'training_final.json',dict(cases=[dict(case_name='train')]))
        for name in ('best','latest'):
            store.atomic(self.output/f'validation_{name}.json',dict(cases=[dict(case_name='val',
                confusion=[[1,0],[0,1]],foreground_fn=0)]))
        store.atomic(self.output/'budget_ledger.json',dict(charged_attempts=2,training_flops=99))
        (self.output/'train.log').write_text('training finished\n')
        self.cloud=Mock();self.cloud.folder='server'
        self.zips=[]
        def upload(path,kind,task,phase):
            self.zips.append(Path(path).read_bytes())
            # Exactly the original failure: a console line appended after hash.
            with (self.output/'train.log').open('a') as f:f.write('server upload complete\n')
            return dict(id='result',local_sha256=store.digest(path))
        self.cloud.put_immutable.side_effect=upload

    def test_live_snapshot_survives_source_log_append_and_does_not_freeze_source_manifest(self):
        receipt,manifest=hooks.commit_result(self.root,self.output,self.task,'02',self.cloud,live=True)
        self.assertEqual(receipt['id'],'result')
        self.assertFalse((self.output/'archive_manifest.json').exists())
        with zipfile.ZipFile(io.BytesIO(self.zips[0])) as z:
            target=self.root/'snapshot';z.extractall(target)
        storage.verify_archive(target,manifest)
        self.assertEqual((target/'train.log').read_text(),'training finished\n')
        self.assertIn('server upload complete',(self.output/'train.log').read_text())
        self.assertTrue((self.output/'checkpoint_latest.pt').exists())
        self.assertEqual(list(self.root.glob('result-snapshot-*')),[])

    def test_old_manifest_repaired_only_for_appended_log_and_includes_shutdown_diagnostics(self):
        original=storage.prepare(self.output,self.task)
        with (self.output/'train.log').open('a') as f:f.write('SERVER VERIFIED RESULT\n')
        store.atomic(self.output/'concurrency_shutdown.json',dict(completed=True))
        fixed=hooks.closed_manifest(self.output,self.task)
        storage.verify_archive(self.output,fixed)
        self.assertIn('concurrency_shutdown.json',fixed['metadata'])
        self.assertEqual(fixed['completion_sha256'],original['completion_sha256'])
        self.assertEqual(len(list(self.root.glob('run_prior_manifest_*.json'))),1)
        again=hooks.closed_manifest(self.output,self.task)
        self.assertEqual(again,fixed)

    def test_log_edit_is_not_misrepresented_as_append(self):
        storage.prepare(self.output,self.task)
        (self.output/'train.log').write_text('different log\n')
        with self.assertRaisesRegex(ValueError,'prefix differs'):
            hooks.closed_manifest(self.output,self.task)

    def test_scientific_change_blocks_repair_and_retains_checkpoints(self):
        storage.prepare(self.output,self.task)
        store.atomic(self.output/'budget_ledger.json',dict(charged_attempts=3,training_flops=120))
        with self.assertRaisesRegex(ValueError,'Scientific archive differs'):
            hooks.closed_manifest(self.output,self.task)
        self.assertTrue((self.output/'checkpoint_latest.pt').exists())

    def test_bundle_bytes_ignore_filesystem_timestamps(self):
        manifest=hooks.closed_manifest(self.output,self.task)
        first=hooks.bundle_results(self.output,manifest).read_bytes()
        import os
        os.utime(self.output/'train.log',(1900000000,1900000000))
        second=hooks.bundle_results(self.output,manifest).read_bytes()
        self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
