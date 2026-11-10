"""Integrity and Drive-read regression checks for disposable local verification."""
from collections import Counter
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.colab_adni import runtime as frozen
from scripts.colab_adni_parallel.archive_verification import verified_snapshot
from scripts.colab_adni_parallel import scheduler as s


class ArchiveVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.durable = self.base/'drive'
        self.local = self.base/'cache'
        self.item = dict(id='saved', model='nnunet', arm='dice', fold=0, seed=0, protocol='nnunet_s0')
        self.binding = dict(bundle_sha256='frozen_source')
        self.folder = self.durable/'runs'/self.item['id']
        frozen.write(self.durable/'environment_binding.json', self.binding)
        frozen.write(self.folder/'completion.json', dict(status='complete', final_training_hard_dice=.9,
            updates=12, consumed_training_flops=120, calibration_flops=5,
            selected_step=7,best_validation_hard_dice=.85,unused_flops=1,discarded_attempts=0))
        frozen.write(self.folder/'training_final.json',dict(cases=[dict(foreground_fn=1,foreground_fp=1)]))
        frozen.write(self.folder/'calibration.json',dict(weights={}))
        frozen.write(self.folder/'validation_latest.json', dict(summary=dict(macro_hard_dice=.8),
            cases=[dict(foreground_fn=2, foreground_fp=3)]))
        frozen.write(self.folder/'nested/config.json', dict(augmentation='none'))
        frozen.write(self.folder/'concurrency_old.json', dict(completed=True, observed_full_run=True,
            peak_reserved_bytes=1024, peak_rss_bytes=2048, control_sha256='old_control'))
        (self.folder/'curve.csv').write_text('step,loss\n1,0.5\n')
        (self.folder/'events.jsonl').write_text('{"step": 1}\n')
        frozen.write(self.folder/'cleanup.json', dict(status='complete'))
        self.manifest = dict(task=dict(self.item), metadata={
            str(p.relative_to(self.folder)):dict(bytes=p.stat().st_size, sha256=frozen.digest(p))
            for p in self.folder.rglob('*') if p.is_file() and p.name != 'cleanup.json'})
        self.write_proof()

    def write_proof(self):
        frozen.write(self.folder/'archive_manifest.json', self.manifest)
        frozen.write(self.folder/'archive_acknowledgment.json', dict(status='verified', source_verified=True,
            manifest_sha256=frozen.digest(self.folder/'archive_manifest.json'),
            source_manifest_sha256=self.binding['bundle_sha256']))

    def snapshot(self, **kwargs):
        return verified_snapshot(frozen, self.durable, self.local, self.item, self.binding, **kwargs)

    def durable_bytes(self):
        return {str(p.relative_to(self.durable)):p.read_bytes()
                for p in self.durable.rglob('*') if p.is_file()}

    def test_each_source_file_read_once_with_full_frozen_verifier_and_same_results(self):
        before = self.durable_bytes()
        expected = frozen.result_row(self.durable, self.item)
        opened = Counter()
        original_open = Path.open
        events = []
        def observed_open(path, mode='r', *args, **kwargs):
            if self.durable in path.parents and 'r' in mode:
                opened[str(path.relative_to(self.durable))] += 1
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', observed_open), patch.object(frozen, 'completed', wraps=frozen.completed) as verifier:
            with self.snapshot(progress=lambda **fields:events.append(fields)) as local:
                self.assertNotEqual(local, self.durable)
                self.assertEqual(frozen.result_row(local, self.item), expected)
                self.assertTrue(frozen.read(local/'runs/saved/concurrency_old.json')['observed_full_run'])
            verifier.assert_called_once_with(local, self.item)
        expected_files = set(self.manifest['metadata']) | {'cleanup.json','archive_manifest.json','archive_acknowledgment.json'}
        self.assertEqual(opened, Counter({'runs/saved/'+name:1 for name in expected_files}))
        self.assertFalse(local.exists())
        self.assertEqual(list(self.local.iterdir()), [])
        self.assertEqual(self.durable_bytes(), before)
        self.assertEqual(events[-1]['archive_operation'], 'verified locally')
        self.assertEqual(max(e.get('archive_files_copied',0) for e in events), len(expected_files))

    def test_recheck_reads_fresh_drive_bytes_and_cannot_trust_prior_success(self):
        with self.snapshot() as local:
            self.assertIsNotNone(local)
        (self.folder/'completion.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Archive differs'):
            with self.snapshot():
                self.fail('Corrupt archive was accepted')
        self.assertEqual(list(self.local.iterdir()), [])

    def test_incomplete_and_absent_cleanup_are_pending_not_verified(self):
        for cleanup in (dict(status='in_progress'), None):
            with self.subTest(cleanup=cleanup):
                if cleanup is None:
                    (self.folder/'cleanup.json').unlink()
                else:
                    frozen.write(self.folder/'cleanup.json', cleanup)
                before = self.durable_bytes()
                with patch.object(frozen, 'completed', wraps=frozen.completed) as verifier:
                    with self.snapshot() as local:
                        self.assertIsNone(local)
                    verifier.assert_not_called()
                self.assertEqual(self.durable_bytes(), before)

    def test_missing_required_file_blocks_and_does_not_fall_back_to_cached_copy(self):
        (self.folder/'validation_latest.json').unlink()
        with self.assertRaises(FileNotFoundError):
            with self.snapshot():
                self.fail('Missing archive member was accepted')
        self.assertEqual(list(self.local.iterdir()), [])

    def test_task_identity_source_and_acknowledgment_gates_remain_mandatory(self):
        for mutation in ('task', 'manifest_sha256', 'source_manifest_sha256', 'source_verified', 'status'):
            with self.subTest(mutation=mutation):
                self.manifest['task'] = dict(self.item)
                if mutation == 'task':
                    self.manifest['task']['seed'] = 99
                self.write_proof()
                if mutation != 'task':
                    ack = frozen.read(self.folder/'archive_acknowledgment.json')
                    ack[mutation] = False if mutation == 'source_verified' else 'changed'
                    frozen.write(self.folder/'archive_acknowledgment.json', ack)
                before = self.durable_bytes()
                with self.assertRaisesRegex(ValueError, 'identity or archive acknowledgment'):
                    with self.snapshot():
                        self.fail('Invalid proof was accepted')
                self.assertEqual(self.durable_bytes(), before)

    def test_readability_is_checked_even_when_corrupt_json_has_matching_hash(self):
        path = self.folder/'nested/config.json'
        path.write_text('not JSON')
        self.manifest['metadata']['nested/config.json'] = dict(bytes=path.stat().st_size, sha256=frozen.digest(path))
        self.write_proof()
        with self.assertRaises(ValueError):
            with self.snapshot():
                self.fail('Unreadable JSON was accepted')

    def test_unsafe_paths_and_proof_overwrites_rejected(self):
        original = dict(self.manifest['metadata'])
        for name in ('../escape', '/outside', '.', '', './completion.json',
                     'cleanup.json', 'archive_manifest.json/child', 'nested/../completion.json'):
            with self.subTest(name=name):
                self.manifest['metadata'] = dict(original)
                self.manifest['metadata'][name] = dict(bytes=0, sha256='invalid')
                self.write_proof()
                with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                    with self.snapshot():
                        self.fail('Unsafe path was accepted')
        self.assertFalse((self.base/'escape').exists())

    def test_interrupt_after_copy_keeps_durable_archive_and_removes_only_local_snapshot(self):
        before = self.durable_bytes()
        checks = 0
        def stop():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError('interrupt requested')
        with self.assertRaises(RuntimeError):
            with self.snapshot(check=stop):
                self.fail('Interrupted verification continued')
        self.assertEqual(list(self.local.iterdir()), [])
        self.assertEqual(self.durable_bytes(), before)

    def test_temporary_copies_are_isolated_and_never_replace_existing_local_files(self):
        self.local.mkdir()
        marker = self.local/'preserve'; marker.write_text('earlier unrelated file')
        with self.snapshot() as one, self.snapshot() as two:
            self.assertNotEqual(one, two)
            self.assertTrue(one.exists())
            self.assertTrue(two.exists())
        self.assertEqual(marker.read_text(), 'earlier unrelated file')
        self.assertEqual(list(self.local.iterdir()), [marker])




if __name__ == '__main__':
    unittest.main()
