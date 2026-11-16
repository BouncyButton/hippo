"""Storage failures must never discard the only completed checkpoint."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_hfh import medsam_archive as archive
from scripts.colab_hfh import medsam_recovery as recovery
from scripts.colab_hfh.medsam_search import controller_source


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'run'; self.output.mkdir()
        self.bundle = self.root/'bundle'; self.bundle.mkdir()
        self.cache = self.root/'cache'; self.cache.mkdir()
        self.protocol = self.root/'protocol.json'
        protocol = dict(train_cases=[f't{i}' for i in range(10)], evaluation_cases=[f'v{i}' for i in range(10)])
        self.protocol.write_text(json.dumps(protocol))
        (self.output/'protocol.json').write_text(json.dumps({**protocol, 'arm': 'dice'}))
        records = [dict(case_name=case, split=split, macro_hard_dice=.8)
            for split, names in [('training_support', protocol['train_cases']), ('validation', protocol['evaluation_cases'])]
            for case in names]
        evaluation = json.dumps(dict(records=records)).encode()
        (self.output/'evaluation.json').write_bytes(evaluation)
        (self.output/'training_history.json').write_text(json.dumps([dict(update=i+1) for i in range(200)]))
        self.checkpoint = self.output/'checkpoint_latest.pt'
        self.checkpoint.write_bytes(b'precious optimizer state')
        done = dict(status='complete', updates=200, fold=0, seed=0, validation_dice=.8,
                    evaluation_sha256=hashlib.sha256(evaluation).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        (self.output/'completion.json').write_text(json.dumps(done))
        archive.CONTEXT.clear()
        archive.CONTEXT.update(bundle=self.bundle, work=self.root, cache=self.cache,
            remote=Path('/content/hfh_storage/results/compact_archives'), inventory={})
        mock_rclone = patch.object(archive, 'rclone', return_value='[]')
        mock_rclone.start(); self.addCleanup(mock_rclone.stop)
        self.durable = Path('/content/hfh_storage/results/runs/medsam3_f0_s0')

    def commit(self):
        return archive.archive(self.output, self.durable, 'medsam3', 'medsam3', 0, 0, self.protocol)

    def test_upload_failure_preserves_checkpoint(self):
        with patch.object(archive, 'copy_verified', side_effect=IOError('upload failed')):
            with self.assertRaises(IOError): self.commit()
        self.assertTrue(self.checkpoint.exists())

    def test_verified_upload_precedes_retirement(self):
        def upload(source, destination):
            if Path(source).suffix == '.zip':
                self.assertTrue(self.checkpoint.exists())
                self.assertEqual(archive.verify_zip(source)['validation_dice'], .8)
        with patch.object(archive, 'copy_verified', side_effect=upload):
            result = self.commit()
        self.assertEqual(result['updates'], 200)
        self.assertFalse(self.checkpoint.exists())

    def test_incomplete_evaluation_never_uploads(self):
        (self.output/'training_history.json').write_text('[]')
        with patch.object(archive, 'copy_verified') as upload:
            with self.assertRaises(ValueError): self.commit()
        upload.assert_not_called()
        self.assertTrue(self.checkpoint.exists())

    def test_corrupt_archive_is_rejected(self):
        with patch.object(archive, 'copy_verified'):
            self.commit()
        path = self.cache/'medsam3_f0_s0.zip'
        corrupt = self.root/'corrupt.zip'
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(corrupt, 'w') as target:
            for item in source.infolist():
                target.writestr(item, b'{}' if item.filename == 'results/evaluation.json' else source.read(item))
        with self.assertRaises(ValueError): archive.verify_zip(corrupt)

    def test_remote_upload_uses_cached_folder_and_checks_hash(self):
        recovery.REMOTE_FOLDERS.clear()
        calls = []
        def fake(*args):
            calls.append(args)
            if args[0] != 'lsjson': return ''
            if '--hash' not in args: return json.dumps(dict(IsDir=True, ID='exact-directory-id'))
            return json.dumps(dict(Size=self.checkpoint.stat().st_size,
                Hashes={'MD5': hashlib.md5(self.checkpoint.read_bytes()).hexdigest()}))
        with patch.object(recovery, 'rclone', side_effect=fake):
            recovery.copy_verified(self.checkpoint, self.durable/'checkpoint.pt')
            recovery.copy_verified(self.checkpoint, self.durable/'checkpoint.pt')
        self.assertEqual(sum(c[0] == 'mkdir' for c in calls), 1)
        for call in (c for c in calls if c[0] == 'copyto'):
            self.assertIn('exact-directory-id', call)

    def test_existing_scheduler_integration_compiles(self):
        source = controller_source()
        compile(source, '<compact-controller>', 'exec')
        self.assertIn('configure(args.bundle, args.work, args.durable)', source)
        self.assertNotIn("copy_verified(output/'worker.log'", source)

    def test_missing_directory_id_falls_back_without_skipping_verification(self):
        recovery.REMOTE_FOLDERS.clear()
        calls = []
        def fake(*args):
            calls.append(args)
            if args[0] != 'lsjson': return ''
            if '--hash' not in args: return json.dumps(dict(IsDir=True))
            return json.dumps(dict(Size=self.checkpoint.stat().st_size,
                Hashes={'MD5': hashlib.md5(self.checkpoint.read_bytes()).hexdigest()}))
        with patch.object(recovery, 'rclone', side_effect=fake):
            recovery.copy_verified(self.checkpoint, self.durable/'checkpoint.pt')
        upload = next(c for c in calls if c[0] == 'copyto')
        self.assertIn('hfh_storage:results/runs/medsam3_f0_s0/checkpoint.pt', upload)
        self.assertNotIn('--drive-root-folder-id', upload)
        self.assertTrue(any('--hash' in c for c in calls))

    def test_directory_id_resolves_from_exact_parent_listing(self):
        recovery.REMOTE_FOLDERS.clear()
        calls = []
        def fake(*args):
            calls.append(args)
            if args[0] != 'lsjson': return ''
            if '--dirs-only' in args:
                return json.dumps([dict(Name='other', ID='wrong-folder'),
                                   dict(Name=self.durable.name, ID='correct-folder')])
            if '--hash' not in args: return json.dumps(dict(IsDir=True))
            return json.dumps(dict(Size=self.checkpoint.stat().st_size,
                Hashes={'MD5': hashlib.md5(self.checkpoint.read_bytes()).hexdigest()}))
        with patch.object(recovery, 'rclone', side_effect=fake):
            recovery.copy_verified(self.checkpoint, self.durable/'checkpoint.pt')
        upload = next(c for c in calls if c[0] == 'copyto')
        self.assertIn('correct-folder', upload)
        self.assertNotIn('wrong-folder', upload)
        self.assertIn('hfh_storage:checkpoint.pt', upload)


if __name__ == '__main__':
    unittest.main()
