"""Verify resume integrity and prevent silent replay of charged attempts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_hfh import recipe_transport as transport


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root/'imported_recipe_archives'
        self.cache.mkdir()
        self.label = 'nnunet_lr0.1_poly_cap'
        self.name = 'nnunet_f0_s0'

    def make_archive(self, corrupt=False):
        folder = self.cache/self.label
        folder.mkdir()
        payload = b'{"validation_dice":0.75}'
        with zipfile.ZipFile(folder/(self.name+'.zip'), 'w') as z:
            z.writestr('results/result.json', payload)
            z.writestr('results/archive_manifest.json', json.dumps({
                'result.json': 'bad' if corrupt else hashlib.sha256(payload).hexdigest()}))
            z.writestr('protocol/protocol.json', '{}')

    @patch.object(transport.subprocess, 'run')
    def test_verified_archive_restores_without_training(self, run):
        run.return_value.returncode = 0
        self.make_archive()
        transport.restore(self.root, '')
        destination = self.root/'recipe_diagnostics'/self.label
        self.assertEqual(json.loads((destination/'archives'/self.name/'result.json').read_text())['validation_dice'], .75)
        self.assertTrue((destination/(self.name+'.remote_verified.json')).exists())

    @patch.object(transport.subprocess, 'run')
    def test_corrupt_archive_is_rejected(self, run):
        run.return_value.returncode = 0
        self.make_archive(corrupt=True)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            transport.restore(self.root, '')

    @patch.object(transport.subprocess, 'run')
    def test_lost_charged_run_requires_audit(self, run):
        run.return_value.returncode = 0
        (self.cache/'campaign_status.json').write_text(json.dumps(dict(
            status='running', current=dict(model='nnunet', lr=.1, fold=0, seed=0))))
        with self.assertRaisesRegex(RuntimeError, 'unfinished charged attempt'):
            transport.restore(self.root, '')


if __name__ == '__main__':
    unittest.main()
