import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.colab_msd_seed0220 import recovery_audit as audit


class RecoveryAuditTest(unittest.TestCase):
    def test_rate_limit_is_not_confused_with_permission_failure(self):
        class Response:
            status_code = 403
            def __init__(self, reason): self.reason = reason
            def json(self): return {'error': {'errors': [{'reason': self.reason}]}}
        self.assertTrue(audit.rate_limited(Response('userRateLimitExceeded')))
        self.assertFalse(audit.rate_limited(Response('insufficientPermissions')))

    def test_empty_lock_is_valid_but_empty_json_is_not(self):
        audit.verify_blob('run.lock', b'', dict(bytes=0, sha256=audit.sha(b'')))
        with self.assertRaises(ValueError):
            audit.verify_blob('config.json', b'', dict(bytes=0, sha256=audit.sha(b'')))

    def test_exact_duplicate_can_be_staged_without_modifying_source(self):
        value = b'{"dice": 0.75}\n'
        manifest = {'metadata': {'validation_latest.json': {'bytes': len(value), 'sha256': audit.sha(value)}}}
        blobs = {'validation_best.json': value}
        failures, candidates = audit.assess_files(manifest, blobs)
        self.assertEqual(failures, [])
        self.assertEqual(candidates['validation_latest.json']['source_file'], 'validation_best.json')
        self.assertEqual(list(blobs), ['validation_best.json'])

    def test_similar_json_is_not_an_exact_repair(self):
        value = b'{"dice": 0.75}\n'
        manifest = {'metadata': {'validation_latest.json': {'bytes': len(value), 'sha256': audit.sha(value)}}}
        failures, candidates = audit.assess_files(manifest, {'validation_best.json': b'{"dice":0.75}\n'})
        self.assertTrue(failures)
        self.assertEqual(candidates, {})

    def test_hash_mismatch_blocks_verification(self):
        value = b'abc'
        with self.assertRaises(ValueError):
            audit.verify_blob('metrics.csv', value, dict(bytes=3, sha256=audit.sha(b'xyz')))

    def test_unsafe_names_rejected(self):
        for name in ('../config.json', '/tmp/file', 'a/../b', '', 'a//b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.safe_name(name)

    def test_archive_reads_and_checks_ack_and_stages_only_locally(self):
        item = dict(id='MSD_model_dice_f0s0', model='model', arm='dice', fold=0, seed=0)
        metadata = {'config.json': b'{"augmentation":"none"}', 'run.lock': b''}
        manifest = {'task': item, 'metadata': {n: dict(bytes=len(v), sha256=audit.sha(v)) for n,v in metadata.items()}}
        content = dict(metadata)
        content['archive_manifest.json'] = json.dumps(manifest).encode()
        content['archive_acknowledgment.json'] = json.dumps(dict(status='verified', manifest_sha256=audit.sha(content['archive_manifest.json']))).encode()
        content['cleanup.json'] = b'{"status":"complete"}'
        class FakeCloud:
            def children(self, folder):
                return [dict(id=n, name=n, size=str(len(b)), mimeType='application/json') for n,b in content.items()]
            def blob(self, file_id):
                return content[file_id]
        before = dict(content)
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.audit_run(FakeCloud(), item, {'id': 'folder'}, Path(tmp))
        self.assertEqual(result['status'], 'cloud_archive_hashes_verified')
        self.assertEqual(content, before)
        content['archive_acknowledgment.json'] = b'{"status":"verified","manifest_sha256":"wrong"}'
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.audit_run(FakeCloud(), item, {'id': 'folder'}, Path(tmp))
        self.assertEqual(result['status'], 'needs_recovery')


if __name__ == '__main__':
    unittest.main()
