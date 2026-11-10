import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import migrate_account_20261021 as migration


class FakeCloud:
    folder = migration.DESTINATION_RESULTS

    def __init__(self, files):
        self.files = files
        self.target = {}
        self.writes = 0
        self.after_copy = None

    def check_account(self):
        return {}

    def download(self, ident, path, sha=None):
        data = self.files[ident]
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('checksum mismatch')
        Path(path).write_bytes(data)

    def children(self):
        return list(self.target.values())

    def put_immutable(self, path, kind, task):
        return self.upload(path, kind+'_'+task)

    def upload(self, path, name):
        self.writes += 1
        ident = 'destination-'+str(self.writes)
        self.files[ident] = Path(path).read_bytes()
        receipt = dict(id=ident, name=name, local_sha256=migration.digest(path),
                       parents=[self.folder])
        self.target[ident] = receipt
        if self.after_copy:
            self.after_copy(self)
            self.after_copy = None
        return receipt


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.charge = dict(step=25, protocol_sha256='protocol')
        self.state = dict(bundle_sha256=migration.BUNDLE_SHA,
                          completed={'finished': {'validation_dice': .7}},
                          charges={'running': self.charge.copy()}, objects={},
                          retired=[{'id': 'source-retired'}], binding={'gpu': 'original'})
        self.files = {}
        self.add_zip('completed/finished', {'completion.json': '{}'})
        self.add_zip('recovery/running', {
            'work/running/checkpoint_latest.pt': 'frozen checkpoint',
            'work/running/checkpoint_ledger.json': json.dumps(self.charge)})
        self.add_zip('progress/running', {
            'work/running/checkpoint_ledger.json': json.dumps(self.charge)})
        self.cloud = FakeCloud(self.files)

    def add_zip(self, key, members):
        path = self.root / 'make.zip'
        manifest = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in members.items()}
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in members.items():
                archive.writestr(name, value)
            archive.writestr('_snapshot_manifest.json', json.dumps(manifest))
        ident = 'source-'+key
        self.files[ident] = path.read_bytes()
        self.state['objects'][key] = dict(id=ident, local_sha256=migration.digest(path))

    def run_migration(self):
        data = json.dumps(self.state).encode()
        self.files[migration.SOURCE_STATE] = data
        return migration.migrate(self.cloud, self.root, hashlib.sha256(data).hexdigest())

    def test_preserves_science_and_uses_destination_owned_objects(self):
        original = copy.deepcopy(self.state)
        migrated = self.run_migration()
        for key in ('completed', 'charges', 'binding', 'bundle_sha256'):
            self.assertEqual(migrated[key], original[key])
        self.assertFalse(migrated['retired'])
        audit = migrated['account_migrations'][migration.EVENT]
        self.assertEqual(audit['original_retired'], original['retired'])
        self.assertTrue(all(r['id'].startswith('destination-') for r in migrated['objects'].values()))
        self.assertEqual(self.state, original)

    def test_repeat_preserves_new_progress_without_writes(self):
        self.run_migration()
        receipt = next(r for r in self.cloud.target.values() if r['name'] == 'campaign_state.json')
        current = json.loads(self.cloud.files[receipt['id']])
        current['completed']['newly-finished'] = {'validation_dice': .8}
        self.cloud.files[receipt['id']] = json.dumps(current).encode()
        before = self.cloud.writes
        self.assertIn('newly-finished', self.run_migration()['completed'])
        self.assertEqual(self.cloud.writes, before)

    def test_old_checkpoint_cannot_be_hidden_by_new_progress(self):
        old = dict(self.charge, step=20)
        self.add_zip('recovery/running', {
            'work/running/checkpoint_latest.pt': 'old checkpoint',
            'work/running/checkpoint_ledger.json': json.dumps(old)})
        with self.assertRaisesRegex(ValueError, 'charged updates'):
            self.run_migration()
        self.assertFalse(any(r['name'] == 'campaign_state.json' for r in self.cloud.target.values()))

    def test_source_restart_aborts_before_state_commit(self):
        self.cloud.after_copy = lambda c: c.files.__setitem__(migration.SOURCE_STATE, b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_migration()
        self.assertFalse(any(r['name'] == 'campaign_state.json' for r in self.cloud.target.values()))

    def test_incomplete_result_inventory_rejected_without_writes(self):
        del self.state['objects']['completed/finished']
        with self.assertRaisesRegex(ValueError, 'archive missing'):
            self.run_migration()
        self.assertEqual(self.cloud.writes, 0)

    def test_wrong_bundle_rejected_without_writes(self):
        self.state['bundle_sha256'] = 'another-recipe'
        with self.assertRaisesRegex(ValueError, 'scientific bundle'):
            self.run_migration()
        self.assertEqual(self.cloud.writes, 0)

    def test_changed_completed_result_rejected_on_second_run(self):
        self.run_migration()
        receipt = next(r for r in self.cloud.target.values() if r['name'] == 'campaign_state.json')
        current = json.loads(self.cloud.files[receipt['id']])
        current['completed']['finished']['validation_dice'] = .99
        self.cloud.files[receipt['id']] = json.dumps(current).encode()
        with self.assertRaisesRegex(ValueError, 'completed result changed'):
            self.run_migration()


if __name__ == '__main__':
    unittest.main()
