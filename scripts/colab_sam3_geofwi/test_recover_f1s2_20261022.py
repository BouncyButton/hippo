import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import recover_f1s2_20261022 as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.charge = dict(step=1, protocol_sha256=recovery.PROTOCOL)
        self.transport = SimpleNamespace(
            root=self.root,
            state=dict(completed={}, charges={recovery.TASK: self.charge.copy()},
                       objects={'recovery/'+recovery.TASK: dict(
                           id=recovery.RECOVERY_ID, local_sha256=recovery.RECOVERY_SHA)}),
            cloud=SimpleNamespace(download=lambda *a: None),
            publish=lambda *a: None, save=lambda: None, probe=lambda: None)
        self.saved_states = []
        self.transport.save = lambda: self.saved_states.append(copy.deepcopy(self.transport.state))

    def unpack(self, archive, root):
        output = root/'work'/recovery.TASK
        output.mkdir(parents=True)
        history = []
        for name, value in {
            'checkpoint_ledger.json': dict(step=0, protocol_sha256=recovery.PROTOCOL),
            'checkpoint_latest.pt': dict(step=0, protocol_sha256=recovery.PROTOCOL, history=history),
            'training_history.json': history, 'protocol.json': {},
        }.items():
            (output/name).write_text(json.dumps(value))

    def run_repair(self):
        recovery.repair(self.transport, self.unpack,
                        lambda p: recovery.PROTOCOL if p.name == 'protocol.json' else 'checkpoint-hash',
                        lambda p: json.loads(p.read_text()))

    def test_recovery_preserves_completed_and_local_evidence(self):
        completed = {'other-run': {'validation_dice': .7}}
        self.transport.state['completed'] = completed.copy()
        output = self.root/'work'/recovery.TASK
        output.mkdir(parents=True)
        (output/'interruption.log').write_text('original evidence')
        self.run_repair()
        self.assertEqual(self.transport.state['charges'][recovery.TASK]['step'], 0)
        self.assertEqual(self.transport.state['completed'], completed)
        audit = self.root/'durable/recovery_audits'/recovery.EVENT
        self.assertEqual(len(list(audit.glob('local_before_*/interruption.log'))), 1)
        self.assertIn(recovery.EVENT, self.saved_states[0]['recovery_adjustments'])
        self.assertEqual(json.loads((output/'checkpoint_latest.pt').read_text())['step'], 0)

    def test_repeated_execution_never_rolls_back_again(self):
        self.run_repair()
        self.transport.state['charges'][recovery.TASK]['step'] = 14
        self.run_repair()
        self.assertEqual(self.transport.state['charges'][recovery.TASK]['step'], 14)
        self.assertEqual(len(self.saved_states), 1)

    def test_finished_run_is_untouched(self):
        self.transport.state['completed'][recovery.TASK] = {}
        self.run_repair()
        self.assertFalse(self.saved_states)

    def test_newer_local_checkpoint_is_preserved(self):
        output = self.root/'work'/recovery.TASK
        output.mkdir(parents=True)
        for name in ('checkpoint_ledger.json', 'checkpoint_latest.pt'):
            (output/name).write_text(json.dumps(self.charge))
        self.run_repair()
        self.assertFalse(self.saved_states)
        self.assertEqual(self.transport.state['charges'][recovery.TASK]['step'], 1)

    def test_changed_charge_rejected(self):
        self.transport.state['charges'][recovery.TASK]['step'] = 11
        with self.assertRaisesRegex(RuntimeError, 'state changed'):
            self.run_repair()
        self.assertFalse(self.saved_states)

    def test_changed_archive_rejected(self):
        self.transport.state['objects']['recovery/'+recovery.TASK]['id'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'archive changed'):
            self.run_repair()

    def test_failed_audit_upload_does_not_rewind_remote_charge(self):
        def fail(*args):
            raise IOError('upload failed')
        self.transport.publish = fail
        with self.assertRaises(IOError):
            self.run_repair()
        self.assertEqual(self.transport.state['charges'][recovery.TASK]['step'], 1)
        self.assertFalse(self.saved_states)


if __name__ == '__main__':
    unittest.main()
