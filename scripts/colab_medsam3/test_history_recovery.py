"""Recover history sidecars without modifying checkpoints or replaying work."""
from pathlib import Path
import tempfile
import unittest
import torch

from scripts.colab_medsam3 import parallel as p, runtime as rt


class HistoryRecoveryTests(unittest.TestCase):
    def fixture(self, directory):
        output, recovery = Path(directory)/'output', Path(directory)/'recovery'
        output.mkdir(); recovery.mkdir()
        rt.write(output/'protocol.json', {'updates': 200, 'augmentation': 'none'})
        sha = rt.digest(output/'protocol.json')
        history = [{'update': i, 'native': 1./i, 'total': 1./i} for i in range(1, 201)]
        torch.save({'step': 200, 'protocol_sha256': sha, 'history': history}, output/'checkpoint_latest.pt')
        ledger = {'step': 200, 'protocol_sha256': sha}
        rt.write(output/'checkpoint_ledger.json', ledger)
        rt.write(recovery/'charged_ledger.json', ledger)
        rt.write(output/'completion.json', {'checkpoint_sha256': rt.digest(output/'checkpoint_latest.pt')})
        rt.write(output/'evaluation.json', {'records': [{'case': 'finished', 'dice': .8}]})
        return output, recovery, history

    def test_exact_history_is_restored_without_touching_finished_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output, recovery, history = self.fixture(directory)
            original = {f.name: rt.digest(f) for f in output.iterdir()}
            p.restore_training_history(rt, output, recovery)
            self.assertEqual(rt.read(output/'training_history.json'), history)
            self.assertEqual({n: rt.digest(output/n) for n in original}, original)
            # Existing history is never replaced by this recovery helper.
            before = (output/'training_history.json').stat().st_mtime_ns
            p.restore_training_history(rt, output, recovery)
            self.assertEqual((output/'training_history.json').stat().st_mtime_ns, before)

    def test_protocol_ledger_and_completion_mismatches_block(self):
        for field in ('protocol', 'charge', 'completion', 'history'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output, recovery, _ = self.fixture(directory)
                if field == 'protocol':
                    rt.write(output/'protocol.json', {'updates': 201})
                elif field == 'charge':
                    rt.write(recovery/'charged_ledger.json', {'step': 201})
                elif field == 'completion':
                    rt.write(output/'completion.json', {'checkpoint_sha256': 'wrong'})
                else:
                    state = torch.load(output/'checkpoint_latest.pt', weights_only=False)
                    state['history'][0]['update'] = 4
                    torch.save(state, output/'checkpoint_latest.pt')
                with self.assertRaises((ValueError, RuntimeError)):
                    p.restore_training_history(rt, output, recovery)
                self.assertFalse((output/'training_history.json').exists())


if __name__ == '__main__':
    unittest.main()
