import unittest
from types import SimpleNamespace
from unittest.mock import patch
import resume_other_account_20261021 as launcher


class SplitTests(unittest.TestCase):
    def setup_runner(self):
        self.calls = []
        runner = SimpleNamespace(tasks=lambda: [dict(id='colab'), dict(id='cluster')],
            run_one=lambda root, task, transport: self.calls.append(task['id']))
        launcher.apply_assignment(runner, dict(colab_tasks=['colab'], stop_compute_epoch=10000))
        return runner

    def test_only_colab_tasks_remain(self):
        self.assertEqual(self.setup_runner().tasks(), [dict(id='colab')])

    def test_short_window_defers_new_run(self):
        runner = self.setup_runner()
        with patch.object(launcher.time, 'time', return_value=9000), self.assertRaises(SystemExit) as exc:
            runner.run_one(None, dict(id='colab'), SimpleNamespace(state={'charges': {}}))
        self.assertEqual(exc.exception.code, 75)
        self.assertFalse(self.calls)

    def test_trained_evaluation_can_use_shorter_window(self):
        runner = self.setup_runner()
        with patch.object(launcher.time, 'time', return_value=9000):
            runner.run_one(None, dict(id='colab'), SimpleNamespace(state={'charges': {'colab': {'step': 25}}}))
        self.assertEqual(self.calls, ['colab'])

    def test_no_run_after_stop_deadline(self):
        runner = self.setup_runner()
        with patch.object(launcher.time, 'time', return_value=10001), self.assertRaises(SystemExit):
            runner.run_one(None, dict(id='colab'), SimpleNamespace(state={'charges': {'colab': {'step': 25}}}))
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
