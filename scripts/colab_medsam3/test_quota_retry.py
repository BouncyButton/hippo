"""Quota failures must not discard work, bypass headroom or duplicate runs."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.colab_medsam3 import parallel as p
from scripts.colab_medsam3.test_parallel import ProcessTests, gpu


def http_error(status=403, reason='rateLimitExceeded'):
    error = RuntimeError('simulated quota response')
    error.resp = SimpleNamespace(status=status)
    error.content = json.dumps({'error': {'errors': [{'reason': reason}]}}).encode()
    return error


class QuotaRetryTests(unittest.TestCase):
    def test_only_temporary_errors_retry(self):
        for status, reason, expected in [(403, 'rateLimitExceeded', True),
                (403, 'userRateLimitExceeded', True), (429, '', True), (503, '', True),
                (403, 'forbidden', False), (403, 'storageQuotaExceeded', False), (401, '', False)]:
            with self.subTest(status=status, reason=reason):
                self.assertEqual(p.transient_quota_error(http_error(status, reason)), expected)
        self.assertFalse(p.transient_quota_error(RuntimeError('real storage shortage')))

    def test_backoff_does_not_launch_or_read_stale_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(quota=Mock(side_effect=http_error()), read=Mock())
            supervisor = p.Supervisor(runtime, directory, directory)
            with patch.object(supervisor, 'event'), patch.object(p.time, 'monotonic', return_value=100), \
                    patch.object(p, 'prepare_task') as prepare:
                supervisor.launch({'id': 'pending'})
                self.assertGreater(supervisor.quota_retry_at, 100)
                supervisor.launch({'id': 'pending'})
                runtime.quota.assert_called_once()
                runtime.read.assert_not_called()
                prepare.assert_not_called()
                self.assertEqual(supervisor.launched, 0)
                self.assertFalse(supervisor.stopping)
                supervisor.request_stop()
                supervisor.quota_retry_at = 0
                supervisor.launch({'id': 'pending'})
                runtime.quota.assert_called_once()

    def test_retry_exhaustion_and_permanent_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            for error, failures in [(http_error(), 10), (http_error(403, 'forbidden'), 0),
                                    (RuntimeError('insufficient storage'), 0)]:
                supervisor = p.Supervisor(SimpleNamespace(quota=Mock(side_effect=error)), directory, directory)
                supervisor.quota_failures = failures
                with self.assertRaises(RuntimeError):
                    supervisor.launch({'id': 'pending'})

    def test_recovered_check_continues_without_replaying_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root, durable, runtime, tasks = ProcessTests().fixture(directory, count=1)
            real_quota = runtime.quota
            runtime.quota = Mock(side_effect=[http_error(), None])
            def quota(path):
                runtime.quota(path)
                real_quota(path)
            wrapped = SimpleNamespace(**vars(runtime))
            wrapped.quota = quota
            supervisor = p.Supervisor(wrapped, root, durable, poll=.005, hours=1)
            def telemetry():
                # Advance only the retry clock, avoiding a real wait in this test.
                supervisor.quota_retry_at = 0
                return gpu()
            supervisor.telemetry = telemetry
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=100*1024**3)):
                supervisor.run()
            self.assertEqual(supervisor.launched, 1)
            self.assertEqual(supervisor.finished, {tasks[0]['id']})
            self.assertEqual(runtime.quota.call_count, 2)
            runtime.quota.reset_mock()
            again = p.Supervisor(wrapped, root, durable, poll=.005, telemetry=gpu)
            again.run()
            self.assertEqual(again.launched, 0)
            runtime.quota.assert_not_called()


if __name__ == '__main__':
    unittest.main()
