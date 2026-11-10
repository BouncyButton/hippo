"""Concurrency, GPU isolation, archive safety, handoff, and actual cohort checks."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import seed3_baselines_job as job
from scripts.cross_dataset_campaign import seed3_baselines_parallel as parallel
from scripts.cross_dataset_campaign.common import read, write


class ParallelTests(unittest.TestCase):
    def exercise(self, *, offline=False, pause=False, failed=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stage = Path(directory)
            write(stage/'control.json', dict(pause=False))
            plan = dict(tasks=job.tasks(), conventional_python='python',
                parallelism=dict(datasets=list(tuple(job.BUNDLES))), storage=dict(minimum_free_gib=10), archive_wait_timeout_seconds=3)
            stack.enter_context(patch.object(parallel.controller, 'command_for', side_effect=lambda stage, plan, task, base: ['python', '-m', 'trainer', '--output', str(stage/'runs'/task['id']), '--seed', '3', '--objective', 'dice']))
            now = [0]; launches = []; processes = []; published = []; cleaned = []
            stack.enter_context(patch.object(parallel.time, 'monotonic', side_effect=lambda:now[0]))
            stack.enter_context(patch.object(parallel.time, 'sleep', side_effect=lambda _:now.__setitem__(0,now[0]+1)))
            stack.enter_context(patch.object(parallel, 'quota', return_value=dict(free_gib=50)))
            def stopping():
                if pause and now[0] >= 1:
                    write(stage/'control.json', dict(pause=True)); return True
                return False
            stack.enter_context(patch.object(parallel, 'stopping', side_effect=stopping))
            def prepare(output, task):
                if not offline: write(output/'archive_acknowledgment.json', dict(status='verified'))
            def retire(output):
                self.assertTrue((output/'archive_acknowledgment.json').exists())
                write(output/'cleanup.json', dict(status='complete')); cleaned.append(output.name)
            stack.enter_context(patch.object(parallel, 'prepare', side_effect=prepare))
            stack.enter_context(patch.object(parallel, 'retire', side_effect=retire))
            class Process:
                def __init__(inner, args, **kwargs):
                    inner.output = Path(args[args.index('--output')+1]); inner.start = now[0]
                    inner.code = None; inner.signals = []
                    task = next(t for t in plan['tasks'] if t['id'] == inner.output.name)
                    previous = [t for t in plan['tasks'] if t['dataset'] == task['dataset']]
                    index = previous.index(task)
                    if index: self.assertIn(previous[index-1]['id'], cleaned)
                    self.assertIn('--gres=gpu:4g.40gb:1', args)
                    self.assertIn('--gpus-per-task=4g.40gb:1', args)
                    self.assertIn('--ntasks=1', args)
                    self.assertIn('--mem=40G', args)
                    self.assertIn('--exclusive', args)
                    self.assertIn('scripts.cross_dataset_campaign.seed3_baselines_gpu_worker', args)
                    launches.append(task); processes.append(inner)
                    self.assertLessEqual(sum(p.code is None for p in processes), 2)
                def poll(inner):
                    if inner.code is None and now[0]-inner.start >= 2:
                        inner.code = 1 if failed and inner.output.name == plan['tasks'][0]['id'] else 0
                        if inner.code == 0: write(inner.output/'completion.json', dict(status='complete'))
                    return inner.code
                def send_signal(inner, sig): inner.signals.append(sig); inner.code = 75
                def wait(inner, timeout=None): return inner.poll()
                def terminate(inner): inner.code = -15
            stack.enter_context(patch.object(parallel.subprocess, 'Popen', side_effect=Process))
            def publish(status, workers):
                published.append((status, {i:w['task']['id'] for i,w in workers.items()}))
            if failed:
                with self.assertRaisesRegex(RuntimeError, 'trainer exited 1'):
                    parallel.dispatch(stage, plan, Path('/tmp/base'), publish)
                result = 'failed'
            else:
                result = parallel.dispatch(stage, plan, Path('/tmp/base'), publish)
            return result, launches, published, cleaned, processes

    def test_two_experiments_overlap_with_per_dataset_order_and_cleanup(self):
        result, launches, published, cleaned, _ = self.exercise()
        self.assertEqual(result, 'complete')
        self.assertEqual(len(launches), 24); self.assertEqual(len(cleaned), 24)
        self.assertTrue(any(len(workers) == 2 for _,workers in published))
        for dataset in tuple(job.BUNDLES):
            self.assertEqual([t['id'] for t in launches if t['dataset'] == dataset], [t['id'] for t in job.tasks() if t['dataset'] == dataset])

    def test_offline_archive_preserves_both_results_and_stops_new_launches(self):
        result, launches, _, cleaned, _ = self.exercise(offline=True)
        self.assertEqual(result, 'waiting_for_archive_offline')
        self.assertEqual(len(launches), 2); self.assertEqual(cleaned, [])

    def test_manual_pause_signals_both_workers_and_preserves_recovery(self):
        result, launches, _, cleaned, processes = self.exercise(pause=True)
        self.assertEqual(result, 'paused'); self.assertEqual(len(launches), 2)
        self.assertEqual(cleaned, []); self.assertTrue(all(p.signals for p in processes))

    def test_failure_does_not_launch_remaining_arms(self):
        result, launches, _, _, _ = self.exercise(failed=True)
        self.assertEqual(result, 'failed'); self.assertEqual(len(launches), 2)

    def test_gpu_probe_rejects_duplicate_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            for i in range(2):
                write(stage/f'gpu_probe/123/{i}.json', dict(status='passed',visible_devices=1,slurm_job_id='123',device_uuid='same'))
            with patch.dict('os.environ', {'SLURM_JOB_ID':'123'}), patch.object(parallel.subprocess, 'run'):
                with self.assertRaisesRegex(RuntimeError, 'distinct'):
                    parallel.verify_gpu_probe(stage, dict(conventional_python='python'))

    def test_probe_requests_two_tasks_with_one_gpu_each(self):
        args = parallel.step_prefix(2)
        self.assertIn('--ntasks=2', args); self.assertIn('--gres=gpu:4g.40gb:2', args)
        self.assertIn('--gpus-per-task=4g.40gb:1', args)
        self.assertIn('--mem=80G', args)


if __name__ == "__main__":
    unittest.main()
