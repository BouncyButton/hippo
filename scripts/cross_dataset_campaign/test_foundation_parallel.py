"""Concurrency, GPU isolation, archive safety, handoff, and actual cohort checks."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import foundation_two_sample_job as job
from scripts.cross_dataset_campaign import foundation_parallel as parallel
from scripts.cross_dataset_campaign.common import read, write


class ParallelTests(unittest.TestCase):
    def exercise(self, *, offline=False, pause=False, failed=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stage = Path(directory)
            write(stage/'control.json', dict(pause=False))
            plan = dict(tasks=job.tasks(), foundation_python='python',
                parallelism=dict(datasets=list(job.DATASETS)), storage=dict(minimum_free_gib=10), archive_wait_timeout_seconds=3)
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
                    self.assertIn('scripts.cross_dataset_campaign.foundation_gpu_worker', args)
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
        self.assertEqual(len(launches), 6); self.assertEqual(len(cleaned), 6)
        self.assertTrue(any(len(workers) == 2 for _,workers in published))
        for dataset in job.DATASETS:
            self.assertEqual([t['arm'] for t in launches if t['dataset'] == dataset], list(job.ARMS))

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
                    parallel.verify_gpu_probe(stage, dict(foundation_python='python'))

    def test_probe_requests_two_tasks_with_one_gpu_each(self):
        args = parallel.step_prefix(2)
        self.assertIn('--ntasks=2', args); self.assertIn('--gres=gpu:4g.40gb:2', args)
        self.assertIn('--gpus-per-task=4g.40gb:1', args)
        self.assertIn('--mem=80G', args)

    def test_submit_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory); write(stage/'submissions/123.json', dict(job_id='123'))
            with patch.object(job.controller,'queue',return_value=[['123','PENDING','next']]), patch.object(job,'replace_queued') as replace:
                self.assertEqual(job.submit(stage), dict(job_id='123')); replace.assert_not_called()

    def test_unrelated_pending_job_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stage = Path(directory)/'next'; old=Path(directory)/'old'; parent=Path(directory)/'parent'
            write(stage/'preflight.json', dict(status='passed',manifest_sha256='hash'))
            write(old/'submission.json', dict(job_id='702445',manifest_sha256=job.PREVIOUS_SHA))
            plan=dict(supersedes=dict(stage=str(old)),storage=dict(minimum_free_gib=10))
            stack.enter_context(patch.object(job,'verify',return_value=plan))
            stack.enter_context(patch.object(job,'check_predecessor',return_value=(parent,{})))
            stack.enter_context(patch.object(job.previous,'verify'))
            stack.enter_context(patch.object(job,'digest',side_effect=lambda p:job.PREVIOUS_SHA if p.parent==old else 'hash'))
            stack.enter_context(patch.object(job.controller,'quota',return_value=dict(free_gib=50)))
            stack.enter_context(patch.object(job.controller,'queue',return_value=[['different','PENDING','unrelated']]))
            cancel=stack.enter_context(patch.object(job.subprocess,'run'))
            with self.assertRaisesRegex(RuntimeError,'no longer pending'):job.replace_queued(stage)
            cancel.assert_not_called()

    def test_parent_archive_wait_blocks_both_gpu_workers(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stage=Path(directory)/'next'; parent=Path(directory)/'parent'
            for p in (stage,parent):write(p/'control.json',dict(pause=False))
            write(parent/'status.json',dict(status='waiting_for_archive_offline'))
            stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'123'}))
            stack.enter_context(patch.object(job,'verify',return_value=dict(graceful_stop_seconds=85200)))
            stack.enter_context(patch.object(job,'check_predecessor',return_value=(parent,{})))
            stack.enter_context(patch.object(job,'predecessor_complete',return_value=False))
            for name in ('submit','cancel_successor','verify','command_for'):
                stack.enter_context(patch.object(job.controller,name))
            stack.enter_context(patch.object(job.controller,'run'))
            summary=stack.enter_context(patch.object(job.controller,'summarize'))
            workers=stack.enter_context(patch.object(job.parallel,'run'))
            job.run(stage);workers.assert_not_called()
            self.assertEqual(summary.call_args.args[2],'waiting_for_archive_offline')


def support_check(stage):
    import torch
    from scripts.cross_dataset_campaign.foundation import Volumes, SupportSlabs
    from scripts.adni_baselines.core import case_name
    torch.set_num_threads(2)
    job.verify(stage); results=[]
    for dataset in job.DATASETS:
        p=read(stage/'protocols'/dataset/'protocol.json'); volumes=Volumes(p)
        if dataset=='ADNI':
            by_case={case_name(r):str(r.subject_id) for r in volumes.dataframe.itertuples()}
            if ([by_case[c] for c in p['train_cases']] != p['support_subjects']
                    or set(p['support_subjects']) & {by_case[c] for c in p['evaluation_cases']}):
                raise ValueError('ADNI sample-to-subject mapping or held-out split differs')
        schedules=[]
        for arm in job.ARMS:
            protocol={**p,'arm':arm}; support=SupportSlabs(volumes,protocol,None)
            schedule=support.schedule(protocol); calibration=support.calibration_schedule(4)
            if len(support.inventory)!=2 or len(schedule)!=400 or len(calibration)!=8:
                raise ValueError('Wrong two-sample training/calibration inventory')
            if schedule!=support.schedule(protocol) or {case for case,_ in schedule}!=set(p['train_cases']):
                raise ValueError('Training sampling differs')
            schedules.append(schedule)
            results.append(dict(dataset=dataset,arm=arm,inventory=support.inventory,training_pairs=400,calibration_pairs=8))
        if any(s!=schedules[0] for s in schedules):raise ValueError('Unmatched sampling across arms')
    write(stage/'support_preflight.json',dict(status='passed',datasets=results))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path);args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ParallelTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    if args.stage:support_check(args.stage)


if __name__=='__main__':main()
