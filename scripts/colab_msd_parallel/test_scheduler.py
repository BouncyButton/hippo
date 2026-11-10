"""CPU tests: task identity, resume, concurrency, failure drains and package preservation."""
import ast
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_msd import runtime as frozen
from scripts.colab_msd_parallel import scheduler as s
from scripts.colab_msd_parallel import local_storage as storage
from scripts.colab_msd_parallel.prepare import ORIGINAL, EXPORT, CONTROL_ZIP, NOTEBOOK, notebook
from scripts.colab_msd_parallel.worker import timed_call


def task(name, arm='dice', model='nnunet', seed=0):
    return dict(id=name, arm=arm, model=model, fold=0, seed=seed, protocol=model+'_s'+str(seed))


def sample():
    return dict(util=5., used=3*s.GIB, total=80*s.GIB, ram=100*s.GIB)


class PolicyTests(unittest.TestCase):
    def test_profile_compatibility_requires_exact_worker_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            durable = Path(tmp)
            folder = durable/'parallel/old'; folder.mkdir(parents=True)
            (folder/'worker.py').write_text('original observer')
            (folder/'scheduler.py').write_text('original scheduler')
            worker_sha = s.sha(folder/'worker.py')
            frozen.write(folder/'control_manifest.json', dict(scientific_bundle_sha256='science',
                files={name:s.sha(folder/name) for name in ('worker.py','scheduler.py')}))
            control_sha = s.sha(folder/'control_manifest.json')
            self.assertEqual(s.compatible_profile_controls(durable,'science',[worker_sha]), {control_sha})
            self.assertEqual(s.compatible_profile_controls(durable,'different',[worker_sha]), set())
            self.assertEqual(s.compatible_profile_controls(durable,'science',['unapproved']), set())
            (folder/'worker.py').write_text('changed observer')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                s.compatible_profile_controls(durable,'science',[worker_sha])

    def test_recovery_index_skips_absent_drive_paths_but_preserves_existing_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'bundle'; durable = Path(tmp)/'drive'
            (root/'work/local').mkdir(parents=True)
            (root/'work/local/budget_ledger.json').write_text('{}')
            frozen.write(durable/'recovery/saved/latest.json', dict(file='checkpoint.tar'))
            names = s.directory_names(durable/'recovery')
            self.assertEqual(names, {'saved'})
            real_exists = Path.exists
            def guarded_exists(path):
                if durable/'recovery' in path.parents and path.parent.name not in names:
                    self.fail('Unnecessary absent recovery probe: '+str(path))
                return real_exists(path)
            with patch.object(Path, 'exists', guarded_exists):
                self.assertTrue(s.has_work(root, durable, task('local'), names))
                self.assertTrue(s.has_work(root, durable, task('saved'), names))
                self.assertFalse(s.has_work(root, durable, task('fresh'), names))
            self.assertEqual(s.directory_names(durable/'absent'), set())
            with patch.object(s.os, 'scandir', side_effect=PermissionError('unreadable Drive')):
                with self.assertRaises(PermissionError):
                    s.directory_names(durable/'recovery')

    def test_startup_heartbeat_reports_main_thread_wait_using_only_local_status(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            path = Path(tmp)/'startup.json'
            with s.StartupProgress(path, interval=.01) as progress:
                progress.update('verifying saved archives', current_task='old_run', checked=4, verified=4)
                threading.Event().wait(.05)
            value = json.loads(path.read_text())
            self.assertEqual(value['current_task'], 'old_run')
            self.assertEqual(value['verified'], 4)
            self.assertGreaterEqual(output.getvalue().count('old_run'), 2)
            self.assertFalse(progress.thread.is_alive())

    def test_resume_audit_preserves_and_blocks_lost_charged_updates(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'work'; output.mkdir(); recovery=root/'drive'; recovery.mkdir()
            ledger=dict(charged_attempts=3,calibration_attempts=0)
            frozen.write(output/'budget_ledger.json',ledger)
            frozen.write(recovery/'charged_ledger.json',ledger)
            torch.save(dict(step=2,charged_attempts=2),output/'checkpoint_latest.pt')
            before=(output/'checkpoint_latest.pt').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'behind charged updates'):
                s.verify_resume(frozen,output,recovery)
            self.assertEqual((output/'checkpoint_latest.pt').read_bytes(),before)
            torch.save(dict(step=3,charged_attempts=3),output/'checkpoint_latest.pt')
            s.verify_resume(frozen,output,recovery)
            ledger['calibration_attempts']=1
            frozen.write(output/'budget_ledger.json',ledger)
            frozen.write(recovery/'charged_ledger.json',ledger)
            with self.assertRaisesRegex(RuntimeError,'calibration is incomplete'):
                s.verify_resume(frozen,output,recovery)
            (output/'checkpoint_latest.pt').unlink()
            ledger['charged_attempts']=0
            frozen.write(output/'budget_ledger.json',ledger)
            frozen.write(recovery/'charged_ledger.json',ledger)
            with self.assertRaisesRegex(RuntimeError,'Charged work without a checkpoint'):
                s.verify_resume(frozen,output,recovery)

    def test_unknown_memory_colocates_with_headroom_and_reserves_future_growth(self):
        item = task('a')
        self.assertTrue(s.fits(item, [], {}, sample()))
        active = [dict(task=task(str(i)), status={}) for i in range(4)]
        self.assertTrue(s.fits(item, active, {}, sample()))  # Five unknown workers on this A100.
        active.append(dict(task=task('fifth'), status={}))
        self.assertFalse(s.fits(item, active, {}, sample()))  # Sixth exceeds provisional headroom.

    def test_unknown_memory_uses_larger_same_model_and_live_peaks(self):
        item = task('new', arm='dice_bands')
        old = dict(peak_reserved_bytes=15*s.GIB, peak_rss_bytes=10*s.GIB)
        budget = s.task_reservation(item, {'nnunet/dice':old})
        self.assertEqual(budget, dict(gpu=32*s.GIB, ram=16*s.GIB))
        active = [dict(task=task('running'), status=dict(peak_reserved_bytes=30*s.GIB,
                                                      peak_rss_bytes=2*s.GIB, reserved_bytes=s.GIB))]
        self.assertFalse(s.fits(item, active, {}, sample()))

    def test_low_current_allocation_cannot_overcommit_future_peaks(self):
        item = task('a')
        profiles = {s.profile_key(item):dict(peak_reserved_bytes=20*s.GIB, peak_rss_bytes=2*s.GIB)}
        self.assertTrue(s.fits(item, [], profiles, sample()))
        self.assertFalse(s.fits(item, [dict(task=task('b'), status=dict(reserved_bytes=s.GIB))], profiles, sample()))

    def test_full_memory_and_ram_bounds(self):
        item = task('a')
        profiles = {s.profile_key(item):dict(peak_reserved_bytes=s.GIB, peak_rss_bytes=2*s.GIB)}
        for field, value in [('used', 75*s.GIB), ('ram', 2*s.GIB)]:
            info = sample(); info[field] = value
            self.assertFalse(s.fits(item, [], profiles, info))

    def test_resume_first_then_matched_baselines(self):
        resumed = task('interrupted', 'dice_bands', seed=1)
        baseline = task('baseline')
        constraint = task('new', 'dice_bands')
        self.assertEqual(s.candidates([baseline, constraint, resumed], {'interrupted'}, []), [resumed, baseline])
        self.assertEqual(s.candidates([baseline, constraint], set(), []), [baseline])
        self.assertEqual(s.candidates([constraint], set(), []), [])
        self.assertEqual(s.candidates([constraint], set(), [baseline]), [constraint])
        other_baseline = task('different', seed=2)
        self.assertEqual(s.candidates([constraint, other_baseline], set(), [baseline]), [other_baseline, constraint])

    def test_exact_trainer_arguments_keep_science_and_resume(self):
        root = Path('/content/hippo_msd_bundle'); item = task('case')
        self.assertEqual(s.trainer_arguments(root, item, dict(trainer_alias='nnunet')), [
            '--model','nnunet','--objective','dice','--fold','0','--seed','0',
            '--pkl',str(root/'data/msd_hippocampus_full.pkl'),'--output',str(root/'work/case'),
            '--protocol-dir',str(root/'protocols/nnunet_s0'),'--resume','--threads','2'])

    def test_timer_preserves_return_and_exception(self):
        events, stats = [], {}
        def operation(x):
            events.append(x)
            if x == 0:
                raise ValueError('same exception')
            return x+1
        wrapped = timed_call(stats, 'seconds', operation)
        self.assertEqual(wrapped(2), 3)
        with self.assertRaisesRegex(ValueError, 'same exception'):
            wrapped(0)
        self.assertEqual(events, [2,0])
        self.assertGreater(stats['seconds'], 0)


FAKE_WORKER = r'''
import argparse, json, os, signal, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--status',type=Path);p.add_argument('--control-sha');p.add_argument('rest',nargs=argparse.REMAINDER)
a=p.parse_args();out=Path(a.rest[a.rest.index('--output')+1]); control=Path(os.environ['HIPPO_CONTROL'])
paused=False
def stop(*_):
    global paused;paused=True
signal.signal(signal.SIGUSR1,stop)
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
def status(done=False):
    write(a.status,dict(ready=True,pid=os.getpid(),peak_reserved_bytes=1024**3,peak_rss_bytes=1024**3,
        reserved_bytes=1024**3,completed=done,observed_full_run=True,ledger=dict(charged_attempts=1,training_flops=10)))
status()
write(out/'started.json',dict(pid=os.getpid(),time=time.time()))
for step in range(30):
    time.sleep(.01)
    if paused or json.loads(control.read_text())['pause']:
        write(out/'saved.json',dict(status='coherent'))
        sys.exit(75)
    if out.name=='failure' and step==12:
        print('injected worker failure',flush=True);sys.exit(1)
write(out/'completion.json',dict(status='complete'))
status(True)
'''


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'bundle'; self.root.mkdir()
        self.durable = Path(self.tmp.name)/'drive'; self.durable.mkdir()
        self.control = Path(self.tmp.name)/'control'; self.control.mkdir()
        (self.root/'work').mkdir(); (self.root/'source').mkdir()
        (self.root/'bundle_manifest.json').write_text('{}')
        (self.control/'worker.py').write_text(FAKE_WORKER)
        (self.control/'scheduler.py').write_text('# fake scheduler record')
        frozen.write(self.control/'control_manifest.json', dict(scientific_bundle_sha256=s.sha(self.root/'bundle_manifest.json'),
            files={n:s.sha(self.control/n) for n in ('worker.py','scheduler.py')}))
        self.tasks = [task('a'), task('b',seed=1), task('c',seed=2), task('d',seed=3)]
        self.done = {}
        self.archive_calls = []
        self.restored = []
        self.runtime = SimpleNamespace(write=frozen.write, read=frozen.read, copy_verified=frozen.copy_verified,
            digest=frozen.digest,
            verify_bundle=lambda root:dict(tasks=self.tasks), binding=lambda *a:dict(bundle_sha256='fixed'),
            report=self.report, result_row=lambda durable,t:dict(t), completed=lambda durable,t:t['id'] in self.done,
            archive_completed=self.archive, restore_recovery=self.restore,
            child_environment=lambda root:dict(os.environ), check_drive_headroom=self.quota)
        frozen.write(self.durable/'preflight.json',dict(status='passed',bundle_sha256='fixed'))
        for item in self.tasks:
            frozen.write(self.root/'protocols'/item['protocol']/'protocol.json',dict(trainer_alias='nnunet'))
        self.patches = [patch.object(s, 'load_runtime', return_value=self.runtime),
            patch.object(s, 'cpu_capacity', return_value=8), patch.object(s, 'gpu_sample', side_effect=sample),
            patch.object(s.shutil, 'disk_usage', return_value=SimpleNamespace(free=200*s.GIB))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        self.sleep = time.sleep

    def report(self, root, durable, rows=None):
        return (list(self.done.values()) if rows is None else rows), []

    def archive(self, root, output, durable, recovery, item):
        self.assertTrue((output/'completion.json').exists())
        self.assertNotIn(item['id'],self.done)
        self.done[item['id']] = dict(item)
        self.archive_calls.append(item['id'])
        self.make_saved_archive(item)

    def restore(self, output, recovery):
        self.restored.append(output.name)
        output.mkdir(exist_ok=True)

    def quota(self, durable):
        frozen.write(durable/'storage_quota.json',dict(limit=200*s.GIB,usage=0))

    def runner(self, max_runs=504):
        return s.Sweep(self.root,self.durable,self.control,3,max_runs,1)

    def execute(self, sweep, callback=None):
        highest = [0]
        def fast_sleep(seconds):
            highest[0] = max(highest[0],len(sweep.active))
            if callback:
                callback(sweep)
            self.sleep(.02)
        with patch.object(s.time, 'sleep', side_effect=fast_sleep), contextlib.redirect_stdout(io.StringIO()):
            sweep.loop()
        return highest[0]

    def test_three_workers_unique_archives_and_existing_results_untouched(self):
        old = task('already_complete'); self.tasks.insert(0,old); self.done[old['id']]=dict(old)
        self.make_saved_archive(old)
        marker = self.durable/'old_result.json'; marker.write_bytes(b'old result must remain exact')
        before = marker.read_bytes()
        highest = self.execute(self.runner())
        self.assertEqual(highest,3)
        self.assertEqual(len(self.archive_calls),4)
        self.assertEqual(len(set(self.archive_calls)),4)
        self.assertNotIn('already_complete',self.restored)
        self.assertEqual(marker.read_bytes(),before)

    def test_max_runs_is_global_across_workers(self):
        self.execute(self.runner(max_runs=2))
        self.assertEqual(len(self.archive_calls),2)

    def test_five_unmeasured_constraint_workers_start_before_any_finishes_and_refill(self):
        baseline = task('matched_baseline')
        arms = ('dice_bands', 'dice_edge', 'dice_depth', 'dice_sdf', 'dice_bands_edge',
                'dice_bands_depth', 'dice_edge_depth')
        self.tasks = [baseline]+[task('new_'+str(i), arm=arm) for i,arm in enumerate(arms)]
        self.done[baseline['id']] = dict(baseline)
        self.make_saved_archive(baseline)
        with patch.object(s, 'cpu_capacity', return_value=5):
            sweep = s.Sweep(self.root,self.durable,self.control,0,504,1)
        launched = []
        original_launch = sweep.launch
        def launch(item):
            archived_before = len(self.archive_calls)
            accepted = original_launch(item)
            # A worker may finish between slot selection and the final launch
            # guard. A deferred attempt is not a second training process.
            if accepted:
                launched.append((item['id'], archived_before))
            return accepted
        with patch.object(sweep, 'launch', side_effect=launch):
            peak = self.execute(sweep)
        self.assertEqual(peak, 5)
        self.assertEqual(sweep.limit, 5)
        self.assertEqual([count for _,count in launched[:5]], [0]*5)
        self.assertEqual(len(launched), 7)
        self.assertEqual(len(set(self.archive_calls)), 7)
        self.assertNotIn('matched_baseline', self.archive_calls)

    def test_high_gpu_activity_and_zero_measured_throughput_do_not_reduce_target(self):
        sweep = self.runner()
        with contextlib.redirect_stdout(io.StringIO()):
            sweep.load()
            with patch.object(s, 'gpu_sample', return_value=dict(sample(), util=100.)):
                for _ in range(20):
                    sweep.observe()
        self.assertEqual(sweep.limit, 3)
        self.assertFalse(sweep.stop_reason)

    def test_stop_during_slot_fill_prevents_next_launch_and_saves_started_worker(self):
        sweep = self.runner()
        original_launch = sweep.launch
        def launch(item):
            result = original_launch(item)
            (self.durable/'STOP').touch()
            return result
        with patch.object(sweep, 'launch', side_effect=launch):
            self.execute(sweep)
        self.assertEqual(sweep.launched, 1)
        self.assertEqual(len(list((self.root/'work').glob('*/saved.json'))), 1)
        self.assertEqual(self.archive_calls, [])
        self.assertTrue((self.durable/'STOP').exists())

    def test_stop_during_recovery_never_starts_trainer(self):
        sweep = self.runner()
        original_restore = self.runtime.restore_recovery
        def restore(output, recovery):
            original_restore(output, recovery)
            sweep.interrupted = True
        self.runtime.restore_recovery = restore
        self.execute(sweep)
        self.assertEqual(sweep.launched, 0)
        self.assertFalse(list((self.root/'work').glob('*/started.json')))

    def test_worker_exit_during_slow_recovery_defers_launch_until_reaped(self):
        sweep = self.runner()
        with contextlib.redirect_stdout(io.StringIO()):
            sweep.load()
            exited = [False]
            sweep.active = [dict(proc=SimpleNamespace(poll=lambda:1 if exited[0] else None))]
            original_restore = self.runtime.restore_recovery
            def restore(output, recovery):
                original_restore(output, recovery)
                exited[0] = True
            self.runtime.restore_recovery = restore
            self.assertFalse(sweep.launch(self.tasks[0]))
        self.assertEqual(sweep.launched, 0)
        self.assertIn(self.tasks[0], sweep.pending)
        self.assertFalse(list((self.root/'work').glob('*/started.json')))

    def test_headroom_wait_keeps_target_and_retries_without_restarting_workers(self):
        sweep = self.runner()
        original_launch = sweep.launch
        admitted = []
        def launch(item):
            admitted.append(item['id'])
            return original_launch(item)
        def resources():
            # A memory shortage after two admissions prevents the third until
            # one worker finishes, while preserving both active jobs.
            return dict(sample(), ram=2*s.GIB) if len(sweep.active) >= 2 else sample()
        with patch.object(sweep, 'launch', side_effect=launch), patch.object(s, 'gpu_sample', side_effect=resources):
            peak = self.execute(sweep)
        self.assertEqual(peak, 2)
        self.assertEqual(sweep.limit, 3)
        self.assertEqual(len(admitted), len(set(admitted)))
        self.assertEqual(len(self.archive_calls), 4)

    def test_completed_matched_baseline_allows_other_constraints_during_baseline_tail(self):
        baseline = task('matched')
        self.done[baseline['id']] = dict(baseline); self.make_saved_archive(baseline)
        constraint = task('constraint', arm='dice_bands')
        self.tasks = [baseline, self.tasks[1], constraint]
        sweep = self.runner()
        before_archives = []
        original_launch = sweep.launch
        def launch(item):
            before_archives.append((item['id'], len(self.archive_calls)))
            return original_launch(item)
        with patch.object(sweep, 'launch', side_effect=launch):
            peak = self.execute(sweep)
        self.assertEqual(peak, 2)
        self.assertEqual(before_archives, [('b', 0), ('constraint', 0)])

    def test_stop_drains_all_workers_and_respects_existing_stop(self):
        sweep = self.runner()
        def stop_after_three(current):
            if len(current.active)==3:
                (self.durable/'STOP').touch()
        self.execute(sweep,stop_after_three)
        self.assertEqual(self.archive_calls,[])
        self.assertFalse(sweep.active)
        self.assertEqual(len(list((self.root/'work').glob('*/saved.json'))),3)
        # Reopening with the manual STOP must not launch anything or clear it.
        next_sweep = self.runner(); self.execute(next_sweep)
        self.assertEqual(next_sweep.launched,0)
        self.assertTrue((self.durable/'STOP').exists())

    def test_failure_saves_other_workers_without_starting_next(self):
        self.tasks[0]['id']='failure'
        sweep = self.runner()
        with self.assertRaisesRegex(RuntimeError,'injected worker failure'):
            self.execute(sweep)
        self.assertFalse(sweep.active)
        self.assertEqual(len(list((self.root/'work').glob('*/saved.json'))),2)
        self.assertFalse((self.root/'work/d/started.json').exists())

    def test_existing_resume_launches_first_without_blocking_spare_slots(self):
        output = self.root/'work/c'; output.mkdir()
        frozen.write(output/'budget_ledger.json',dict(charged_attempts=1))
        import torch
        torch.save(dict(step=1,charged_attempts=1),output/'checkpoint_latest.pt')
        sweep = self.runner()
        with patch.object(sweep, 'launch', wraps=sweep.launch) as launches:
            highest = self.execute(sweep)
        self.assertEqual(highest, 3)
        self.assertEqual(launches.call_args_list[0].args[0]['id'], 'c')
        self.assertNotIn('c',self.restored)

    def test_archive_failure_still_drains_remaining_children(self):
        calls = []
        def fail(*args):
            calls.append(args[-1]['id'])
            raise IOError('archive upload failed')
        self.runtime.archive_completed=fail
        sweep=self.runner()
        with self.assertRaisesRegex(RuntimeError,'archive upload failed'):
            self.execute(sweep)
        self.assertFalse(sweep.active)
        self.assertTrue(calls)

    def test_lock_contends_with_original_runner(self):
        with (self.root/'work/sweep.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            args=['scheduler.py','--bundle',str(self.root),'--durable',str(self.durable)]
            with patch.object(sys,'argv',args):
                with self.assertRaisesRegex(RuntimeError,'another adaptive sweep is running'):
                    s.main()

    def make_saved_archive(self, item):
        folder = self.durable/'runs'/item['id']
        frozen.write(folder/'metrics.json', dict(dice=.8))
        frozen.write(folder/'cleanup.json', dict(status='complete'))
        frozen.write(folder/'archive_manifest.json', dict(task=item, metadata={
            'metrics.json':dict(bytes=(folder/'metrics.json').stat().st_size,
                                sha256=frozen.digest(folder/'metrics.json'))}))
        frozen.write(folder/'archive_acknowledgment.json', dict(status='verified', source_verified=True,
            manifest_sha256=frozen.digest(folder/'archive_manifest.json'), source_manifest_sha256='fixed'))
        frozen.write(self.durable/'environment_binding.json', dict(bundle_sha256='fixed'))
        return folder

    def test_startup_verifies_only_present_archives_with_frozen_verifier(self):
        folder = self.make_saved_archive(self.tasks[0])
        frozen.write(folder/'concurrency_unlisted.json', dict(completed=True, observed_full_run=True,
            peak_reserved_bytes=s.GIB, peak_rss_bytes=s.GIB, control_sha256='irrelevant'))
        self.runtime.completed = frozen.completed
        with patch.object(self.runtime, 'completed', wraps=frozen.completed) as verify:
            sweep = s.Sweep(self.root,self.durable,self.control,0,504,1)
            sweep.load()
        self.assertEqual(verify.call_count, 1)
        self.assertNotEqual(verify.call_args.args[0], self.durable)
        self.assertFalse(list(sweep.archive_cache.iterdir()))
        self.assertEqual(sweep.rows, [self.tasks[0]])
        self.assertEqual(sweep.pending, self.tasks[1:])
        self.assertEqual(sweep.profiles, {})
        self.assertEqual(frozen.read(sweep.audit_dir/'preservation.json')['existing_verified_ids'], ['a'])

    def test_startup_corrupt_archive_blocks_training_and_preserves_comparison(self):
        folder = self.make_saved_archive(self.tasks[0])
        (folder/'metrics.json').write_text('{"dice": 0.9}')
        comparison = self.durable/'comparison.json'; comparison.write_text('previous report')
        self.runtime.completed = frozen.completed
        sweep = s.Sweep(self.root,self.durable,self.control,0,504,1)
        with self.assertRaisesRegex(ValueError, 'Archive differs'):
            sweep.loop()
        self.assertEqual(sweep.launched, 0)
        self.assertEqual(comparison.read_text(), 'previous report')
        self.assertFalse(sweep.audit_dir.exists())

    def test_interrupt_during_startup_stops_before_report_or_training(self):
        self.make_saved_archive(self.tasks[0])
        comparison = self.durable/'comparison.json'; comparison.write_text('previous report')
        sweep = s.Sweep(self.root,self.durable,self.control,0,504,1)
        def verify(durable, item):
            sweep.interrupted = True
            return frozen.completed(durable, item)
        self.runtime.completed = verify
        with patch.object(self.runtime, 'report', wraps=self.report) as report:
            sweep.loop()
        report.assert_not_called()
        self.assertEqual(sweep.launched, 0)
        self.assertEqual(comparison.read_text(), 'previous report')
        self.assertFalse(sweep.audit_dir.exists())

    def test_new_runs_use_nvme_but_existing_work_stays_in_place(self):
        store = Path(self.tmp.name)/'nvme/work'; store.mkdir(parents=True)
        existing = self.root/'work/c'; existing.mkdir()
        frozen.write(existing/'budget_ledger.json',dict(charged_attempts=1))
        import torch
        torch.save(dict(step=1,charged_attempts=1),existing/'checkpoint_latest.pt')
        sweep = self.runner()
        with patch.object(storage, 'choose_work_store', return_value=store):
            with patch.object(sweep, 'launch', wraps=sweep.launch) as launches:
                self.execute(sweep)
        self.assertFalse(existing.is_symlink())
        self.assertTrue((existing/'checkpoint_latest.pt').exists())
        self.assertTrue((self.root/'work/a').is_symlink())
        self.assertEqual((self.root/'work/a').resolve(), (store/'a').resolve())
        self.assertEqual(len(self.archive_calls),4)
        self.assertEqual(launches.call_args_list[0].args[0]['id'], 'c')


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'bundle';(self.root/'work').mkdir(parents=True)
        self.drive=Path(self.tmp.name)/'drive';self.drive.mkdir()
        self.mount=Path(self.tmp.name)/'nvme';self.mount.mkdir()

    def choose(self):
        real_stat=Path.stat; device=self.root.stat().st_dev
        def stat(path,*args,**kwargs):
            if path==self.mount:return SimpleNamespace(st_dev=device+1)
            return real_stat(path,*args,**kwargs)
        with patch.object(Path,'is_mount',lambda path:path==self.mount), patch.object(Path,'stat',stat), \
                patch.object(storage.shutil,'disk_usage',return_value=SimpleNamespace(free=100*storage.GIB)):
            return storage.choose_work_store(self.root,self.drive,'science',self.mount)

    def test_requires_separate_mounted_disk_and_preserves_owner(self):
        self.assertIsNone(storage.choose_work_store(self.root,self.drive,'science',self.mount))
        self.assertFalse((self.mount/'hippo_msd_science').exists())
        store=self.choose()
        self.assertEqual(store,self.mount/'hippo_msd_science/work')
        self.assertEqual(self.choose(),store)
        frozen.write(store.parent/'owner.json',dict(bundle='another study'))
        with self.assertRaisesRegex(RuntimeError,'different bundle'):
            self.choose()

    def test_new_target_reused_and_existing_checkpoint_not_moved(self):
        store=self.choose();item=task('new')
        output=storage.ensure_output(self.root,item,store)
        self.assertTrue(output.is_symlink())
        (output/'checkpoint_latest.pt').write_bytes(b'coherent checkpoint')
        self.assertEqual(storage.ensure_output(self.root,item,store),output)
        self.assertEqual((output/'checkpoint_latest.pt').read_bytes(),b'coherent checkpoint')
        old=self.root/'work/old';old.mkdir();(old/'checkpoint_latest.pt').write_bytes(b'old checkpoint')
        self.assertEqual(storage.ensure_output(self.root,task('old'),store),old)
        self.assertFalse(old.is_symlink())
        self.assertFalse((store/'old').exists())

    def test_missing_or_wrong_target_blocks_without_deleting_link(self):
        store=self.choose();output=storage.ensure_output(self.root,task('a'),store)
        with self.assertRaisesRegex(RuntimeError,'unavailable or unexpected'):
            storage.ensure_output(self.root,task('a'),None)
        (store/'a').rmdir()
        with self.assertRaisesRegex(RuntimeError,'unavailable or unexpected'):
            storage.ensure_output(self.root,task('a'),store)
        self.assertTrue(output.is_symlink())

    def test_frozen_archive_metadata_is_not_changed(self):
        store=self.choose();output=storage.ensure_output(self.root,task('a'),store)
        storage.record_location(output,store)
        self.assertEqual(frozen.read(output/'local_storage.json')['backend'],'local_nvme')
        (output/'local_storage.json').unlink()
        frozen.write(output/'archive_manifest.json',dict(metadata={}))
        storage.record_location(output,store)
        self.assertFalse((output/'local_storage.json').exists())

    def test_real_recovery_roundtrip_uses_physical_siblings_and_survives_new_runtime(self):
        store=self.choose();output=storage.ensure_output(self.root,task('a'),store)
        (output/'checkpoint_latest.pt').write_bytes(b'checkpoint, optimizer and RNG')
        frozen.write(output/'budget_ledger.json',dict(charged_attempts=7))
        recovery=self.drive/'recovery/a'
        frozen.write(recovery/'charged_ledger.json',dict(charged_attempts=7))
        writer=frozen.RecoveryWriter(output.resolve(),recovery)
        original_persist=writer._persist
        def persist(snapshot):
            self.assertEqual(snapshot.parent,store.resolve())
            return original_persist(snapshot)
        writer._persist=persist
        writer.close(force=True)
        restored=self.root/'restored'
        frozen.restore_recovery(restored,recovery)
        self.assertEqual((restored/'checkpoint_latest.pt').read_bytes(),b'checkpoint, optimizer and RNG')
        self.assertEqual(frozen.read(restored/'budget_ledger.json')['charged_attempts'],7)

    def test_frozen_cleanup_accepts_run_directory_link_and_deletes_only_verified_checkpoint(self):
        from scripts.cross_dataset_campaign import storage as archive_storage
        store=self.choose();output=storage.ensure_output(self.root,task('a'),store)
        frozen.write(output/'metrics.json',dict(dice=.8))
        (output/'checkpoint_latest.pt').write_bytes(b'verified checkpoint')
        def info(path):return dict(bytes=path.stat().st_size,sha256=frozen.digest(path))
        manifest=dict(metadata={'metrics.json':info(output/'metrics.json')},
                      deletion_candidates=[dict(name='checkpoint_latest.pt',**info(output/'checkpoint_latest.pt'))])
        frozen.write(output/'archive_manifest.json',manifest)
        frozen.write(output/'archive_acknowledgment.json',dict(status='verified',source_verified=True,
            manifest_sha256=frozen.digest(output/'archive_manifest.json'),
            durable_archive='/content/drive/MyDrive/test/runs/a'))
        result=archive_storage.retire(output)
        self.assertEqual(result['status'],'complete')
        self.assertFalse((store/'a/checkpoint_latest.pt').exists())
        self.assertTrue(output.is_symlink())
        self.assertEqual(frozen.read(output/'metrics.json'),dict(dice=.8))



class PackageTests(unittest.TestCase):
    def test_notebook_complete_and_keeps_existing_bundle_and_results(self):
        book = notebook('code','data','control','science')
        text = '\n'.join(cell['source'] for cell in book['cells'])
        for cell in book['cells']:
            if cell['cell_type']=='code':
                ast.parse(cell['source']); self.assertEqual(cell['outputs'],[])
        self.assertIn("RESULTS = DRIVE_ROOT/'results_noaug_v1'",text)
        self.assertIn("MAX_WORKERS = 0",text)
        self.assertIn("str(CONTROL/'scheduler.py')",text)
        self.assertIn("campaign('preflight')",text)
        self.assertIn('msd_max_workers_control.zip',text)
        self.assertIn('Scientific loss calibration still runs exactly as prescribed',text)
        self.assertNotIn('it runs alone',text)
        self.assertNotIn('backs off future launches',text)
        self.assertEqual(book['metadata']['colab']['name'], NOTEBOOK)
        self.assertNotIn('rmtree',text)

    def test_control_package_preserves_scientific_identity(self):
        old = frozen.read(ORIGINAL/'export_summary.json')
        new = frozen.read(EXPORT/'export_summary.json')
        self.assertEqual(old['bundle_manifest_sha256'],new['scientific_bundle_sha256'])
        self.assertEqual(new['extra_scientific_runs'],0)
        self.assertEqual(new['total_runs'],504)
        self.assertFalse(new['solo_memory_profiling'])
        self.assertFalse(new['throughput_backoff'])
        self.assertFalse(new['scientific_loss_calibration_changed'])
        self.assertEqual(new['control_zip_sha256'],s.sha(EXPORT/CONTROL_ZIP))
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(EXPORT/CONTROL_ZIP) as archive:
                archive.extractall(tmp)
            _, manifest=s.verify_control(Path(tmp))
            self.assertEqual(manifest['scientific_bundle_sha256'],old['bundle_manifest_sha256'])
        for name, expected in frozen.read(ORIGINAL/'bundle/bundle_manifest.json').items():
            self.assertEqual(s.sha(ORIGINAL/'bundle'/name),expected,name)


class WorkerTests(unittest.TestCase):
    def test_wrapper_preserves_cpu_loss_optimizer_rng_and_startup_stop(self):
        # Run the actual wrapper against a tiny CPU entry point exposing the same
        # storage hooks; compare parameters, optimizer state, RNG and call order.
        source = r'''
import json, os, sys, hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
trace=[]
def write(path,value):
    trace.append('write:'+Path(path).name);Path(path).write_text(json.dumps(value))
def read(path):
    trace.append('read:'+Path(path).name);return json.loads(Path(path).read_text())
class RecoveryWriter:
    def __init__(self,output,durable):
        assert not Path(output).is_symlink(), 'Recovery must resolve the physical checkpoint directory'
        self.output=output;self.durable=durable
    def capture(self,force=False):
        assert (self.output/'budget_ledger.json').exists()
        trace.append('capture')
    def close(self,force=True):self.capture(force=force)
def save_checkpoint(path,value):
    trace.append('checkpoint');torch.save(value,path)
base=SimpleNamespace(save_checkpoint=save_checkpoint)
scientific=SimpleNamespace(experiment=SimpleNamespace(base=base))
def main():
    out=Path(sys.argv[sys.argv.index('--output')+1]);durable=Path(os.environ['HIPPO_COLAB_RECOVERY'])
    torch.manual_seed(0);torch.set_num_threads(2)
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    writer=RecoveryWriter(out,durable)
    if os.environ.get('TEST_STARTUP_STOP'):
        base.save_checkpoint(out/'checkpoint_latest.pt',dict(step=0,charged_attempts=0))
        writer.close();raise SystemExit(75)
    ledger=dict(charged_attempts=1,training_flops=100)
    write(durable/'charged_ledger.json',ledger)
    assert read(durable/'charged_ledger.json')==ledger
    write(out/'budget_ledger.json',ledger)
    loss=model(torch.ones(1,3)).square().sum();loss.backward();optimizer.step();trace.append('update')
    base.save_checkpoint(out/'checkpoint_latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
    writer.capture();writer.close()
    signature=dict(loss=float(loss.detach()),weights=[v.detach().tolist() for v in model.parameters()],
        rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        optimizer=[{k:v.tolist() for k,v in item.items()} for item in optimizer.state.values()],trace=trace)
    (out/'signature.json').write_text(json.dumps(signature))
    (out/'completion.json').write_text('{"status":"complete"}')
if __name__=='__main__':main()
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bundle=root/'bundle';bundle.mkdir()
            (bundle/'entry.py').write_text(source)
            signatures=[]
            worker=Path(s.__file__).with_name('worker.py')
            for mode in ('original','wrapped','stop'):
                out=root/mode
                if mode=='wrapped':
                    physical=root/'physical_nvme_run';physical.mkdir();out.symlink_to(physical,target_is_directory=True)
                else:out.mkdir()
                durable=root/(mode+'_drive');durable.mkdir()
                args=['--output',str(out),'--protocol-dir',str(bundle)]
                if mode=='original':
                    command=[sys.executable,str(bundle/'entry.py'),*args]
                else:
                    command=[sys.executable,str(worker),'--bundle',str(bundle),'--status',str(root/(mode+'.json')),
                        '--control-sha','test','--',*args]
                env=dict(os.environ,HIPPO_COLAB_RECOVERY=str(durable))
                if mode=='stop':env['TEST_STARTUP_STOP']='1'
                proc=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(proc.returncode,75 if mode=='stop' else 0,proc.stderr)
                if mode!='stop':signatures.append(json.loads((out/'signature.json').read_text()))
                else:
                    info=json.loads((root/'stop.json').read_text())
                    self.assertTrue(info['uncharged_startup_snapshot_skipped'])
                    self.assertFalse(info['completed'])
                    self.assertFalse((durable/'charged_ledger.json').exists())
            self.assertEqual(signatures[0],signatures[1])


if __name__=='__main__':
    unittest.main()
