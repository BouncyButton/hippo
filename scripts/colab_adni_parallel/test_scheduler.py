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
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_adni import runtime as frozen
from scripts.colab_adni_parallel import scheduler as s
from scripts.colab_adni_parallel.prepare import ORIGINAL, EXPORT, notebook
from scripts.colab_adni_parallel.worker import timed_call, instrument_trainer
from scripts.colab_adni_parallel.telemetry import LiveStatus, directories
from scripts.colab_adni_parallel.archive_verification import verified_snapshot
from scripts.colab_adni_parallel import local_storage as storage
import threading


def task(name, arm='dice', model='nnunet', seed=0):
    return dict(id=name, arm=arm, model=model, fold=0, seed=seed, dataset='ADNI',
                kind='conventional', protocol=f'{model}_f0_s{seed}')


def sample():
    return dict(util=5., used=3*s.GIB, total=80*s.GIB, ram=100*s.GIB)


class PolicyTests(unittest.TestCase):
    def test_unknown_allowance_uses_same_adni_model_and_live_observations(self):
        item = task('new', 'dice_sdf')
        profiles = {'nnunet/dice':dict(peak_reserved_bytes=10*s.GIB, peak_rss_bytes=8*s.GIB),
                    'swinunetr/dice':dict(peak_reserved_bytes=70*s.GIB, peak_rss_bytes=90*s.GIB)}
        self.assertEqual(s.task_reservation(item, profiles)['gpu'], 22*s.GIB)
        self.assertEqual(s.task_reservation(item, profiles, dict(peak_reserved_bytes=15*s.GIB))['gpu'],32*s.GIB)

    def test_live_same_model_growth_blocks_candidate_even_if_unprofiled(self):
        active = [dict(task=task('running'),status=dict(peak_reserved_bytes=30*s.GIB, reserved_bytes=s.GIB))]
        self.assertFalse(s.fits(task('new','dice_bands'), active, {}, sample()))


    def test_completed_work_meter_retains_final_updates_and_excludes_old_resume_charges(self):
        meter=s.WorkMeter();meter.last_tick=0
        first=task('resumed');second=task('replacement',seed=1)
        meter.start(first,1000)
        meter.record(first,dict(completed_training_flops=1050))
        self.assertEqual(meter.observe(5,['nnunet/dice']),10)
        # The worker's final counter is captured before it leaves the active list.
        meter.record(first,dict(completed_training_flops=1100))
        meter.start(second)
        meter.record(second,dict(completed_training_flops=25))
        self.assertEqual(meter.observe(10,['nnunet/dice']),12.5)
        self.assertEqual(len(meter.window),2)
        meter.record(first,dict(completed_training_flops=1100))
        meter.record(second,dict(completed_training_flops=20)) # stale snapshot
        self.assertEqual(meter.observe(15,['nnunet/dice']),125/15)







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

    def test_unknown_memory_colocates_but_future_growth_is_reserved(self):
        item = task('a')
        self.assertTrue(s.fits(item, [], {}, sample()))
        active=[dict(task=task('b'+str(i)),status={}) for i in range(4)]
        self.assertTrue(s.fits(item,active,{},sample()))
        active.append(dict(task=task('fifth'),status={}))
        self.assertFalse(s.fits(item,active,{},sample()))

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
        self.assertEqual(s.candidates([baseline, constraint, resumed], {'interrupted'}, []), [resumed,baseline])
        self.assertEqual(s.candidates([baseline, constraint], set(), []), [baseline])
        self.assertEqual(s.candidates([constraint], set(), []), [])
        self.assertEqual(s.candidates([constraint], set(), [baseline]), [constraint])

    def test_exact_trainer_arguments_keep_science_and_resume(self):
        root = Path('/content/hippo_adni_bundle'); item = task('case')
        self.assertEqual(s.trainer_arguments(root, item, dict(trainer_alias='nnunet')), [
            '--model','nnunet','--objective','dice','--fold','0','--seed','0',
            '--pkl',str(root/'data/adni_hippocampus_full.pkl'),'--output',str(root/'work/case'),
            '--protocol-dir',str(root/'protocols/nnunet_f0_s0'),'--resume','--threads','2'])

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
p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--status',type=Path);p.add_argument('--control-sha');p.add_argument('--acceleration-receipt');p.add_argument('rest',nargs=argparse.REMAINDER)
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
            report=self.report, result_row=lambda durable,t:dict(t), completed=frozen.completed,
            archive_completed=self.archive, restore_recovery=self.restore,
            restore_matched_baseline=lambda *args: None,
            child_environment=lambda root:dict(os.environ), check_drive_headroom=self.quota)
        frozen.write(self.durable/'preflight.json',dict(status='passed',bundle_sha256='fixed'))
        frozen.write(self.durable/'environment_binding.json',dict(bundle_sha256='fixed'))
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
        self.archive_fixture(item)

    def archive_fixture(self, item):
        out = self.durable/'runs'/item['id']; out.mkdir(parents=True, exist_ok=True)
        manifest = dict(task=item, metadata={p.name:dict(sha256=frozen.digest(p),bytes=p.stat().st_size)
            for p in out.iterdir() if p.name not in ('archive_manifest.json','cleanup.json','archive_acknowledgment.json')})
        frozen.write(out/'archive_manifest.json',manifest)
        frozen.write(out/'archive_acknowledgment.json',dict(status='verified',source_verified=True,
            source_manifest_sha256='fixed',manifest_sha256=frozen.digest(out/'archive_manifest.json')))
        frozen.write(out/'cleanup.json',dict(status='complete'))

    def restore(self, output, recovery):
        self.restored.append(output.name)
        output.mkdir(exist_ok=True)

    def quota(self, durable):
        frozen.write(durable/'storage_quota.json',dict(limit=200*s.GIB,usage=0))

    def runner(self, max_runs=504):
        sweep = s.Sweep(self.root,self.durable,self.control,3,max_runs,1)
        original_load = sweep.load
        def load():
            for item in self.done.values():
                self.archive_fixture(item)
            original_load()
            sweep.profiles['nnunet/dice'] = dict(peak_reserved_bytes=s.GIB,peak_rss_bytes=s.GIB)
        sweep.load = load
        return sweep

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
        marker = self.durable/'old_result.json'; marker.write_bytes(b'old result must remain exact')
        before = marker.read_bytes()
        highest = self.execute(self.runner())
        self.assertEqual(highest,3)
        self.assertEqual(len(self.archive_calls),4)
        self.assertEqual(len(set(self.archive_calls)),4)
        self.assertNotIn('already_complete',self.restored)
        self.assertEqual(marker.read_bytes(),before)

    def test_acceleration_gate_failure_stops_before_any_launch_or_rebinding(self):
        manifest=frozen.read(self.control/'control_manifest.json')
        manifest['acceleration']=dict(candidate_sha256='tested-candidate')
        frozen.write(self.control/'control_manifest.json',manifest)
        before={p.name:p.read_bytes() for p in self.durable.iterdir() if p.is_file()}
        sweep=self.runner()
        with patch.object(s.acceleration_gate,'existing',side_effect=RuntimeError('environment mismatch')):
            with self.assertRaisesRegex(RuntimeError,'environment mismatch'):self.execute(sweep)
        self.assertEqual(sweep.launched,0)
        self.assertEqual(self.archive_calls,[])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.durable.iterdir() if p.is_file()})
        self.assertFalse(sweep.lease.exists())

    def test_validated_receipt_reaches_worker_without_environment_rebinding(self):
        manifest=frozen.read(self.control/'control_manifest.json')
        manifest['acceleration']=dict(candidate_sha256='tested-candidate')
        frozen.write(self.control/'control_manifest.json',manifest)
        receipt=dict(status='verified_external_acceleration',candidate_sha256='tested-candidate')
        sweep=self.runner(max_runs=1)
        with patch.object(s.acceleration_gate,'existing',return_value=(dict(bundle_sha256='fixed'),receipt)), \
             patch.object(self.runtime,'binding',side_effect=AssertionError('must not rebind')):
            self.execute(sweep)
        self.assertEqual(sweep.launched,1)
        self.assertEqual(frozen.read(sweep.local_audit/'acceleration_receipt.json'),receipt)
        self.assertEqual(frozen.read(sweep.audit_dir/'acceleration_receipt.json'),receipt)

    def test_old_memory_profiles_keep_provisional_floor_until_accelerated_full_run(self):
        sweep=self.runner()
        sweep.acceleration_spec=dict(candidate_sha256='tested-candidate')
        item=self.tasks[0]
        sweep.learn(item,dict(peak_reserved_bytes=s.GIB,peak_rss_bytes=s.GIB))
        old=sweep.profiles[s.profile_key(item)]
        self.assertEqual(old['peak_reserved_bytes'],2*s.GIB)
        self.assertFalse(old['acceleration_observed'])
        self.assertGreaterEqual(s.task_reservation(item,sweep.profiles)['gpu'],12*s.GIB)
        sweep.learn(item,dict(peak_reserved_bytes=3*s.GIB,peak_rss_bytes=s.GIB,
            acceleration=dict(candidate_sha256='tested-candidate')))
        self.assertTrue(sweep.profiles[s.profile_key(item)]['acceleration_observed'])
        self.assertEqual(s.task_reservation(item,sweep.profiles)['gpu'],8*s.GIB)

    def test_auto_max_fills_unmeasured_models_immediately_and_refills(self):
        self.tasks = [task('new'+str(i),model=('nnunet','swinunetr','unetr_vit_b16','basic_unet16')[i%4],seed=i)
                      for i in range(8)]
        for item in self.tasks:
            frozen.write(self.root/'protocols'/item['protocol']/'protocol.json',dict(trainer_alias='nnunet'))
        sweep = s.Sweep(self.root,self.durable,self.control,0,504,1)
        started = []
        original_launch = sweep.launch
        def launch(item):
            self.assertNotIn(item['id'], started)
            result = original_launch(item)
            if result: started.append(item['id'])
            return result
        sweep.launch = launch
        peak = self.execute(sweep)
        self.assertEqual(sweep.limit,8)
        self.assertGreaterEqual(peak,5)  # all unknown, conservative 12 GiB per run
        self.assertEqual(len(self.archive_calls),8)
        self.assertEqual(len(started),8)
        self.assertEqual(sweep.live.state['target'],8)

    def test_resume_is_first_but_other_slots_fill_before_it_finishes(self):
        output=self.root/'work/c';output.mkdir()
        self.tasks[2]['arm']='dice'  # initialization-only directory remains resumable
        sweep=self.runner()
        first=[]
        original_launch=sweep.launch
        def launch(item):
            first.append(item['id'])
            if len(first)==2:
                self.assertTrue(any(j['task']['id']=='c' for j in sweep.active))
                self.assertNotIn('c',self.archive_calls)
            return original_launch(item)
        sweep.launch=launch
        self.execute(sweep)
        self.assertEqual(first[0],'c')
        self.assertEqual(len(first),4)

    def test_unfilled_memory_gate_keeps_maximum_target(self):
        sweep=self.runner()
        original=s.fits
        def fits(item, active, profiles, info):
            return len(active)<2 and original(item,active,profiles,info)
        with patch.object(s,'fits',side_effect=fits):
            peak=self.execute(sweep)
        self.assertEqual(peak,2)
        self.assertEqual(sweep.limit,3)
        self.assertEqual(sweep.live.state['target'],3)

    def test_report_failure_preserves_previous_inventory_and_archive_bytes(self):
        marker=self.durable/'comparison.json';marker.write_bytes(b'previous full verified inventory')
        self.archive_fixture(self.tasks[0])
        self.archive_fixture(self.tasks[1])
        (self.durable/'runs'/self.tasks[1]['id']/'archive_acknowledgment.json').write_text('{}')
        def report(root,durable,rows=None):
            marker.write_text(json.dumps(rows));return rows,[]
        self.runtime.report=report
        with self.assertRaises((KeyError,ValueError)):
            self.execute(self.runner(max_runs=0))
        self.assertEqual(marker.read_bytes(),b'previous full verified inventory')
        self.assertFalse(list((self.root/'work').glob('*/started.json')))

    def test_storage_headroom_gates_pending_launch(self):
        sweep=self.runner();sweep.load()
        self.runtime.check_drive_headroom=lambda durable:frozen.write(durable/'storage_quota.json',dict(limit=3*s.GIB,usage=0))
        try:
            with self.assertRaisesRegex(RuntimeError,'headroom'):
                sweep.storage_ready(self.tasks[0])
        finally:
            sweep.live.close()

    def test_max_runs_is_global_across_workers(self):
        self.execute(self.runner(max_runs=2))
        self.assertEqual(len(self.archive_calls),2)

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

    def test_existing_resume_finishes_before_fresh_launches(self):
        output = self.root/'work/c'; output.mkdir()
        frozen.write(output/'budget_ledger.json',dict(charged_attempts=1))
        import torch
        torch.save(dict(step=1,charged_attempts=1),output/'checkpoint_latest.pt')
        self.execute(self.runner())
        self.assertIn('c', self.archive_calls)
        self.assertNotIn('c',self.restored)

    def test_fresh_vm_restores_durable_charge_and_checkpoint_before_any_fresh_run(self):
        import torch
        previous=Path(self.tmp.name)/'previous_vm';previous.mkdir()
        recovery=self.durable/'recovery/c'
        ledger=dict(charged_attempts=1,calibration_attempts=0,training_flops=100)
        frozen.write(previous/'budget_ledger.json',ledger)
        frozen.write(recovery/'charged_ledger.json',ledger)
        torch.save(dict(step=1,charged_attempts=1),previous/'checkpoint_latest.pt')
        expected=(previous/'checkpoint_latest.pt').read_bytes()
        writer=frozen.RecoveryWriter(previous,recovery);writer.close(force=True)
        self.runtime.restore_recovery=frozen.restore_recovery
        original_popen=subprocess.Popen
        launches=[]
        def check(command,**kwargs):
            output=Path(command[command.index('--output')+1]);launches.append(output.name)
            if output.name=='c':
                self.assertEqual((output/'checkpoint_latest.pt').read_bytes(),expected)
                self.assertEqual(frozen.read(output/'budget_ledger.json'),ledger)
                self.assertEqual(frozen.read(recovery/'charged_ledger.json'),ledger)
            return original_popen(command,**kwargs)
        with patch.object(s.subprocess,'Popen',side_effect=check):
            self.execute(self.runner())
        self.assertEqual(launches[0],'c')
        self.assertEqual(launches.count('c'),1)
        self.assertIn('c', self.archive_calls)

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

    def test_duplicate_task_launch_is_rejected_before_popen(self):
        sweep=self.runner(); sweep.load()
        try:
            sweep.started_ids.add('a')
            with patch.object(s.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'Duplicate task'):
                    sweep.launch(self.tasks[0])
                launch.assert_not_called()
        finally:
            sweep.live.close()

    def test_existing_drive_owner_blocks_without_expiring_lock(self):
        lock=self.durable/'.adni_controller_lock'; lock.mkdir()
        frozen.write(lock/'owner.json',dict(session='other-runtime',started_at=0))
        before=(lock/'owner.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError,'controller lock already exists'):
            self.execute(self.runner())
        self.assertEqual((lock/'owner.json').read_bytes(),before)
        self.assertEqual(self.restored,[])

    def test_no_negative_probe_for_hundreds_of_absent_run_directories(self):
        self.tasks=[task('absent'+str(i),seed=i) for i in range(504)]
        sweep=self.runner(max_runs=0)
        calls=[]
        original=s.directories
        def listing(path):
            calls.append(path)
            return original(path)
        with patch.object(s, 'directories', side_effect=listing), patch.object(s, 'has_work', side_effect=AssertionError('per-task Drive probe')):
            self.execute(sweep)
        self.assertEqual(calls.count(self.durable/'runs'),1)
        self.assertEqual(calls.count(self.durable/'recovery'),1)
        self.assertEqual(sweep.launched,0)

    def test_verified_compatible_old_memory_profile_is_reused_but_unrelated_one_is_not(self):
        manifest=frozen.read(self.control/'control_manifest.json')
        manifest['compatible_profile_control_sha256']=['verified-previous-observer']
        frozen.write(self.control/'control_manifest.json',manifest)
        old=task('complete_profile');self.tasks.insert(0,old);self.done[old['id']]=dict(old)
        out=self.durable/'runs'/old['id']
        info=dict(completed=True,observed_full_run=True,peak_reserved_bytes=s.GIB,peak_rss_bytes=2*s.GIB)
        frozen.write(out/'concurrency_old.json',dict(info,control_sha256='verified-previous-observer'))
        frozen.write(out/'concurrency_unrelated.json',dict(info,control_sha256='unknown-observer',peak_reserved_bytes=70*s.GIB))
        self.archive_fixture(old)
        sweep=s.Sweep(self.root,self.durable,self.control,0,0,1)
        self.execute(sweep)
        self.assertEqual(sweep.profiles['nnunet/dice']['peak_reserved_bytes'],s.GIB)
        self.assertEqual(sweep.launched,0)

    def test_corrupt_existing_archive_blocks_new_training_and_preserves_bytes(self):
        item=self.tasks[0]
        out=self.durable/'runs'/item['id'];out.mkdir(parents=True)
        frozen.write(out/'result.json',dict(score=.8))
        self.archive_fixture(item)
        (out/'result.json').write_text('{"score":0.9}')
        before={p.name:p.read_bytes() for p in out.iterdir()}
        sweep=self.runner()
        with self.assertRaisesRegex(ValueError,'Archive differs'):
            self.execute(sweep)
        self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
        self.assertEqual(self.restored,[])

    def test_completed_proof_checks_readability_identity_cleanup_and_ack(self):
        item=self.tasks[0]
        out=self.durable/'runs'/item['id'];out.mkdir(parents=True)
        local=self.root/'cache'; frozen.write(local/'environment_binding.json',dict(bundle_sha256='fixed'))
        live=LiveStatus(self.root/'live')
        for kind in ('readability','identity','ack','cleanup'):
            with self.subTest(kind=kind):
                frozen.write(out/'result.json',dict(score=.8))
                if kind=='readability': (out/'result.json').write_text('invalid JSON')
                self.archive_fixture(item)
                if kind=='identity':
                    manifest=frozen.read(out/'archive_manifest.json');manifest['task']['seed']=99
                    frozen.write(out/'archive_manifest.json',manifest)
                    ack=frozen.read(out/'archive_acknowledgment.json');ack['manifest_sha256']=frozen.digest(out/'archive_manifest.json')
                    frozen.write(out/'archive_acknowledgment.json',ack)
                if kind=='ack':
                    ack=frozen.read(out/'archive_acknowledgment.json');ack['source_verified']=False
                    frozen.write(out/'archive_acknowledgment.json',ack)
                if kind=='cleanup':
                    frozen.write(out/'cleanup.json',dict(status='in_progress'))
                    with verified_snapshot(frozen,self.durable,local,item,dict(bundle_sha256='fixed')) as result:
                        self.assertIsNone(result)
                else:
                    with self.assertRaises(ValueError):
                        with verified_snapshot(frozen,self.durable,local,item,dict(bundle_sha256='fixed')):
                            pass

    def test_stop_request_during_slow_recovery_cannot_launch_after_it_returns(self):
        sweep=self.runner(); sweep.load()
        def recover(output,recovery):
            sweep.request_stop('stop during Drive read')
        self.runtime.restore_recovery=recover
        with patch.object(s.subprocess,'Popen') as launch:
            sweep.launch(self.tasks[0])
            launch.assert_not_called()
        self.assertIn(self.tasks[0],sweep.pending)
        self.assertEqual(sweep.launched,0)

    def matched_fixture(self, wrong_seed=False, resumed=False):
        baseline=task('ADNI_nnunet_dice_f0s1',seed=1)
        constrained=task('ADNI_nnunet_dice_bands_f0s1','dice_bands',seed=1)
        self.tasks=[baseline,constrained]
        self.done[baseline['id']]=dict(baseline)
        frozen.write(self.root/'protocols'/constrained['protocol']/'protocol.json',
                     dict(trainer_alias='nnunet',baseline_reference=dict(initialization_sha256='seed1')))
        source=self.durable/'runs'/baseline['id']/'initialization.json'
        frozen.write(source,dict(model_state_sha256='seed0' if wrong_seed else 'seed1'))
        if resumed:
            import torch
            output=self.root/'work'/constrained['id'];output.mkdir()
            frozen.write(output/'budget_ledger.json',dict(charged_attempts=1))
            torch.save(dict(step=1,charged_attempts=1),output/'checkpoint_latest.pt')
        def restore_matched(*args):
            with patch.object(frozen,'completed',side_effect=lambda durable,t:t['id'] in self.done):
                frozen.restore_matched_baseline(*args)
        self.runtime.restore_matched_baseline=restore_matched
        return baseline,constrained,source

    def test_fresh_vm_restores_verified_baseline_before_launch(self):
        baseline,constrained,source=self.matched_fixture()
        before=source.read_bytes()
        original_popen=subprocess.Popen
        def checked_popen(*args,**kwargs):
            self.assertEqual((self.root/'work'/baseline['id']/'initialization.json').read_bytes(),before)
            return original_popen(*args,**kwargs)
        with patch.object(s.subprocess,'Popen',side_effect=checked_popen):
            self.execute(self.runner())
        self.assertEqual(self.archive_calls,[constrained['id']])
        self.assertEqual(source.read_bytes(),before)

    def test_resumed_constraint_also_restores_its_matched_baseline(self):
        baseline,constrained,source=self.matched_fixture(resumed=True)
        self.execute(self.runner())
        self.assertEqual((self.root/'work'/baseline['id']/'initialization.json').read_bytes(),source.read_bytes())
        self.assertEqual(self.archive_calls,[constrained['id']])
        self.assertNotIn(constrained['id'],self.restored)

    def test_wrong_seed_baseline_blocks_before_any_training(self):
        _,_,source=self.matched_fixture(wrong_seed=True)
        before=source.read_bytes()
        with patch.object(s.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'initialization differs'):
                self.execute(self.runner())
            launch.assert_not_called()
        self.assertEqual(source.read_bytes(),before)
        self.assertEqual(self.archive_calls,[])


class HeartbeatTests(unittest.TestCase):
    def test_hung_resource_probe_cannot_delay_heartbeat(self):
        release=threading.Event()
        def slow(): release.wait(2)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            live=LiveStatus(Path(tmp),provider=slow,interval=.03,poll=.01)
            live.start();time.sleep(.10)
            try:
                self.assertGreaterEqual(out.getvalue().count('HEARTBEAT'),2)
                self.assertTrue((Path(tmp)/'status.json').exists())
            finally:
                release.set();live.close()

    def test_local_heartbeat_survives_blocked_drive_and_distinguishes_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            live=LiveStatus(Path(tmp), interval=.04, poll=.01)
            entered, release = threading.Event(), threading.Event()
            def blocked_drive():
                with live.phase('waiting on Drive', 'run_A'):
                    entered.set(); release.wait(2)
                    live.update(progress=True, checked=1, verified=1)
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                live.start(); thread=threading.Thread(target=blocked_drive);thread.start()
                self.assertTrue(entered.wait(1))
                time.sleep(.13)
                status=json.loads((Path(tmp)/'status.json').read_text())
                self.assertEqual(status['phase'],'waiting on Drive')
                self.assertEqual(status['checked'],0)
                self.assertIn('STILL WAITING',out.getvalue())
                release.set();thread.join(1);time.sleep(.06);live.close()
            self.assertIn('progress since last heartbeat',out.getvalue())
            self.assertEqual(json.loads((Path(tmp)/'status.json').read_text())['verified'],1)

    def test_failed_gpu_observation_does_not_kill_local_heartbeat(self):
        def failed(): raise RuntimeError('nvidia-smi timeout')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            live=LiveStatus(Path(tmp),provider=failed,interval=.03,poll=.01)
            live.start();time.sleep(.08);live.close()
            self.assertIn('HEARTBEAT',out.getvalue())
            self.assertIn('nvidia-smi timeout',json.loads((Path(tmp)/'status.json').read_text())['observation_error'])


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
        self.assertFalse((self.mount/'hippo_adni_science').exists())
        store=self.choose()
        self.assertEqual(store,self.mount/'hippo_adni_science/work')
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
        self.assertNotIn('rmtree',text)
        self.assertNotIn('MSD',text)
        self.assertNotIn('msd',text)
        self.assertIn('5 training subjects/10 crops and 27 validation subjects/54 crops',text)
        self.assertIn('binary ADNI loss graphs',text)
        self.assertIn('A/P swaps do not apply',text)
        self.assertNotIn('cached GPU inputs',text)

    def test_control_package_preserves_scientific_identity(self):
        old = frozen.read(ORIGINAL/'export_summary.json')
        new = frozen.read(EXPORT/'export_summary.json')
        self.assertEqual(old['bundle_manifest_sha256'],new['scientific_bundle_sha256'])
        self.assertEqual(new['extra_scientific_runs'],0)
        self.assertEqual(new['total_runs'],504)
        self.assertEqual(new['control_zip_sha256'],s.sha(EXPORT/'adni_validated_acceleration_control_v5.zip'))
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(EXPORT/'adni_validated_acceleration_control_v5.zip') as archive:
                archive.extractall(tmp)
            _, manifest=s.verify_control(Path(tmp))
            self.assertEqual(manifest['scientific_bundle_sha256'],old['bundle_manifest_sha256'])
        for name, expected in frozen.read(ORIGINAL/'bundle/bundle_manifest.json').items():
            self.assertEqual(s.sha(ORIGINAL/'bundle'/name),expected,name)


class WorkerTests(unittest.TestCase):
    def test_phase_and_update_observers_preserve_optimizer_data_and_rng(self):
        import torch
        for optimizer_type in (torch.optim.SGD,torch.optim.AdamW):
            signatures=[]
            for instrumented in (False,True):
                torch.manual_seed(73)
                trace=[];stats=dict(completed_updates=0,phase='starting')
                class Dataset:
                    def __init__(self, data, names): self.items=[data[i] for i in names]
                    def __getitem__(self,index): return self.items[index]
                def evaluate(model,data): return model(data[0]).detach().tolist()
                def atomic_json(path,value): trace.append((path,dict(value)))
                base=SimpleNamespace(PreparedPatients=Dataset, evaluate=evaluate,
                    _load_pkl_dataframe=lambda:[torch.randn(2,3)],
                    build_model=lambda:torch.nn.Linear(3,2), atomic_json=atomic_json,
                    build_loss_and_optimizer=lambda model:(None,optimizer_type(model.parameters(),lr=.01)))
                trainer=SimpleNamespace(calibrate=lambda:{'weight':.1},training_loss=lambda pred:pred.square().sum())
                def train():
                    model=base.build_model();_,optimizer=base.build_loss_and_optimizer(model)
                    data=base.PreparedPatients(base._load_pkl_dataframe(),[0])
                    base.atomic_json('config.json',dict(max_steps=3))
                    trainer.calibrate()
                    losses=[]
                    for i in range(3):
                        stats['ledger']=dict(charged_attempts=i+1,training_flops=(i+1)*100)
                        optimizer.zero_grad();loss=trainer.training_loss(model(data[0]))
                        loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    return dict(losses=losses, weights=[p.detach().tolist() for p in model.parameters()],
                        evaluation=base.evaluate(model,data),rng=torch.get_rng_state().tolist(),trace=trace,
                        optimizer=[{k:v.tolist() if hasattr(v,'tolist') else v for k,v in item.items()} for item in optimizer.state.values()])
                trainer.main=train
                entry=SimpleNamespace(scientific=SimpleNamespace(experiment=SimpleNamespace(base=base,trainer=trainer)))
                if instrumented:instrument_trainer(entry,stats)
                signatures.append(trainer.main())
                if instrumented:
                    self.assertEqual(stats['completed_updates'],3)
                    self.assertEqual(stats['completed_training_flops'],300)
                    self.assertEqual(stats['max_updates'],3)
                    for key in ('data_loading_seconds','batch_loading_seconds','evaluation_seconds','calibration_seconds'):
                        self.assertGreater(stats[key],0)
            self.assertEqual(signatures[0],signatures[1])

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
    def __init__(self,output,durable):self.output=output;self.durable=durable
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
                out=root/mode;out.mkdir();durable=root/(mode+'_drive');durable.mkdir()
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
