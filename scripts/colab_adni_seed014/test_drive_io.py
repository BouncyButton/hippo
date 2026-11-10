"""CPU-only transport/crash fixtures. No training, GPU trials or live Drive I/O."""
import argparse
import ast
import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_adni_seed014 import drive_io as d, prepare_drive_io as p


def fresh_runtime():
    spec = importlib.util.spec_from_file_location('transport_test_runtime', p.old.PARENT/'runtime.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.drive, self.output = self.root/'drive', self.root/'work/run'
        self.output.mkdir(parents=True); self.drive.mkdir()
        self.recovery = self.drive/'results/recovery/run'
        self.r = fresh_runtime()
        self.raw_write = self.r.write
        self.io = d.install(self.r, self.drive, self.root/'ssd')
        self.io.spacing = 0
        self.env = patch.dict(os.environ, os.environ.copy()); self.env.start(); self.addCleanup(self.env.stop)
        self.ledger = dict(charged_attempts=3, training_flops=300, calibration_attempts=1,
                           calibration_flops=50, calibration_sha256='same')
        self.raw_write(self.output/'budget_ledger.json', self.ledger)
        self.raw_write(self.recovery/'charged_ledger.json', self.ledger)
        for name in ('config.json','calibration.json','initialization.json','runtime_recipe.json'):
            self.raw_write(self.output/name, dict(name=name, augmentation='none'))
        (self.output/'checkpoint_latest.pt').write_bytes(b'opaque checkpoint/optimizer/RNG bytes\x00'*10000)

    def writer(self):
        return self.r.RecoveryWriter(self.output, self.recovery)

    def change_checkpoint(self):
        temporary = self.output/'new.pt'
        temporary.write_bytes(b'new checkpoint'*10000)
        temporary.replace(self.output/'checkpoint_latest.pt')

    def test_compressed_roundtrip_all_payload_hashes_unchanged_and_ledgers_exact(self):
        expected = {f.name:f.read_bytes() for f in self.output.iterdir()}
        self.writer().close()
        pointer = self.r.read(self.recovery/'latest.json')
        self.assertTrue(pointer['file'].endswith('.tar.gz'))
        self.r.restore_recovery(self.root/'restored', self.recovery)
        for name, data in expected.items():self.assertEqual((self.root/'restored'/name).read_bytes(), data)
        self.assertEqual(self.r.read(self.recovery/'charged_ledger.json'),self.ledger)
        self.assertLess(self.io.stats['recovery_compressed_bytes'],self.io.stats['recovery_input_bytes'])

    def test_old_uncompressed_recovery_still_restores_with_new_transport(self):
        old = fresh_runtime()
        old.RecoveryWriter(self.output, self.recovery).close()
        before = (self.recovery/'latest.json').read_bytes()
        self.r.restore_recovery(self.root/'restored', self.recovery)
        self.assertEqual((self.recovery/'latest.json').read_bytes(), before)
        self.assertEqual((self.root/'restored/checkpoint_latest.pt').read_bytes(), (self.output/'checkpoint_latest.pt').read_bytes())

    def test_forced_close_deduplicates_only_verified_identical_full_snapshot(self):
        writer = self.writer(); self.assertEqual(writer.interval,180)
        writer.capture(); writer.future.result()
        before = {x.name:x.read_bytes() for x in self.recovery.iterdir()}
        copies = self.io.stats['copies']
        writer.close()
        self.assertEqual(self.io.stats['copies'], copies)
        self.assertEqual(self.io.stats['recovery_duplicates'], 1)
        self.assertEqual(before, {x.name:x.read_bytes() for x in self.recovery.iterdir()})

    def test_same_charge_with_changed_calibration_is_not_suppressed(self):
        writer = self.writer(); writer.capture(); writer.future.result()
        old = self.r.read(self.recovery/'latest.json')
        self.raw_write(self.output/'calibration.json', dict(changed_metadata=True))
        writer.close()
        latest = self.r.read(self.recovery/'latest.json')
        self.assertNotEqual(latest['sha256'], old['sha256'])
        self.assertFalse((self.recovery/old['file']).exists())
        self.assertEqual(latest['charged_attempts'],3)

    def test_force_flush_does_not_drop_new_checkpoint(self):
        writer = self.writer(); writer.capture(); writer.future.result()
        self.change_checkpoint()
        writer.close()
        self.r.restore_recovery(self.root/'restored', self.recovery)
        self.assertEqual((self.root/'restored/checkpoint_latest.pt').read_bytes(),(self.output/'checkpoint_latest.pt').read_bytes())
        self.assertEqual(self.io.stats['copies'],2)

    def test_async_snapshot_keeps_original_inode(self):
        writer = self.writer(); expected = (self.output/'checkpoint_latest.pt').read_bytes()
        writer.capture(); self.change_checkpoint(); writer.close(force=False)
        self.r.restore_recovery(self.root/'restored',self.recovery)
        self.assertEqual((self.root/'restored/checkpoint_latest.pt').read_bytes(),expected)

    def test_corruption_blocks_dedup_and_preserves_local_snapshot(self):
        writer=self.writer();writer.capture();writer.future.result()
        pointer=self.r.read(self.recovery/'latest.json')
        (self.recovery/pointer['file']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'hash differs'):writer.close()
        self.assertTrue(Path(self.io.stats['retained_local_recovery']).exists())
        self.assertEqual(self.r.read(self.recovery/'latest.json'),pointer)

    def test_upload_failure_keeps_previous_pointer_checkpoint_and_local_staging(self):
        self.writer().close(); old={x.name:x.read_bytes() for x in self.recovery.iterdir()}
        self.change_checkpoint()
        with patch.object(self.io,'raw_copy',side_effect=OSError('transfer interrupted')):
            with self.assertRaises(OSError):self.writer().close()
        self.assertEqual(old,{x.name:x.read_bytes() for x in self.recovery.iterdir()})
        self.assertTrue(Path(self.io.stats['retained_local_archive']).exists())
        self.assertTrue((self.output/'checkpoint_latest.pt').exists())

    def test_pointer_readback_failure_never_retires_previous_archive(self):
        self.writer().close();old=self.r.read(self.recovery/'latest.json')
        self.change_checkpoint(); original=self.io.raw_read
        count=0
        def read(path):
            nonlocal count
            if Path(path).name=='latest.json':
                count+=1
                if count>=2:return dict(file='incorrect publication')
            return original(path)
        with patch.object(self.io,'raw_read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'readback differs'):self.writer().close()
        self.assertTrue((self.recovery/old['file']).exists())
        self.assertTrue(Path(self.io.stats['retained_local_archive']).exists())

    def test_newer_charge_blocks_restore_and_never_resets_ledger(self):
        self.writer().close()
        newer=dict(self.ledger,charged_attempts=4,training_flops=400)
        self.r.write(self.recovery/'charged_ledger.json',newer)
        with self.assertRaisesRegex(RuntimeError,'budget audit'):self.r.restore_recovery(self.root/'restore',self.recovery)
        self.assertEqual(self.r.read(self.recovery/'charged_ledger.json'),newer)

    def test_charge_is_never_coalesced_even_for_identical_values(self):
        with patch.object(self.io,'raw_write',wraps=self.raw_write) as write:
            for _ in range(2):self.r.write(self.recovery/'charged_ledger.json',self.ledger)
            self.assertEqual(write.call_count,2)

    def test_retry_uses_same_payload_bounded_and_publishes_local_stop_without_rng_draws(self):
        calls=[]
        def operation():
            calls.append(self.ledger.copy())
            raise OSError('User rate limit exceeded')
        state=random.getstate()
        with patch.object(self.io,'pace'):
            with self.assertRaisesRegex(RuntimeError,'throttling persisted'):
                self.io.call('charge',operation,mutation=True)
        self.assertEqual(len(calls),5)
        self.assertTrue(all(c==self.ledger for c in calls))
        self.assertEqual(random.getstate(),state)
        self.assertIsNotNone(self.io.quota_event())
        self.assertEqual(os.environ['HIPPO_DEADLINE'],'0')
        self.assertEqual(self.r.raw_read(self.recovery/'charged_ledger.json') if hasattr(self.r,'raw_read') else self.io.raw_read(self.recovery/'charged_ledger.json'),self.ledger)

    def test_nonquota_corruption_and_permissions_not_retried(self):
        for exc in (ValueError('hash differs'),PermissionError('403 forbidden'),FileNotFoundError('gone'),OSError('EIO')):
            with patch.object(self.io,'pace'),patch.object(self.io,'raw_copy',side_effect=exc) as copy:
                with self.assertRaises(type(exc)):self.r.copy_verified(self.output/'checkpoint_latest.pt',self.drive/'partial/a')
                self.assertEqual(copy.call_count,1)

    def test_idempotent_copy_recovers_after_rename_without_second_upload(self):
        raw_copy=self.io.raw_copy;calls=[]
        def ambiguous(source,target):
            raw_copy(source,target);calls.append(1)
            raise OSError('User rate limit exceeded')
        with patch.object(self.io,'pace'),patch.object(self.io,'raw_copy',side_effect=ambiguous):
            self.r.copy_verified(self.output/'checkpoint_latest.pt',self.drive/'partial/a')
        self.assertEqual(len(calls),1)
        self.assertEqual(self.io.stats['duplicate_copies'],1)

    def test_bulk_uploads_serialized_but_ledger_lane_not_blocked(self):
        entered=threading.Event();release=threading.Event(); counts=[0,0]
        raw_copy=self.io.raw_copy
        def slow(source,target):
            counts[0]+=1;counts[1]=max(counts)
            entered.set();release.wait(5)
            raw_copy(source,target);counts[0]-=1
        with patch.object(self.io,'raw_copy',side_effect=slow),ThreadPoolExecutor(2) as pool:
            first=pool.submit(self.r.copy_verified,self.output/'checkpoint_latest.pt',self.drive/'bulk/a')
            self.assertTrue(entered.wait(3))
            second=pool.submit(self.r.copy_verified,self.output/'checkpoint_latest.pt',self.drive/'bulk/b')
            self.r.write(self.recovery/'charged_ledger.json',self.ledger)
            self.assertEqual(self.io.raw_read(self.recovery/'charged_ledger.json'),self.ledger)
            release.set();first.result();second.result()
        self.assertEqual(counts[1],1)

    def test_pacing_shared_between_processes_on_local_ssd(self):
        module=Path(d.__file__).resolve()
        code="""import importlib.util,sys,time
from pathlib import Path
s=importlib.util.spec_from_file_location('dio',sys.argv[1]);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
from types import SimpleNamespace
r=SimpleNamespace(write=None,read=None,copy_verified=None,check_drive_headroom=None)
io=d.DriveIO(r,sys.argv[2],sys.argv[3],spacing=.12)
io.pace(True)
print(time.time(),flush=True)
"""
        processes=[subprocess.Popen([sys.executable,'-c',code,str(module),str(self.drive),str(self.root/'ssd')],stdout=subprocess.PIPE,text=True) for _ in range(3)]
        times=[]
        for proc in processes:
            out,_=proc.communicate(timeout=10);self.assertEqual(proc.returncode,0);times.append(float(out))
        times.sort()
        self.assertGreaterEqual(min(b-a for a,b in zip(times,times[1:])),.09)

    def test_local_operations_never_acquire_drive_locks(self):
        with patch.object(self.io,'pace',side_effect=AssertionError('Drive called')):
            self.r.write(self.root/'local.json',dict(local=True))
            self.assertEqual(self.r.read(self.root/'local.json'),dict(local=True))
            self.r.copy_verified(self.root/'local.json',self.root/'copy.json')

    def test_coordinator_sees_quota_during_phase_shutdown_but_not_old_session_events(self):
        started=time.time()
        d.local_json(self.io.local/'quota_event.json',dict(at=started-1,error='old'))
        self.assertIsNone(d.quota_since(self.drive,started,self.root/'ssd'))
        d.local_json(self.io.local/'quota_event.json',dict(at=started+1,error='final report throttled'))
        self.assertEqual(d.quota_since(self.drive,started,self.root/'ssd')['error'],'final report throttled')

    def test_reports_coalesce_but_final_flush_retains_latest_and_charge_unaffected(self):
        publisher=d.ReportPublisher(self.r,self.root/'report.json',self.drive/'comparison.json')
        with patch.object(self.io,'raw_write',wraps=self.raw_write) as write:
            publisher.publish(dict(completed=329));publisher.publish(dict(completed=330));publisher.publish(dict(completed=331))
            self.assertEqual(self.r.read(self.root/'report.json'),dict(completed=331))
            self.assertEqual(self.r.read(self.drive/'comparison.json'),dict(completed=329))
            publisher.flush()
            self.assertEqual(self.r.read(self.drive/'comparison.json'),dict(completed=331))
            writes=[x for x in write.call_args_list if x.args[0]==self.drive/'comparison.json']
            self.assertEqual(len(writes),2)

    def test_frozen_entry_orders_charge_readback_before_update_and_failure_blocks_update(self):
        # Execute the ACTUAL frozen main function with a tiny non-training stub.
        # No optimizer/model is instantiated; only storage ordering is exercised.
        tree=ast.parse((p.old.PARENT/'entry.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        for mode in ('success','write_failure','bad_readback'):
            events=[]
            base=SimpleNamespace(atomic_json=self.raw_write,save_checkpoint=lambda *a:None)
            next_ledger=dict(self.ledger,charged_attempts=4,training_flops=400)
            def scientific_main():
                base.atomic_json(self.output/'budget_ledger.json',next_ledger)
                events.append('optimizer')
            scientific=SimpleNamespace(experiment=SimpleNamespace(base=base),main=scientific_main)
            self.raw_write(self.output/'budget_ledger.json',self.ledger)
            self.raw_write(self.recovery/'charged_ledger.json',self.ledger)
            protocol=self.root/'protocol';protocol.mkdir(exist_ok=True)
            self.raw_write(protocol/'protocol.json',dict(augmentation='none'))
            def write(path,value):
                events.append('charge')
                if mode=='write_failure':raise OSError('write failed')
                self.r.write(path,value)
            def read(path):
                value=self.r.read(path)
                if Path(path).name=='charged_ledger.json' and value==next_ledger:
                    events.append('readback')
                    if mode=='bad_readback':return dict(value,charged_attempts=99)
                return value
            class NoSnapshot:
                def __init__(self,*a):pass
                def close(self,force):events.append(('close',force))
            scope=dict(argparse=argparse,os=os,Path=Path,__file__=str(p.old.PARENT/'entry.py'),
                       install_initialization=lambda *a:None,read=read,write=write,
                       RecoveryWriter=NoSnapshot,scientific=scientific)
            exec(compile(ast.Module(body=[main],type_ignores=[]),'frozen_entry_main','exec'),scope)
            with patch.dict(os.environ,HIPPO_COLAB_RECOVERY=str(self.recovery)),patch.object(sys,'argv',['entry','--output',str(self.output),'--protocol-dir',str(protocol)]):
                if mode=='success':scope['main']();self.assertEqual(events[:3],['charge','readback','optimizer'])
                else:
                    with self.assertRaises(OSError):scope['main']()
                    self.assertNotIn('optimizer',events);self.assertEqual(events[-1],('close',False))
            if mode=='bad_readback':self.assertEqual(self.io.raw_read(self.recovery/'charged_ledger.json'),next_ledger)


class PackageTests(unittest.TestCase):
    def test_actual_packaged_worker_publishes_readable_recovery_and_keeps_startup_stop(self):
        source='''
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from runtime import write,read,RecoveryWriter
def save(path,value):path.write_bytes(value)
scientific=SimpleNamespace(experiment=SimpleNamespace(base=SimpleNamespace(save_checkpoint=save)))
def main():
    out=Path(sys.argv[sys.argv.index('--output')+1])
    recovery=Path(os.environ['HIPPO_COLAB_RECOVERY'])
    writer=RecoveryWriter(out,recovery)
    if os.environ.get('TEST_STARTUP_STOP'):
        save(out/'checkpoint_latest.pt',b'zero-state');writer.close();raise SystemExit(75)
    ledger=dict(charged_attempts=1,training_flops=100,calibration_flops=50)
    write(recovery/'charged_ledger.json',ledger)
    assert read(recovery/'charged_ledger.json')==ledger
    write(out/'budget_ledger.json',ledger)
    save(out/'checkpoint_latest.pt',b'opaque fixture state')
    writer.capture();writer.close()
    write(out/'completion.json',dict(status='complete'))
'''
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=root/'bundle';bundle.mkdir()
            (bundle/'entry.py').write_text(source)
            (bundle/'runtime.py').write_bytes((p.old.PARENT/'runtime.py').read_bytes())
            for stop in (False,True):
                output=root/('stopped' if stop else 'finished');output.mkdir()
                recovery=root/'drive/results/recovery'/output.name
                status=root/(output.name+'.json')
                command=[sys.executable,str(p.OUT/'package/control01/worker.py'),
                    '--bundle',str(bundle),'--status',str(status),'--control-sha','fixture',
                    '--','--output',str(output),'--protocol-dir',str(bundle)]
                env=dict(os.environ,HIPPO_COLAB_RECOVERY=str(recovery),HIPPO_DRIVE_IO_LOCAL=str(root/'ssd'))
                if stop:env['TEST_STARTUP_STOP']='1'
                result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=90)
                self.assertEqual(result.returncode,75 if stop else 0,result.stdout+result.stderr)
                info=json.loads(status.read_text())
                if stop:
                    self.assertTrue(info['uncharged_startup_snapshot_skipped'])
                    self.assertFalse((recovery/'charged_ledger.json').exists())
                else:
                    self.assertTrue(info['completed']);self.assertEqual(info['drive_io']['recovery_duplicates'],1)
                    r=fresh_runtime();r.restore_recovery(root/'restored',recovery)
                    self.assertEqual((root/'restored/checkpoint_latest.pt').read_bytes(),b'opaque fixture state')
                    self.assertEqual(r.read(root/'restored/budget_ledger.json'),r.read(recovery/'charged_ledger.json'))

    def test_packaged_hashes_and_frozen_science_unchanged(self):
        with zipfile.ZipFile(p.OUT/p.ZIP_NAME) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()),len(set(archive.namelist())))
            manifest=json.loads(archive.read('package_manifest.json'))
            for name,sha in manifest.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),sha,name)
                if name.startswith('seed4_bundle/') or name == 'preflight_initfix.py':
                    self.assertEqual(archive.read(name),(p.previous.OUT/'package'/name).read_bytes(),name)
            for control in ('control01','control04'):
                m=json.loads(archive.read(control+'/control_manifest.json'))
                for name,sha in m['files'].items():self.assertEqual(hashlib.sha256(archive.read(control+'/'+name)).hexdigest(),sha)
                for name in ('initialization_compat.py','acceleration_candidate.py','acceleration_gate.py','acceleration_evidence.zip','archive_verification.py','local_storage.py'):
                    self.assertEqual(archive.read(control+'/'+name),(p.previous.OUT/'package'/control/name).read_bytes())
                worker=archive.read(control+'/worker.py').decode()
                self.assertLess(worker.index('checked = verify('),worker.index('drive_io = install_drive_io('))
                self.assertLess(worker.index('drive_io = install_drive_io('),worker.index('entry.main()'))
                scheduler=archive.read(control+'/scheduler.py').decode()
                self.assertIn('self.drive_io.quota_event()',scheduler)
                self.assertIn('self.r.check_drive_headroom(self.durable)',scheduler)
            self.assertFalse(any('/data/' in n or '/work/' in n for n in archive.namelist()))
            coordinator=archive.read('campaign.py').decode()
            self.assertLess(coordinator.index('if transport.quota_since('),coordinator.index('if index == 1:'))
        for name,sha in fresh_runtime().read(p.old.PARENT/'bundle_manifest.json').items():
            self.assertEqual(fresh_runtime().digest(p.old.PARENT/name),sha,name)

    def test_notebook_full_resume_same_results_and_new_zip_no_dataset_upload_or_reset(self):
        r=fresh_runtime();book=r.read(p.OUT/p.BOOK_NAME)
        text='\n'.join(c['source'] for c in book['cells'])
        for cell in book['cells']:
            if cell['cell_type']=='code':ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])
        for value in (p.ZIP_NAME,r.digest(p.OUT/p.ZIP_NAME),"ORIGINAL_RESULTS=DRIVE_ROOT/'results_noaug_v1'", "RESULTS=DRIVE_ROOT/'results_seeds014_noaug_v1'",'preflight_initfix.py',"campaign('run'","campaign('report'"):
            self.assertIn(value,text)
        for forbidden in ('shutil.rmtree','unlink(',"files.upload(",p.previous.ZIP_NAME):self.assertNotIn(forbidden,text)


# Reuse the existing controller's entire failure/resume/uniqueness fixture suite
# against the ACTUAL packaged scheduler, with the new I/O layer installed.
from scripts.colab_adni_parallel import test_scheduler as controller_tests


class PackagedSchedulerTests(controller_tests.IntegrationTests):
    def setUp(self):
        path=p.OUT/'package/control01/scheduler.py'
        spec=importlib.util.spec_from_file_location('packaged_drive_scheduler',path)
        scheduler=importlib.util.module_from_spec(spec)
        with patch.object(sys,'path',[str(path.parent),*sys.path]):
            spec.loader.exec_module(scheduler)
        self.module_patch=patch.object(controller_tests,'s',scheduler)
        self.module_patch.start();self.addCleanup(self.module_patch.stop)
        super().setUp()
        self.runtime.RecoveryWriter=fresh_runtime().RecoveryWriter
        install=d.install
        def fixture_install(runtime,drive_root,local_root=None):
            # The fixture's results folder is named drive directly; production
            # has DriveRoot/results_noaug_v1 and DriveRoot/results_seeds014_noaug_v1.
            io=install(runtime,self.durable,Path(self.tmp.name)/'ssd_coordination')
            io.spacing=0
            return io
        prior_module=sys.modules.get('drive_io')
        sys.modules['drive_io']=d
        def restore_module():
            if prior_module is None:sys.modules.pop('drive_io',None)
            else:sys.modules['drive_io']=prior_module
        self.addCleanup(restore_module)
        hook=patch.object(d,'install',side_effect=fixture_install)
        hook.start();self.addCleanup(hook.stop)

    def test_quota_event_stops_new_launches_and_preserves_existing_results(self):
        sweep=self.runner()
        d.local_json(sweep.drive_io.local/'quota_event.json',dict(at=time.time(),error='rate limit exceeded'))
        self.execute(sweep)
        self.assertEqual(sweep.launched,0)
        self.assertIn('Drive rate limit',sweep.stop_reason)

    def test_background_heartbeat_requests_worker_stop_on_quota(self):
        sweep=self.runner()
        d.local_json(sweep.drive_io.local/'quota_event.json',dict(at=time.time(),error='rate limit exceeded'))
        sweep.telemetry()
        self.assertIn('Drive rate limit',sweep.stop_reason)


if __name__=='__main__':unittest.main()
