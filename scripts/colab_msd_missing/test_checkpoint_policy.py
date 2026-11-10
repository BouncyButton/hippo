"""Exercise the real asynchronous checkpoint writer without a GPU or Drive writes."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/colab_msd_missing'))
sys.path.insert(0,str(ROOT/'scripts/colab_msd_cloud'))
import hooks
import store
from traffic import DEFAULT_LIMIT, RECOVERY_RESERVE, RoutineBudgetExceeded, reserved_window
from scripts.colab_msd import runtime

GIB=1024**3


class BudgetTests(unittest.TestCase):
    def test_actual_exhausted_journal_can_recover_without_losing_any_reservation(self):
        path=ROOT/'reports/msd_upload_budget_fix_20261019/traffic_before.json'
        state=json.loads(path.read_text())
        now=state['reservations'][-1]['time']+1
        # Exact prior failure: the old 200 GiB budget blocks a UNETR checkpoint.
        with self.assertRaises(store.DriveBlocked):
            reserved_window(state,2*GIB,'unfinished',now,limit=200*GIB)
        migrated=reserved_window(state,2*GIB,'unfinished',now,priority='recovery')
        self.assertEqual(migrated['reservations'][:-1],state['reservations'])
        self.assertEqual(migrated['reserved_bytes'],state['reserved_bytes']+2*GIB)
        self.assertEqual(migrated['policy_history'][-1]['previous_limit_bytes'],200*GIB)

    def test_routine_cannot_spend_reserve_but_five_workers_can_save(self):
        state=reserved_window({},DEFAULT_LIMIT-RECOVERY_RESERVE,'prior',100000.)
        with self.assertRaises(RoutineBudgetExceeded):
            reserved_window(state,1,'routine',100001.,priority='routine')
        for i in range(5):
            state=reserved_window(state,2*GIB,'worker'+str(i),100001.+i,priority='recovery')
        self.assertEqual(state['reserved_bytes'],DEFAULT_LIMIT-RECOVERY_RESERVE+10*GIB)

    def test_hard_limit_still_blocks_recovery(self):
        state=reserved_window({},DEFAULT_LIMIT,'full',100000.)
        with self.assertRaises(store.DriveBlocked):
            reserved_window(state,1,'extra',100001.,priority='recovery')

    def test_policy_migration_record_is_not_repeated(self):
        one=reserved_window({},10,'a',100000.)
        two=reserved_window(one,10,'b',100001.)
        self.assertEqual(two['policy_history'],one['policy_history'])


class CheckpointPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'bundle/work/task'
        self.output.mkdir(parents=True);self.durable=self.root/'local/02/recovery/task'
        self.cfg=dict(folder_id='server',local_root=str(self.root/'local'),checkpoint_interval_seconds=900)
        self.cloud=Mock();self.cloud.folder='server';self.sent=[];self.meta={}
        self.cloud.put_immutable.side_effect=self.upload
        self.cloud.metadata.side_effect=lambda id:self.meta[id]
        self.conf=patch('hooks.configured',return_value=(self.cfg,self.cloud))
        self.conf.start();self.addCleanup(self.conf.stop)
        env=patch.dict('os.environ',HIPPO_CLOUD_PHASE='02',HIPPO_CLOUD_TASK='{"id":"task"}')
        env.start();self.addCleanup(env.stop)
        self.entry=SimpleNamespace(write=runtime.write,read=runtime.read,RecoveryWriter=runtime.RecoveryWriter)
        hooks.install_entry(self.entry,self.output,self.root,{})
        self.save(7)

    def save(self,step):
        # Replace the inode exactly as the scientific trainer does.
        tmp=self.output/'next.tmp';tmp.write_bytes(b'model optimizer RNG exact state '+str(step).encode())
        tmp.replace(self.output/'checkpoint_latest.pt')
        runtime.write(self.output/'budget_ledger.json',dict(charged_attempts=step,training_flops=step*100))
        runtime.write(self.output/'config.json',dict(augmentation='none',seed=2,fold=2))

    def upload(self,path,kind,task,phase,**kwargs):
        payload=Path(path).read_bytes()
        with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
            files={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
        id='saved'+str(len(self.sent))
        result=dict(id=id,parents=['server'],size=str(len(payload)),md5Checksum=hashlib.md5(payload).hexdigest(),local_sha256=store.digest(path))
        self.sent.append(dict(kind=kind,task=task,phase=phase,options=kwargs,files=files))
        self.meta[id]=result
        return result

    def writer(self):
        writer=self.entry.RecoveryWriter(self.output,self.durable)
        self.addCleanup(writer.pool.shutdown,wait=True)
        # The generated missing-only hook applies this first-upload delay.
        writer.last=time.monotonic()
        return writer

    def test_15_minute_interval_and_local_checkpoint_bytes_unchanged(self):
        writer=self.writer();before=(self.output/'checkpoint_latest.pt').read_bytes()
        writer.last=time.monotonic()-899;writer.capture()
        self.assertIsNone(writer.future);self.assertEqual(writer.interval,900)
        writer.last=time.monotonic()-901;writer.capture();writer.future.result()
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.sent[0]['files']['checkpoint_latest.pt'],before)
        self.assertEqual((self.output/'checkpoint_latest.pt').read_bytes(),before)
        writer.close(force=False)

    def test_clean_stop_forces_verified_checkpoint_before_interval(self):
        writer=self.writer();writer.close(force=True)
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.sent[0]['options'],{'priority':'recovery'})
        pointer=runtime.read(self.durable/'cloud_latest.json')
        self.assertEqual(pointer['charged_attempts'],7)
        self.assertTrue((self.output/'checkpoint_latest.pt').exists())

    def test_routine_budget_stop_does_not_poison_forced_recovery(self):
        writer=self.writer()
        def budget(path,kind,task,phase,**kwargs):
            if kwargs.get('priority')!='recovery':raise RoutineBudgetExceeded('ordinary allowance full')
            return self.upload(path,kind,task,phase,**kwargs)
        self.cloud.put_immutable.side_effect=budget
        writer.last=time.monotonic()-901;writer.capture();writer.future.result()
        self.assertTrue((self.root/'local/CHECKPOINT_BUDGET_STOP.json').exists())
        self.assertFalse((self.root/'local/SAVE_BLOCKED.json').exists())
        self.save(8);writer.close(force=True)
        self.assertEqual(runtime.read(self.durable/'cloud_latest.json')['charged_attempts'],8)
        self.assertEqual(self.sent[0]['files']['checkpoint_latest.pt'],(self.output/'checkpoint_latest.pt').read_bytes())

    def test_stop_marker_blocks_new_launch_and_drains_existing_workers(self):
        sweep=SimpleNamespace(launch=Mock(),check_stop=Mock(),request_stop=Mock(),r=SimpleNamespace())
        original=sweep.launch;hooks.install_sweep(sweep,'02')
        store.atomic(self.root/'local/CHECKPOINT_BUDGET_STOP.json',{'reason':'budget'})
        sweep.launch({'id':'another'});sweep.check_stop()
        original.assert_not_called();self.assertEqual(sweep.request_stop.call_count,2)

    def test_unchanged_setup_checkpoint_reuses_verified_server_snapshot(self):
        writer=self.writer();writer.close(force=True)
        again=self.writer();again.close(force=True)
        self.assertEqual(len(self.sent),1)
        self.cloud.metadata.assert_called_once()

    def test_changed_remote_checkpoint_does_not_skip_save_or_delete_local(self):
        writer=self.writer();writer.close(force=True)
        self.meta['saved0']=dict(self.meta['saved0'],md5Checksum='changed')
        again=self.writer()
        with self.assertRaisesRegex(RuntimeError,'checkpoint changed'):again.close(force=True)
        self.assertTrue((self.output/'checkpoint_latest.pt').exists())
        self.assertTrue((self.root/'local/SAVE_BLOCKED.json').exists())

    def test_drive_failure_retains_local_snapshot_and_checkpoint(self):
        writer=self.writer();self.cloud.put_immutable.side_effect=store.DriveBlocked('network')
        with self.assertRaises(store.DriveBlocked):writer.close(force=True)
        self.assertTrue((self.output/'checkpoint_latest.pt').exists())
        self.assertTrue(list(self.output.parent.glob('recovery-*.tar.gz')))
        self.assertFalse((self.durable/'cloud_latest.json').exists())

    def test_forced_priority_does_not_change_inflight_routine_upload(self):
        writer=self.writer();started=threading.Event();release=threading.Event()
        def slow(path,kind,task,phase,**kwargs):
            if not kwargs:
                started.set();self.assertTrue(release.wait(5))
            return self.upload(path,kind,task,phase,**kwargs)
        self.cloud.put_immutable.side_effect=slow
        writer.last=time.monotonic()-901;writer.capture();self.assertTrue(started.wait(5))
        self.save(8)
        errors=[]
        def stop():
            try:writer.close(force=True)
            except Exception as e:errors.append(e)
        thread=threading.Thread(target=stop);thread.start();release.set();thread.join(5)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual([s['options'] for s in self.sent],[{},dict(priority='recovery')])
        self.assertEqual(runtime.read(self.durable/'cloud_latest.json')['charged_attempts'],8)

    def test_prepare_protects_only_unfinished_work_without_changing_bytes(self):
        package=ROOT/'reports/msd_missing_324_second_account_20261018/package/cloud'
        sys.path.insert(0,str(package));self.addCleanup(sys.path.remove,str(package))
        spec=importlib.util.spec_from_file_location('missing_policy_test',package/'missing.py')
        missing=importlib.util.module_from_spec(spec);spec.loader.exec_module(missing)
        scope={'missing_tasks':[{'id':'task','seed':2},{'id':'completed','seed':0}]}
        before={p.name:p.read_bytes() for p in self.output.iterdir()}
        missing.protect_unfinished(runtime,scope,{'02':self.root/'bundle'},self.root/'local',{'completed'})
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual([x['task'] for x in self.sent],['task'])
        self.assertEqual(self.sent[0]['options'],{'priority':'recovery'})


if __name__=='__main__':unittest.main()
