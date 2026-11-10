"""Failure-injection tests for durability and recovery; no GPU/network required."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0,str(Path(__file__).parent))
import store
import hooks
from driver import recovery_event, restore_server_results, next_lease, release_lease
from recover import candidates, checked_bytes, derive_completion, json_bytes


class Media:
    def __init__(self,path,**kwargs):self.path=Path(path);self.large=kwargs['resumable']
    def resumable(self):return self.large


class HTTPError(Exception):
    def __init__(self,status=403,reason='userRateLimitExceeded'):
        self.resp=types.SimpleNamespace(status=status)
        self.content=json.dumps({'error':{'errors':[{'reason':reason}]}}).encode()


def api_modules():
    http=types.ModuleType('googleapiclient.http');http.MediaFileUpload=Media
    errors=types.ModuleType('googleapiclient.errors');errors.HttpError=HTTPError
    package=types.ModuleType('googleapiclient');package.http=http;package.errors=errors
    return {'googleapiclient':package,'googleapiclient.http':http,'googleapiclient.errors':errors}


class Files:
    def __init__(self):self.data={};self.calls=[];self.timeout=False;self.bad_checksum=False
    def request(self,fn):return types.SimpleNamespace(execute=lambda **kwargs:fn())
    def generateIds(self,**kwargs):return self.request(lambda:{'ids':['id1']})
    def meta(self,id):
        b=self.data[id]
        return dict(id=id,name='object',size=str(len(b)),md5Checksum='bad' if self.bad_checksum else hashlib.md5(b).hexdigest(),parents=['newstore'])
    def put(self,id,media):
        self.data[id]=media.path.read_bytes()
        if self.timeout:raise TimeoutError('Lost final response')
        return self.meta(id)
    def create(self,body,media_body,**kwargs):
        self.calls.append(('create',body['id']));return self.request(lambda:self.put(body['id'],media_body))
    def update(self,fileId,media_body,**kwargs):
        self.calls.append(('update',fileId));return self.request(lambda:self.put(fileId,media_body))
    def get(self,fileId,fields):
        self.calls.append(('get',fileId));return self.request(lambda:self.meta(fileId))


class DurabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.file=self.root/'ledger.json';self.file.write_text('{"charged_attempts": 5}')
        self.files=Files();self.cloud=store.Cloud.__new__(store.Cloud)
        self.cloud.folder='newstore';self.cloud.owned=set();self.cloud.local=threading.local()
        self.cloud.service=lambda:types.SimpleNamespace(files=lambda:self.files)
        self.cloud.execute=lambda req:req.execute()
        self.modules=patch.dict(sys.modules,api_modules());self.modules.start()
    def tearDown(self):self.modules.stop();self.tmp.cleanup()
    def test_reuse_ledger_id_and_no_redundant_metadata_get(self):
        first=self.cloud.upload(self.file)
        self.file.write_text('{"charged_attempts": 6}')
        second=self.cloud.upload(self.file,file_id=first['id'])
        self.assertEqual(second['id'],first['id'])
        self.assertEqual(self.files.calls,[('create','id1'),('update','id1')])
        self.assertTrue(self.cloud.matches(self.file,second))
    def test_identical_acknowledged_charge_avoids_duplicate_write(self):
        first=self.cloud.upload(self.file)
        second=self.cloud.upload(self.file,file_id=first['id'])
        self.assertEqual(second,first)
        self.assertEqual(self.files.calls,[('create','id1')])
    def test_failed_upload_keeps_original_error_in_diagnostic(self):
        self.files.create=lambda **kw: self.files.request(lambda: (_ for _ in ()).throw(RuntimeError('Resume Incomplete without Location')))
        with self.assertRaisesRegex(store.DriveBlocked,'Resume Incomplete without Location'):
            self.cloud.upload(self.file)
    def test_unknown_final_outcome_queries_same_id_without_duplicate_create(self):
        self.files.timeout=True
        result=self.cloud.upload(self.file)
        self.assertEqual(result['id'],'id1')
        self.assertEqual(self.files.calls,[('create','id1'),('get','id1')])
    def test_wrong_checksum_is_not_acknowledged(self):
        self.files.bad_checksum=True
        with self.assertRaises(store.DriveBlocked):self.cloud.upload(self.file)
        self.assertTrue(self.file.exists())
    def test_zero_byte_truncation_rejected(self):
        self.assertFalse(self.cloud.matches(self.file,dict(size=0,md5Checksum=hashlib.md5(b'').hexdigest())))
    def test_transient_quota_retry_is_bounded(self):
        request=Mock();request.execute.side_effect=HTTPError()
        with patch('store.time.sleep') as sleep:
            with self.assertRaises(store.DriveBlocked):store.Cloud.execute(self.cloud,request)
            self.assertEqual(request.execute.call_count,6);self.assertEqual(sleep.call_count,5)
    def test_permission_error_is_not_retried(self):
        request=Mock();request.execute.side_effect=HTTPError(reason='insufficientPermissions')
        with patch('store.time.sleep') as sleep:
            with self.assertRaises(HTTPError):store.Cloud.execute(self.cloud,request)
            sleep.assert_not_called()

    def test_ledger_timeout_before_commit_retries_same_id_and_bytes(self):
        first=self.cloud.upload(self.file)
        self.file.write_text('{"charged_attempts": 6}')
        original=self.files.put
        attempts=[]
        def put(file_id,media):
            attempts.append((file_id,media.path.read_bytes()))
            if len(attempts)==1:raise TimeoutError('read operation timed out')
            return original(file_id,media)
        self.files.put=put
        with patch('store.time.sleep') as sleep:
            receipt=self.cloud.upload(self.file,file_id=first['id'])
        self.assertEqual(len(attempts),2)
        self.assertEqual(attempts[0],attempts[1])
        self.assertEqual(receipt['id'],first['id'])
        self.assertEqual(sleep.call_count,1)
        self.assertTrue(self.cloud.matches(self.file,receipt))
        self.assertEqual([c for c in self.files.calls if c[0]=='create'],[('create','id1')])

    def test_permanent_timeout_preserves_previous_charge_and_stops_after_six(self):
        first=self.cloud.upload(self.file)
        previous=self.files.data['id1']
        self.file.write_text('{"charged_attempts": 6}')
        attempts=[]
        def put(*args):
            attempts.append(1);raise TimeoutError('still unavailable')
        self.files.put=put
        with patch('store.time.sleep') as sleep:
            with self.assertRaisesRegex(store.DriveBlocked,'still unavailable'):
                self.cloud.upload(self.file,file_id=first['id'])
        self.assertEqual(len(attempts),6);self.assertEqual(sleep.call_count,5)
        self.assertEqual(self.files.data['id1'],previous)
        self.assertEqual(self.cloud._acknowledged['id1'],first)
        self.assertTrue(self.file.exists())

    def test_first_create_timeout_before_commit_reuses_preallocated_id(self):
        original=self.files.put;attempts=[]
        def put(file_id,media):
            attempts.append(file_id)
            if len(attempts)==1:raise TimeoutError('request never reached server')
            return original(file_id,media)
        self.files.put=put
        self.files.meta=lambda file_id: (dict(id=file_id,size=str(len(self.files.data[file_id])),
            md5Checksum=hashlib.md5(self.files.data[file_id]).hexdigest(),parents=['newstore'])
            if file_id in self.files.data else (_ for _ in ()).throw(HTTPError(404)))
        with patch('store.time.sleep'):
            receipt=self.cloud.upload(self.file)
        self.assertEqual(attempts,['id1','id1']);self.assertEqual(receipt['id'],'id1')

    def test_metadata_get_retries_network_timeout(self):
        request=Mock();request.method='GET'
        request.execute.side_effect=[TimeoutError('metadata unavailable'),{'id':'saved'}]
        with patch('store.time.sleep'):
            self.assertEqual(store.Cloud.execute(self.cloud,request),{'id':'saved'})
        self.assertEqual(request.execute.call_count,2)

    def test_unknown_generic_mutation_is_not_blindly_retried(self):
        request=Mock();request.method='POST';request.execute.side_effect=TimeoutError('unknown')
        with patch('store.time.sleep') as sleep:
            with self.assertRaises(TimeoutError):store.Cloud.execute(self.cloud,request)
        sleep.assert_not_called();self.assertEqual(request.execute.call_count,1)

    def test_permission_error_and_checksum_conflict_are_not_upload_retried(self):
        first=self.cloud.upload(self.file)
        self.file.write_text('{"charged_attempts": 6}')
        self.files.put=lambda *a: (_ for _ in ()).throw(HTTPError(403,'insufficientPermissions'))
        with patch('store.time.sleep') as sleep:
            with self.assertRaises(store.DriveBlocked):self.cloud.upload(self.file,file_id=first['id'])
        sleep.assert_not_called()


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.output=self.root/'work/task';self.output.mkdir(parents=True)
        self.cloud=Mock();self.cfg=dict(folder_id='newstore',local_root=str(self.root))
        self.env=patch.dict('os.environ',{'HIPPO_CLOUD_PHASE':'02','HIPPO_CLOUD_TASK':'{"id":"task"}'});self.env.start()
        self.conf=patch('hooks.configured',return_value=(self.cfg,self.cloud));self.conf.start()
        class Writer:
            def __init__(self,output,durable,*a,**kw):
                self.output=output;self.durable=durable;self.calls=[];self.interval=180
            def close(self,force=True):self.calls.append(force)
        self.entry=types.SimpleNamespace(write=Mock(),read=Mock(),RecoveryWriter=Writer)
    def tearDown(self):self.conf.stop();self.env.stop();self.tmp.cleanup()
    def test_failed_charge_stops_before_local_charge_or_update(self):
        self.cloud.upload.side_effect=store.DriveBlocked('quota')
        old_write=self.entry.write
        hooks.install_entry(self.entry,self.output,self.root,{})
        with self.assertRaises(store.DriveBlocked):self.entry.write(self.root/'charged_ledger.json',{'charged_attempts':1})
        old_write.assert_not_called()
        self.assertTrue((self.root/'SAVE_BLOCKED.json').exists())
        with self.assertRaises(RuntimeError):self.entry.write(self.root/'charged_ledger.json',{'charged_attempts':1})
        self.assertEqual(self.cloud.upload.call_count,1)
    def test_completed_run_commits_result_before_skip_of_forced_checkpoint(self):
        (self.output/'completion.json').write_text('{}')
        stats={};hooks.install_entry(self.entry,self.output,self.root,stats)
        writer=self.entry.RecoveryWriter(self.output,self.root/'recovery')
        with patch('hooks.commit_result',return_value=({'id':'durable_result'},{})) as commit:
            writer.close(force=True)
        commit.assert_called_once();self.assertEqual(writer.calls,[False]);self.assertTrue(writer.result_committed)
        self.assertEqual(stats['server_result_id'],'durable_result')
    def test_clean_pause_still_forces_checkpoint(self):
        hooks.install_entry(self.entry,self.output,self.root,{})
        writer=self.entry.RecoveryWriter(self.output,self.root/'recovery');writer.close(force=True)
        self.assertEqual(writer.calls,[True])
    def test_archive_failure_never_retires_checkpoint(self):
        sweep=types.SimpleNamespace(launch=Mock(),check_stop=Mock(),r=types.SimpleNamespace(),request_stop=Mock())
        hooks.install_sweep(sweep,'02')
        with patch('hooks.commit_result',side_effect=store.DriveBlocked('quota')):
            with patch('scripts.cross_dataset_campaign.storage.retire') as retire:
                with self.assertRaises(store.DriveBlocked):sweep.r.archive_completed(self.root,self.output,self.root/'durable',self.root/'recovery',{'id':'task'})
                retire.assert_not_called()
    def test_queued_snapshot_skipped_only_after_verified_result(self):
        hooks.install_entry(self.entry,self.output,self.root,{})
        writer=self.entry.RecoveryWriter(self.output,self.root/'recovery')
        writer.result_committed=True
        snapshot=self.root/'snapshot';snapshot.mkdir();(snapshot/'checkpoint_latest.pt').write_bytes(b'checkpoint')
        writer._persist(snapshot)
        self.cloud.put_immutable.assert_not_called();self.assertFalse(snapshot.exists())
    def test_lossless_snapshot_and_failed_upload_retention(self):
        hooks.install_entry(self.entry,self.output,self.root,{})
        writer=self.entry.RecoveryWriter(self.output,self.root/'recovery')
        snapshot=self.root/'snapshot';snapshot.mkdir()
        (snapshot/'checkpoint_latest.pt').write_bytes(b'exact original checkpoint'*100)
        store.atomic(snapshot/'budget_ledger.json',{'charged_attempts':5})
        self.cloud.put_immutable.side_effect=store.DriveBlocked('quota')
        with self.assertRaises(store.DriveBlocked):writer._persist(snapshot)
        self.assertTrue(snapshot.exists());self.assertTrue(snapshot.with_suffix('.tar.gz').exists())
        import tarfile
        with tarfile.open(snapshot.with_suffix('.tar.gz')) as tar:
            self.assertEqual(tar.extractfile('checkpoint_latest.pt').read(),(snapshot/'checkpoint_latest.pt').read_bytes())


class RecoveryTests(unittest.TestCase):
    def test_released_lease_uses_new_id_and_active_lease_blocks(self):
        cloud=Mock();cloud.folder='newstore'
        cloud.metadata.side_effect=[{'parents':['newstore'],'appProperties':{'kind':'released_lease','next_lease_id':'fresh'}},HTTPError(404)]
        with patch.dict(sys.modules,api_modules()):
            self.assertEqual(next_lease(cloud,'used'),'fresh')
            cloud.metadata.side_effect=None
            cloud.metadata.return_value={'parents':['newstore'],'appProperties':{'kind':'lease'}}
            with self.assertRaisesRegex(RuntimeError,'existing controller'):
                next_lease(cloud,'active')
    def test_release_preserves_lease_and_allocates_successor(self):
        cloud=Mock();cloud.folder='newstore'
        cloud.metadata.return_value={'parents':['newstore'],'appProperties':{'kind':'lease','token':'owner'}}
        cloud.execute.side_effect=[{'ids':['successor']},{'id':'old'}]
        release_lease(cloud,'old','owner')
        args=cloud.service().files().update.call_args.kwargs
        self.assertEqual(args['body']['appProperties']['next_lease_id'],'successor')
        self.assertEqual(args['body']['appProperties']['kind'],'released_lease')
        cloud.service().files().delete.assert_not_called()
    def test_checkpoint_cannot_exceed_new_server_charge(self):
        with self.assertRaisesRegex(ValueError,'exceeds'):
            recovery_event('task',{'charged_attempts':5},{'charged_attempts':6},{},'server_store')
    def test_lost_charge_is_explicit_and_not_erased(self):
        e=recovery_event('task',{'charged_attempts':110,'training_flops':11000,'calibration_flops':30},
            {'charged_attempts':80,'training_flops':8000,'calibration_flops':20},{},'test')
        self.assertEqual(e['lost_charged_attempts'],30);self.assertEqual(e['lost_training_flops'],3000)
        self.assertEqual(e['lost_calibration_flops'],10)
    def test_unknown_cost_is_not_zero(self):
        e=recovery_event('task',{},None,{},'test')
        self.assertIsNone(e['lost_training_flops']);self.assertIsNone(e['lost_charged_attempts'])
    def test_exact_empty_lock_is_valid(self):
        checked_bytes('run.lock',b'',dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest()))
    def test_empty_json_cannot_pass_readability(self):
        with self.assertRaises(ValueError):checked_bytes('completion.json',b'',dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest()))
    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','no')
            with self.assertRaises(ValueError):store.extract_zip(p,root/'out')
            self.assertFalse((root/'escape').exists())
    def test_completion_cannot_be_derived_from_summary_alone(self):
        self.assertIsNone(derive_completion({},dict(updates=10),None))
    def test_server_committed_result_survives_missing_cleanup_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cache=root/'cache';cache.mkdir();local=root/'local'
            completion=json_bytes({'status':'complete','updates':10})
            manifest={'task':{'id':'task'},'metadata':{'completion.json':{'bytes':len(completion),'sha256':hashlib.sha256(completion).hexdigest()}}}
            archive=cache/'result.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('completion.json',completion);z.writestr('archive_manifest.json',json_bytes(manifest))
            obj={'id':'committed','appProperties':{'kind':'result','task':'task','phase':'02','sha256':store.digest(archive)}}
            cloud=Mock();cloud.download.return_value=archive
            result=restore_server_results(cloud,[obj],local,cache,{'02':{'bundle_sha256':'source'}})
            self.assertEqual(result,{'task'})
            p=local/'02/runs/task'
            self.assertEqual((p/'completion.json').read_bytes(),completion)
            ack=json.loads((p/'archive_acknowledgment.json').read_text())
            self.assertEqual(ack['server_commit']['id'],'committed')
            self.assertEqual(json.loads((p/'cleanup.json').read_text())['deleted'],[])
    def test_conflicting_committed_results_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cache=root/'cache';cache.mkdir();paths=[];objects=[]
            for i in (1,2):
                data=json_bytes({'status':'complete','updates':i})
                m={'task':{'id':'task'},'metadata':{'completion.json':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
                p=cache/(str(i)+'.zip')
                with zipfile.ZipFile(p,'w') as z:z.writestr('completion.json',data);z.writestr('archive_manifest.json',json_bytes(m))
                paths.append(p);objects.append({'id':str(i),'appProperties':{'kind':'result','task':'task','phase':'02','sha256':store.digest(p)}})
            cloud=Mock();cloud.download.side_effect=paths
            with self.assertRaisesRegex(RuntimeError,'Conflicting completed'):
                restore_server_results(cloud,objects,root/'local',cache,{'02':{'bundle_sha256':'source'}})


if __name__=='__main__':unittest.main()
