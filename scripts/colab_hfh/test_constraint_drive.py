import unittest
import hashlib
import json
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.colab_hfh.constraint_drive import Drive, resolve_folder, upload_verified, STORAGE_ROOT, STORAGE_EMAIL, FOLDER_MIME, MAX_REQUEST_ATTEMPTS, load_metadata_cache, save_metadata_cache


class FolderResolutionTests(unittest.TestCase):
    def folder(self):
        return dict(id='folder123',name='results_constraints_test',mimeType=FOLDER_MIME,
                    parents=[STORAGE_ROOT],owners=[{'emailAddress':STORAGE_EMAIL}])

    def test_existing_folder_never_creates(self):
        calls=[]
        def request(method,path,**kwargs):
            calls.append((method,path,kwargs))
            return {'files':[self.folder()]}
        self.assertEqual(resolve_folder(request,'results_constraints_test'),'folder123')
        self.assertEqual(len(calls),1)
        self.assertIn("name='results_constraints_test'",calls[0][2]['params']['q'])

    def test_missing_folder_created_in_second_account_root(self):
        calls=[]
        def request(method,path,**kwargs):
            calls.append((method,kwargs))
            return {'files':[]} if method=='GET' else self.folder()
        self.assertEqual(resolve_folder(request,'results_constraints_test'),'folder123')
        self.assertEqual([c[0] for c in calls],['GET','POST'])
        self.assertEqual(calls[1][1]['json']['parents'],[STORAGE_ROOT])

    def test_ambiguous_and_wrong_identity_fail_closed(self):
        wrong_owner=self.folder();wrong_owner['owners']=[{'emailAddress':'wrong@example.com'}]
        wrong_parent=self.folder();wrong_parent['parents']=['wrong']
        wrong_type=self.folder();wrong_type['mimeType']='text/plain'
        cases=[{'files':[self.folder(),self.folder()]},{'files':[],'nextPageToken':'more'},
               {'files':[wrong_owner]},{'files':[wrong_parent]},{'files':[wrong_type]}]
        for result in cases:
            with self.subTest(result=result),self.assertRaises(ValueError):
                resolve_folder(lambda *a,**k:result,'results_constraints_test')

    def test_unsafe_name_rejected_before_request(self):
        def forbidden(*a,**k):raise AssertionError('No network call permitted')
        with self.assertRaises(ValueError):resolve_folder(forbidden,"results_'x")

    def test_cached_folder_avoids_search_but_revalidates_identity(self):
        calls=[]
        def request(method,path,**kwargs):
            calls.append((method,path));return self.folder()
        self.assertEqual(resolve_folder(request,'results_constraints_test','folder123'),'folder123')
        self.assertEqual(calls,[('GET','files/folder123')])
        for change in [{'parents':['other']},{'trashed':True},
                       {'owners':[{'emailAddress':'other@example.com'}]}]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                resolve_folder(lambda *a,**k:{**self.folder(),**change},'results_constraints_test','folder123')


class MetadataCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.directory=Path(self.temporary.name)
        patcher=patch('scripts.colab_hfh.constraint_drive.CACHE_DIRECTORY',self.directory)
        patcher.start();self.addCleanup(patcher.stop)

    def test_search_folder_is_discovered_on_first_use(self):
        folder,files=load_metadata_cache('results_constraint_search_v1')
        self.assertEqual((folder,files),(None,{}))

    def test_cache_roundtrip_is_scoped_per_campaign(self):
        save_metadata_cache('results_one','folder1',{'run.zip':'file1'})
        save_metadata_cache('results_two','folder2',{})
        self.assertEqual(load_metadata_cache('results_one'),('folder1',{'run.zip':'file1'}))
        self.assertEqual(load_metadata_cache('results_two'),('folder2',{}))
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_wrong_scope_or_unsafe_id_fails_closed(self):
        for change in [{'owner':'other@example.com'},{'root':'other'},
                       {'folder_id':'../unsafe'},{'files':{'run.zip':'../unsafe'}}]:
            save_metadata_cache('results_one','folder1',{})
            path=self.directory/'results_one.json';data=json.loads(path.read_text());data.update(change)
            path.write_text(json.dumps(data))
            with self.subTest(change=change),self.assertRaises(ValueError):load_metadata_cache('results_one')
        with self.assertRaises(ValueError):load_metadata_cache('../unsafe')


class VerifiedUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.path=Path(self.temporary.name)/'campaign_status.json'
        self.payload=b'{"status":"complete"}\n';self.path.write_bytes(self.payload)
        self.metadata=dict(id='file123',name=self.path.name,mimeType='application/json',
            parents=['folder123'],owners=[{'emailAddress':STORAGE_EMAIL}],
            size=str(len(self.payload)),md5Checksum=hashlib.md5(self.payload).hexdigest())

    def test_existing_matching_bytes_are_reused_without_write(self):
        calls=[]
        def request(method,endpoint,**kwargs):
            calls.append(method);return {'files':[self.metadata]}
        upload_verified(request,'folder123',self.path)
        self.assertEqual(calls,['GET'])

    def test_replacement_preserves_id_and_verifies_persisted_hash(self):
        calls=[]
        def request(method,endpoint,**kwargs):
            calls.append((method,endpoint))
            if endpoint=='files':return {'files':[{**self.metadata,'md5Checksum':'old'}]}
            if method=='PATCH':
                self.assertEqual(kwargs['data'],self.payload);self.assertTrue(kwargs['upload'])
            return self.metadata
        upload_verified(request,'folder123',self.path)
        self.assertEqual(calls,[('GET','files'),('PATCH','files/file123'),('GET','files/file123')])

    def test_create_has_explicit_parent_and_independent_readback(self):
        calls=[]
        def request(method,endpoint,**kwargs):
            calls.append((method,endpoint))
            if method=='GET' and endpoint=='files':return {'files':[]}
            if method=='POST':
                self.assertIn(b'"parents": ["folder123"]',kwargs['data'])
                self.assertIn(self.payload,kwargs['data'])
            return self.metadata
        upload_verified(request,'folder123',self.path)
        self.assertEqual(calls,[('GET','files'),('POST','files'),('GET','files/file123')])

    def test_corrupt_readback_never_marks_success(self):
        def request(method,endpoint,**kwargs):
            if endpoint=='files' and method=='GET':return {'files':[]}
            return self.metadata if method=='POST' else {**self.metadata,'md5Checksum':'wrong'}
        with self.assertRaisesRegex(ValueError,'checksum'):
            upload_verified(request,'folder123',self.path)

    def test_wrong_owner_duplicate_and_invalid_name_fail_before_write(self):
        for files in ([{**self.metadata,'owners':[{'emailAddress':'other@example.com'}]}],
                      [self.metadata,self.metadata]):
            def request(method,endpoint,**kwargs):
                self.assertEqual(method,'GET');return {'files':files}
            with self.assertRaises(ValueError):upload_verified(request,'folder123',self.path)
        with self.assertRaises(ValueError):
            upload_verified(lambda *a,**k:self.fail('No request'), 'folder123',self.path,'../escape')

    def test_lost_create_is_not_blindly_replayed(self):
        writes=[]
        def request(method,endpoint,**kwargs):
            if method=='GET':return {'files':[]}
            writes.append(method);raise RuntimeError('Lost response')
        with self.assertRaises(RuntimeError):upload_verified(request,'folder123',self.path)
        self.assertEqual(writes,['POST'])

    def test_fresh_folder_create_skips_query_but_reads_back_and_caches(self):
        calls=[];cache={}
        def request(method,endpoint,**kwargs):
            calls.append((method,endpoint));return self.metadata
        upload_verified(request,'folder123',self.path,file_ids=cache,fresh_folder=True)
        self.assertEqual(calls,[('POST','files'),('GET','files/file123')])
        self.assertEqual(cache,{self.path.name:'file123'})
        calls.clear()
        upload_verified(request,'folder123',self.path,file_ids=cache,fresh_folder=True)
        self.assertEqual(calls,[('GET','files/file123')])

    def test_lost_fresh_create_requires_recovery_discovery(self):
        calls=[];cache={}
        def request(method,endpoint,**kwargs):
            calls.append(method);raise RuntimeError('Lost POST response')
        with self.assertRaises(RuntimeError):
            upload_verified(request,'folder123',self.path,file_ids=cache,fresh_folder=True)
        self.assertEqual(calls,['POST']);self.assertEqual(cache,{})

    def test_known_ids_avoid_name_search_but_revalidate_owner_and_parent(self):
        calls=[];cache={self.path.name:'file123'}
        def request(method,endpoint,**kwargs):
            calls.append((method,endpoint));return self.metadata
        upload_verified(request,'folder123',self.path,file_ids=cache)
        self.assertEqual(calls,[('GET','files/file123')])
        with self.assertRaises(ValueError):
            upload_verified(lambda *a,**k:{**self.metadata,'parents':['moved_elsewhere']},
                            'folder123',self.path,file_ids=cache)
        with self.assertRaises(ValueError):
            upload_verified(lambda *a,**k:{**self.metadata,'trashed':True},
                            'folder123',self.path,file_ids=cache)


class RequestRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        patcher=patch('scripts.colab_hfh.constraint_drive.CACHE_DIRECTORY',Path(self.temporary.name))
        patcher.start();self.addCleanup(patcher.stop)
        self.session=Mock()
        parser=Mock()
        parser.__getitem__=Mock(return_value={'root_folder_id':STORAGE_ROOT,
            'token':json.dumps({'access_token':'fixture'})})
        for name,value in [('requests.Session',self.session),
                           ('scripts.colab_hfh.constraint_drive.configparser.ConfigParser',parser)]:
            patcher=patch(name,return_value=value);patcher.start();self.addCleanup(patcher.stop)
        for name in ['scripts.colab_hfh.constraint_drive.time.sleep','builtins.print']:
            patcher=patch(name);mock=patcher.start();self.addCleanup(patcher.stop)
            if name.endswith('sleep'):self.sleep=mock
        self.folder=dict(id='folder123',name='results_constraints_test',mimeType=FOLDER_MIME,
            parents=[STORAGE_ROOT],owners=[{'emailAddress':STORAGE_EMAIL}])

    def response(self,status,body):
        response=Mock(status_code=status);response.json.return_value=body;return response

    def quota(self):
        return self.response(403,{'error':{'errors':[{'reason':'rateLimitExceeded'}]}})

    def ready(self):
        self.session.request.side_effect=[self.response(200,{'files':[self.folder]})]
        drive=Drive('results_constraints_test');self.session.request.reset_mock();self.sleep.reset_mock()
        return drive

    def test_recovers_quota_burst_longer_than_previous_window(self):
        self.session.request.side_effect=[self.quota() for _ in range(8)]+[
            self.response(200,{'files':[self.folder]})]
        self.assertEqual(Drive('results_constraints_test').folder_id,'folder123')
        self.assertEqual(self.session.request.call_count,9)

    def test_persistent_quota_is_bounded(self):
        self.session.request.side_effect=[self.quota() for _ in range(MAX_REQUEST_ATTEMPTS)]
        with self.assertRaisesRegex(RuntimeError,'403 rateLimitExceeded'):
            Drive('results_constraints_test')
        self.assertEqual(self.session.request.call_count,MAX_REQUEST_ATTEMPTS)
        waits=[call.args[0] for call in self.sleep.call_args_list if call.args[0]>=1]
        self.assertEqual(sum(waits),363)

    def test_permission_failure_is_not_retried(self):
        self.session.request.side_effect=[self.response(403,{'error':{'errors':[{'reason':'insufficientPermissions'}]}})]
        with self.assertRaisesRegex(RuntimeError,'insufficientPermissions'):
            Drive('results_constraints_test')
        self.assertEqual(self.session.request.call_count,1)

    def test_confirmed_post_quota_can_recover(self):
        drive=self.ready()
        self.session.request.side_effect=[self.quota() for _ in range(8)]+[self.response(201,{'id':'new'})]
        self.assertEqual(drive.request('POST','files',json={}),{'id':'new'})
        self.assertEqual(self.session.request.call_count,9)

    def test_ambiguous_post_network_or_server_failure_never_replays(self):
        import requests
        drive=self.ready()
        for failure in [requests.ConnectionError('lost response'),self.response(503,{})]:
            with self.subTest(failure=type(failure).__name__):
                self.session.request.reset_mock();self.session.request.side_effect=[failure]
                with self.assertRaises(RuntimeError):drive.request('POST','files',json={})
                self.assertEqual(self.session.request.call_count,1)

    def test_only_new_empty_folder_skips_recovery_listing(self):
        self.session.request.side_effect=[self.response(200,{'files':[]}),self.response(201,self.folder)]
        drive=Drive('results_constraints_test')
        with patch('scripts.colab_hfh.bands_campaign.Drive.call',return_value='downloaded') as inherited:
            self.assertEqual(drive.call('copy',drive.folder,'local'),'')
            inherited.assert_not_called()
            drive.file_ids['campaign_status.json']='file1'
            self.assertEqual(drive.call('copy',drive.folder,'local'),'downloaded')
        self.session.request.side_effect=[self.response(200,self.folder)]
        resumed=Drive('results_constraints_test')
        self.assertFalse(resumed.fresh_folder)
        with patch('scripts.colab_hfh.bands_campaign.Drive.call',return_value='downloaded'):
            self.assertEqual(resumed.call('copy',resumed.folder,'local'),'downloaded')


class OptionalReportTests(unittest.TestCase):
    def test_quota_deferred_and_mandatory_retry_budget_restored(self):
        drive=object.__new__(Drive);drive.request_attempts=12
        for message in ('Drive folder request failed with HTTP 403 rateLimitExceeded',
                        'Drive folder request failed with HTTP 429 '):
            def fail(*args):
                self.assertEqual(drive.request_attempts,2)
                raise RuntimeError(message)
            with patch.object(drive,'upload',side_effect=fail):
                self.assertFalse(drive.upload_report(Path('matrix_summary.json')))
            self.assertEqual(drive.request_attempts,12)

    def test_permission_and_integrity_errors_are_not_suppressed(self):
        drive=object.__new__(Drive);drive.request_attempts=12
        for error in (RuntimeError('HTTP 403 insufficientPermissions'),ValueError('Remote checksum mismatch')):
            with patch.object(drive,'upload',side_effect=error):
                with self.assertRaises(type(error)):drive.upload_report(Path('matrix_summary.json'))
            self.assertEqual(drive.request_attempts,12)

    def test_results_and_recovery_never_use_optional_sync(self):
        drive=object.__new__(Drive);drive.request_attempts=12
        for name in ('nnunet_f0_s0.zip','campaign_status.json','search_status.json'):
            with self.assertRaises(ValueError):drive.upload_report(Path(name))
        self.assertEqual(drive.request_attempts,12)

    def test_success_still_requires_verified_upload(self):
        drive=object.__new__(Drive);drive.request_attempts=12
        with patch.object(drive,'upload') as upload:
            self.assertTrue(drive.upload_report(Path('full_constraint_matrix.csv')))
            upload.assert_called_once()
        self.assertEqual(drive.request_attempts,12)


if __name__=='__main__':unittest.main()
