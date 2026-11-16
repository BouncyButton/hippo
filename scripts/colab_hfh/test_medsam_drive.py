"""Storage identity, independent verification, and recovery regression checks."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from scripts.colab_hfh import medsam_constraints_drive as drive
from scripts.colab_hfh import medsam_recovery as recovery


class DriveTests(unittest.TestCase):
    def service(self, email=drive.EMAIL, owner=drive.EMAIL):
        service = MagicMock()
        service.about().get().execute.return_value = dict(user={'emailAddress': email}, storageQuota={'usage':'10','limit':'100'})
        service.files().get().execute.return_value = dict(id=drive.ROOT_ID, mimeType=drive.FOLDER, owners=[{'emailAddress':owner}])
        return service

    def test_wrong_account_or_owner_prevents_any_write(self):
        for email,owner in [('compute@example.org',drive.EMAIL),(drive.EMAIL,'someone@example.org')]:
            service = self.service(email,owner)
            with self.assertRaises(ValueError):
                drive.Drive(service)
            service.files().create.assert_not_called()
            service.files().update.assert_not_called()

    def test_duplicate_names_rejected(self):
        service=self.service();api=drive.Drive(service)
        service.files().list().execute.return_value={'files':[{'name':'x','id':'a'},{'name':'x','id':'b'}]}
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            api.resolve('hfh_storage:x')

    def test_ambiguous_create_verifies_preallocated_id(self):
        service=self.service();api=drive.Drive(service)
        service.files().list().execute.return_value={'files':[]}
        service.files().generateIds().execute.return_value={'ids':['reserved']}
        service.files().create().execute.side_effect=TimeoutError()
        service.files().get().execute.return_value={'id':'reserved','name':'x','mimeType':drive.FOLDER}
        self.assertEqual(api.resolve('hfh_storage:x',create=True)['id'],'reserved')
        self.assertEqual(service.files().create().execute.call_count,1)

    def test_upload_checks_independent_remote_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'result.json';source.write_bytes(b'original')
            service=self.service();api=drive.Drive(service)
            api.resolve=MagicMock(side_effect=[{'id':'folder'}, {'id':'file'}])
            api.metadata=MagicMock(return_value={'id':'file','size':'8','md5Checksum':'bad'})
            service.files().update().next_chunk.return_value=(None,{'id':'file'})
            http = types.ModuleType('googleapiclient.http')
            http.MediaFileUpload = MagicMock()
            with patch.dict('sys.modules', {'googleapiclient.http': http}),patch.object(drive.time,'sleep'):
                with self.assertRaisesRegex(IOError,'independent checksum'):
                    api.upload(source,'hfh_storage:folder/result.json')
            self.assertEqual(source.read_bytes(),b'original')

    def test_failed_patch_retries_same_id_and_requires_verified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'latest.json';source.write_bytes(b'new pointer')
            service=self.service();api=drive.Drive(service)
            api.resolve=MagicMock(side_effect=[{'id':'folder'},{'id':'same-file'}])
            wrong={'id':'same-file','size':'3','md5Checksum':'old'}
            correct={'id':'same-file','size':str(source.stat().st_size),'md5Checksum':drive.checksum(source)}
            api.metadata=MagicMock(side_effect=[wrong,wrong,wrong,correct])
            service.files().update().next_chunk.side_effect=[TimeoutError('network interrupted'),(None,{'id':'same-file'})]
            http=types.ModuleType('googleapiclient.http');http.MediaFileUpload=MagicMock()
            with patch.dict('sys.modules',{'googleapiclient.http':http}),patch.object(drive.time,'sleep'):
                api.upload(source,'hfh_storage:run/latest.json')
            self.assertEqual(service.files().update().next_chunk.call_count,2)
            service.files().create.assert_not_called()
            self.assertEqual(api.cache[('run','latest.json')]['md5Checksum'],drive.checksum(source))

    def test_remote_recovery_downloaded_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'output';remote=root/'run/recovery'
            remote.mkdir(parents=True)
            api=MagicMock()
            marker={'file':'recovery-'+'a'*64+'.tar','sha256':'a'*64}
            def download(source,target):
                target.write_text(json.dumps(marker) if target.name=='latest.json' else 'archive')
            api.download.side_effect=download
            with patch.dict(os.environ,HFH_DRIVE_TRANSPORT='colab_adc'),patch.object(recovery,'MOUNT',root),\
                    patch.object(drive,'client',return_value=api),patch.object(recovery.runtime,'restore') as restore:
                recovery.restore(output,remote)
                self.assertEqual(api.download.call_count,2)
                restore.assert_called_once_with(output,remote)

    def test_existing_recovery_without_pointer_never_retrains(self):
        api=MagicMock();api.download.side_effect=FileNotFoundError('latest.json')
        with patch.dict(os.environ,HFH_DRIVE_TRANSPORT='colab_adc'),patch.object(drive,'client',return_value=api),\
                patch.object(recovery.runtime,'restore') as restore:
            with self.assertRaises(FileNotFoundError):
                recovery.restore(Path('/tmp/output'),recovery.MOUNT/'run/recovery')
            restore.assert_not_called()

    def test_paths_and_recovery_deletion_fail_closed(self):
        for remote in ('other:foo','hfh_storage:../foo','hfh_storage:a/../../b'):
            with self.assertRaises(ValueError): drive.parts(remote)
        api=drive.Drive(self.service())
        with self.assertRaises(OSError): api.command('delete','hfh_storage:recovery')


if __name__ == '__main__':
    unittest.main()
