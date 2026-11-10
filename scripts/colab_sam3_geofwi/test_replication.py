"""CPU checks for scientific scope, controls, cloud acknowledgment and recovery."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from scripts.colab_sam3_geofwi import runtime as rt, missing_constraints as recipe, diagnose as diag
from scripts.colab_sam3_geofwi import replication as runner
from scripts.colab_sam3_geofwi.test_missing_constraints import fixture
from scripts.colab_sam3_geofwi.test_diagnose import Tiny
from scripts.colab_msd_cloud import store
from scripts.colab_drive_transport import request_gate
sys.modules.setdefault('store',store)
sys.modules.setdefault('request_gate',request_gate)
sys.modules.setdefault('runtime',rt)
from scripts.colab_sam3_geofwi import replication_store as rs


class FakeCloud:
    def __init__(self, folder): self.folder=folder; self.data={}; self.deleted=[]; self.fail_index=False; self.counter=0
    def check_account(self): return {'user':{'emailAddress':rs.ACCOUNT},'storageQuota':{}}
    def children(self): return []
    def upload(self,path,name=None,properties=None,file_id=None):
        if self.fail_index and name=='campaign_state.json': raise RuntimeError('quota')
        self.counter+=1
        identifier=file_id or str(self.counter)
        payload=Path(path).read_bytes()
        meta=dict(id=identifier,name=name,size=len(payload),parents=[self.folder],
                  md5Checksum=hashlib.md5(payload).hexdigest(),local_sha256=hashlib.sha256(payload).hexdigest())
        self.data[identifier]=(payload,meta);return meta
    def put_immutable(self,path,kind,task): return self.upload(path,kind+task)
    def metadata(self,key): return self.data[key][1]
    def matches(self,path,meta): return store.Cloud.matches(path,meta)
    def download(self,key,path,expected_sha=None):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_bytes(self.data[key][0])
        assert not expected_sha or store.digest(path)==expected_sha
    def service(self):return self
    def files(self):return self
    def delete(self,fileId):return fileId
    def execute(self,key):self.deleted.append(key);del self.data[key]


class ReplicationTests(unittest.TestCase):
    def test_scope_and_disjoint_folds(self):
        items=runner.tasks()
        self.assertEqual(len(items),126)
        self.assertEqual(len({x['id'] for x in items}),126)
        from scripts.cross_dataset_campaign.common import ARMS
        self.assertEqual(runner.ARMS,ARMS)
        root=Path(__file__).resolve().parents[2]/'reports/sam3_geofwi_f013_s012_126_20261019/bundle'
        cohorts=[]
        for fold in runner.FOLDS:
            protocols=[rt.read(root/f'protocols/f{fold}s{s}.json') for s in runner.SEEDS]
            self.assertEqual(len({tuple(p['train_cases']) for p in protocols}),1)
            self.assertEqual(len({tuple(p['evaluation_cases']) for p in protocols}),1)
            for p in protocols:
                self.assertEqual((len(p['train_cases']),len(p['evaluation_cases']),p['updates']),(5,96,25))
                self.assertFalse(set(p['train_cases'])&set(p['evaluation_cases']))
            cohorts.append(set(protocols[0]['evaluation_cases']))
        self.assertEqual(len(set.union(*cohorts)),288)

    def test_controls_and_adapted_edge_use_exactly_25_updates(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(recipe,'ARMS',runner.ARMS),redirect_stdout(io.StringIO()):
            for arm in ('dice','dice_bands','dice_bands_edge'):
                original,slabs,p,shared=fixture(arm);original.stopping=lambda:False
                output=Path(tmp)/arm;recovery=Path(tmp)/(arm+'_recovery')
                rt.write(output/'protocol.json',p)
                writer=rt.RecoveryWriter(output,recovery,interval=99999)
                model=Tiny()
                try:recipe.train(original,diag,rt,model,None,[model.lora_A],slabs,p,shared,2.,output,recovery,writer,Path(tmp)/'durable')
                finally:writer.close()
                history=rt.read(output/'training_history.json');cal=rt.read(output/'calibration.json')
                self.assertEqual(len(history),25)
                for row in history:
                    self.assertEqual(row['weights']['bands'],2. if 'bands' in arm else 0.)
                    self.assertEqual(row['weights']['sdf'],0.)
                    self.assertEqual(row['weights']['depth'],0.)
                    self.assertEqual(row['weights']['edge']>0,'edge' in arm and row['update']>5)
                self.assertEqual(len(cal['rows']),20 if 'edge' in arm else 0)
                self.assertEqual(len(cal['monitoring']['25']),0 if arm=='dice' else 5)
                sha=rt.digest(output/'checkpoint_latest.pt')
                # Re-enter completed training without changing the evaluated checkpoint identity.
                model=Tiny();writer=rt.RecoveryWriter(output,recovery,interval=99999)
                try:recipe.train(original,diag,rt,model,None,[model.lora_A],slabs,p,shared,2.,output,recovery,writer,Path(tmp)/'durable')
                finally:writer.close()
                self.assertEqual(sha,rt.digest(output/'checkpoint_latest.pt'))

    def test_snapshot_integrity_and_stable_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'work/run/file.json';rt.write(source,{'a':1})
            archive=root/'snapshot.zip';rs.pack(root,[source],archive);sha=rt.digest(archive)
            source.touch();rs.pack(root,[source],archive);self.assertEqual(sha,rt.digest(archive))
            source.unlink();rs.unpack(archive,root);self.assertEqual(rt.read(source),{'a':1})
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('work/run/file.json','bad')
                z.writestr('_snapshot_manifest.json',json.dumps({'work/run/file.json':'wrong'}))
            with self.assertRaises(ValueError):rs.unpack(archive,root)
            self.assertEqual(rt.read(source),{'a':1})

    def test_cloud_pointer_must_commit_before_old_snapshot_deleted(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rs,'Cloud',FakeCloud):
            root=Path(tmp);rt.write(root/'bundle_manifest.json',{})
            t=rs.Transport(root);path=root/'work/run/checkpoint_latest.pt';rt.write(path,{'step':1})
            first=t.publish('recovery/run',[path])
            rt.write(path,{'step':2});t.cloud.fail_index=True
            with self.assertRaises(RuntimeError):t.publish('recovery/run',[path])
            self.assertIn(first['id'],t.cloud.data)
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):t.retire('run')
            self.assertTrue(path.exists())

    def test_lost_charged_update_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rs,'Cloud',FakeCloud):
            root=Path(tmp);rt.write(root/'bundle_manifest.json',{})
            t=rs.Transport(root);task=runner.tasks()[0];output=root/'work'/task['id']
            rt.write(output/'checkpoint_latest.pt',{'step':3})
            rt.write(output/'checkpoint_ledger.json',{'step':3,'protocol_sha256':'p'})
            t.charge(task['id'],{'step':4,'protocol_sha256':'p'})
            with self.assertRaisesRegex(RuntimeError,'charged updates'):runner.restore_run(t,task)
            self.assertTrue((output/'checkpoint_latest.pt').exists())

    def test_wrong_account_and_low_storage_fail(self):
        cloud=object.__new__(rs.Cloud)
        with patch.object(cloud,'about',return_value={'user':{'emailAddress':'wrong@example.com'},'storageQuota':{}}):
            with self.assertRaises(RuntimeError):cloud.check_account()
        with patch.object(cloud,'about',return_value={'user':{'emailAddress':rs.ACCOUNT},'storageQuota':{'limit':'100','usage':'0'}}):
            with self.assertRaises(RuntimeError):cloud.check_account()

    def test_worker_archives_before_checkpoint_retirement(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rs,'Cloud',FakeCloud),patch.object(recipe,'ARMS',runner.ARMS),redirect_stdout(io.StringIO()):
            root=Path(tmp);rt.write(root/'bundle_manifest.json',{})
            t=rs.Transport(root);t.state['binding']={}
            task=next(t for t in runner.tasks() if t['arm']=='dice_bands_edge_sdf_depth')
            original,slabs,p,shared=fixture(task['arm']);p.update(fold=0,seed=0)
            original.stopping=lambda:False
            shared['weights']={'bands':2.}
            rt.write(root/'durable/calibration/f0s0/calibration.json',shared)
            def load(root,p):
                model=Tiny();slabs.protocol=p
                return original,model,None,[model.lora_A],None,None,slabs
            fake=SimpleNamespace(load=load,write=rt.write)
            def evaluate(original,model,processor,volumes,p,output,root,durable):
                self.assertEqual(len(rt.read(output/'training_history.json')),25)
                records=[dict(split=s,case_name=c,confusion=[[1,0],[0,1]],foreground_fn=0,foreground_fp=0,truth_sha256=c)
                    for s,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])] for c in cases]
                result=dict(records=records,mean_dice={'training_support':1.,'validation':1.})
                fake.write(output/'evaluation.json',result);return result
            fake.evaluate=evaluate
            modules={'engine':fake,'runtime':rt,'missing_constraints':recipe,'sam3_edge_diagnose_v1':diag,'replication_store':rs}
            with patch.dict(sys.modules,modules),patch.object(runner,'protocol',return_value=p):
                runner.run_one(root,task,t)
            self.assertIn(task['id'],t.state['completed'])
            self.assertNotIn(task['id'],t.state['charges'])
            self.assertNotIn('recovery/'+task['id'],t.state['objects'])
            self.assertFalse((root/'work'/task['id']/'checkpoint_latest.pt').exists())
            self.assertTrue(t.restore('completed/'+task['id']))
            runner.validate_complete(rt,root/'work'/task['id'],p)


if __name__=='__main__':unittest.main()
