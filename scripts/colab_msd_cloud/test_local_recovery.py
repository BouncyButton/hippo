"""Same-VM timeout recovery keeps checkpoint bytes and charges the lost tail."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from driver import recovery_event
from local_recovery import checkpoint_ledger, reconcile_local
from store import atomic, digest, DriveBlocked


class LocalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.output=self.root/'work/task';self.output.mkdir(parents=True)
        self.local=self.root/'local';self.cache=self.root/'cache';self.cache.mkdir()
        self.task=dict(id='task',model='unetr_vit_b16',arm='dice_bands_sdf',fold=1,seed=2,protocol='test')
        self.cfg=dict(model_identity='unetr_vit_b16',objective='dice_bands_sdf',fold=1,seed=2,
                      augmentation='none',budget_flops=31976450667090)
        atomic(self.output/'config.json',self.cfg)
        self.ledger=dict(config_sha256=digest(self.output/'config.json'),charged_attempts=42,
                         training_flops=4200,calibration_flops=0,calibration_attempts=0)
        self.state=dict(configuration=self.cfg,step=40,charged_attempts=40,model={'weight':torch.tensor([1.,2.])},
                        optimizer={'momentum':torch.tensor([.1,.2])},torch_rng=torch.get_rng_state())
        self.schedule=dict(prefix_updates=50,sdf_prefix_updates=50,baseline_flops_per_update=100,
                           early_flops_per_update=120,flops_per_update=120,calibration_flops=300,max_steps=100)
        self.protocol=dict(budget_flops=31976450667090,trainer_alias='unet',models={'unet':{'objectives':{'dice_bands_sdf':self.schedule}}})
        atomic(self.root/'protocols/test/protocol.json',self.protocol)
        self.cp=self.output/'checkpoint_latest.pt';torch.save(self.state,self.cp)
        atomic(self.output/'budget_ledger.json',self.ledger)
        self.recovery=self.local/'02/recovery/task/charged_ledger.json';atomic(self.recovery,self.ledger)
        self.server=self.root/'server.json';atomic(self.server,self.ledger)
        self.cloud=Mock();self.cloud.download.return_value=self.server
        self.events=[];self.order=[]
        def event(path,*args):
            self.order.append('evidence');self.events.append(Path(path).read_bytes())
        def upload(path,**kwargs):
            self.order.append('ledger')
            self.assertEqual(json.loads((self.output/'budget_ledger.json').read_text()),self.ledger)
            self.server.write_bytes(Path(path).read_bytes())
            return dict(id='ledger')
        self.cloud.put_immutable.side_effect=event;self.cloud.upload.side_effect=upload
        self.group=[dict(id='ledger',appProperties={'kind':'ledger','phase':'02','task':'task'})]

    def reconcile(self):
        reconcile_local(self.cloud,self.group,self.output,self.root,self.local,self.cache,'02',self.task,recovery_event)

    def test_newer_local_checkpoint_used_and_only_lost_tail_charged_as_overhead(self):
        original=digest(self.cp)
        self.reconcile()
        self.assertEqual(self.order,['evidence','ledger'])
        saved=json.loads((self.output/'budget_ledger.json').read_text())
        self.assertEqual(saved['charged_attempts'],40);self.assertEqual(saved['training_flops'],4000)
        event=json.loads(self.events[0]);self.assertEqual(event['lost_charged_attempts'],2)
        self.assertEqual(event['lost_training_flops'],200)
        self.assertEqual(digest(self.cp),original)
        self.assertEqual(json.loads(self.recovery.read_text()),saved)
        self.assertEqual(json.loads(self.server.read_text()),saved)
        self.reconcile()
        self.assertEqual(len(self.events),1)  # A second setup never charges the same loss twice.

    def test_evidence_upload_failure_never_reduces_ledgers(self):
        self.cloud.put_immutable.side_effect=DriveBlocked('offline')
        with self.assertRaises(DriveBlocked):self.reconcile()
        self.cloud.upload.assert_not_called()
        self.assertEqual(json.loads((self.output/'budget_ledger.json').read_text()),self.ledger)

    def test_ledger_upload_failure_retries_identical_evidence_without_local_reset(self):
        original=self.cloud.upload.side_effect
        self.cloud.upload.side_effect=DriveBlocked('timeout')
        with self.assertRaises(DriveBlocked):self.reconcile()
        self.assertEqual(json.loads(self.recovery.read_text()),self.ledger)
        self.cloud.upload.side_effect=original
        self.reconcile()
        self.assertEqual(self.events[0],self.events[1])

    def test_timed_out_charge_that_committed_is_retained(self):
        atomic(self.server,dict(self.ledger,charged_attempts=43,training_flops=4300))
        self.reconcile()
        self.assertEqual(json.loads(self.events[0])['lost_charged_attempts'],3)

    def test_checkpoint_ahead_of_local_charge_blocks(self):
        torch.save(dict(self.state,charged_attempts=45,step=45),self.cp)
        with self.assertRaisesRegex(ValueError,'arithmetic'):self.reconcile()
        self.cloud.upload.assert_not_called()

    def test_discarded_updates_cannot_be_hidden(self):
        torch.save(dict(self.state,step=39),self.cp)
        with self.assertRaisesRegex(ValueError,'discarded'):self.reconcile()

    def test_wrong_configuration_blocks_recovery(self):
        torch.save(dict(self.state,configuration=dict(self.cfg,seed=20)),self.cp)
        with self.assertRaisesRegex(ValueError,'configuration'):self.reconcile()

    def test_completed_calibration_is_preserved_but_partial_calibration_blocks(self):
        ledger=dict(self.ledger,charged_attempts=93,training_flops=5000+43*120,
                    calibration_attempts=1,calibration_flops=300,calibration_sha256='cal')
        cp=dict(self.state,charged_attempts=90,step=90)
        report=dict(status='complete',config_sha256=ledger['config_sha256'],anchor_training_attempts=50,anchor_successful_updates=50)
        saved=checkpoint_ledger(cp,self.cfg,ledger,self.protocol,report,'cal')
        self.assertEqual(saved['calibration_flops'],300)
        self.assertEqual(saved['training_flops'],9800)
        with self.assertRaisesRegex(ValueError,'calibration'):
            checkpoint_ledger(cp,self.cfg,ledger,self.protocol,None,None)

    def test_bad_calibration_hash_blocks(self):
        ledger=dict(self.ledger,calibration_attempts=1,calibration_flops=300,calibration_sha256='correct')
        with self.assertRaisesRegex(ValueError,'calibration'):
            checkpoint_ledger(self.state,self.cfg,ledger,self.protocol,{'status':'complete'},'wrong')

    def test_repeated_setup_of_audited_import_before_first_new_charge(self):
        saved=dict(self.ledger,charged_attempts=40,training_flops=4000)
        atomic(self.output/'budget_ledger.json',saved)
        atomic(self.local/'recovery_events/task.json',dict(restored_checkpoint_charge=saved))
        self.group=[]
        self.reconcile()
        self.cloud.upload.assert_not_called();self.cloud.put_immutable.assert_not_called()

    def test_missing_server_ledger_without_import_receipt_blocks(self):
        self.group=[]
        with self.assertRaisesRegex(ValueError,'exactly one server charge ledger'):self.reconcile()


if __name__=='__main__':unittest.main()
