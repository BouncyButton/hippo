"""Selection, completion and recovery regression tests; no scientific GPU runs."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.colab_geofwi_seed_search import runtime as r


def passing(seed,score=.6):
    return [dict(**t,final_dice=score,initialization_sha256=t['model']) for t in r.tasks(seed)]


class SelectionTests(unittest.TestCase):
    def test_strict_threshold(self):
        for value in (.0,.499999,.5):
            rows=passing(0);rows[-1]['final_dice']=value
            self.assertEqual(r.decide(0,rows)['status'],'rejected')
        self.assertEqual(r.decide(0,passing(0,.50000000001))['status'],'qualified')

    def test_any_fold_any_model_rejects(self):
        for index in range(12):
            rows=passing(0);rows[index]['final_dice']=.49
            self.assertEqual(r.decide(0,rows)['status'],'rejected')

    def test_missing_cell_not_qualified(self):
        self.assertEqual(r.decide(0,passing(0)[:-1])['status'],'pending')

    def test_first_failure_skips_remaining(self):
        d=r.decide(0,[dict(**r.tasks(0)[0],final_dice=.1)])
        self.assertEqual(d['status'],'rejected');self.assertEqual(len(d['unrun_cells']),11)

    def test_seed_order_and_three_seed_completion(self):
        rows=passing(0)+[dict(**r.tasks(1)[0],final_dice=.2)]+passing(2)+passing(3)
        s=r.summary(rows)
        self.assertEqual(s['status'],'complete');self.assertEqual(s['qualified_seeds'],[0,2,3])
        self.assertEqual(r.summary(rows[:-1])['status'],'in_progress')

    def test_later_seeds_do_not_skip_pending_seed(self):
        self.assertEqual(r.summary(passing(1)+passing(2)+passing(3))['qualified_seeds'],[])

    def test_nonfinite_and_duplicate_are_errors(self):
        for value in (float('nan'),float('inf'),-.01,1.01):
            with self.assertRaises(ValueError):r.decide(0,passing(0,value))
        rows=passing(0)
        with self.assertRaises(ValueError):r.decide(0,rows+[rows[0]])

    def test_initialization_mismatch_is_error(self):
        rows=passing(0);rows[-1]['initialization_sha256']='changed'
        with self.assertRaises(ValueError):r.decide(0,rows)

    def test_complete_marker_requires_36_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for status in ('paused','technical_error'):
                r.publish(p,passing(0)+passing(1),status)
                self.assertFalse((p/'COMPLETE.json').exists())
            r.publish(p,passing(0)+passing(1)+passing(2))
            self.assertEqual(r.read(p/'COMPLETE.json')['cells'],36)


class RecoveryTests(unittest.TestCase):
    def test_verified_snapshot_and_lost_charge_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'local';out.mkdir();dur=p/'drive'
            (out/'checkpoint_latest.pt').write_bytes(b'checkpoint bytes')
            ledger=dict(charged_attempts=5,training_flops=500)
            r.write(out/'budget_ledger.json',ledger)
            r.write(out/'config.json',dict(seed=4,fold=2))
            writer=r.io.RecoveryWriter(out,dur,interval=0)
            writer.close(force=True)
            r.write(dur/'charged_ledger.json',ledger)
            r.io.restore_recovery(p/'restored',dur)
            self.assertEqual((p/'restored/checkpoint_latest.pt').read_bytes(),b'checkpoint bytes')
            r.write(dur/'charged_ledger.json',dict(charged_attempts=6,training_flops=600))
            with self.assertRaises(RuntimeError):r.io.restore_recovery(p/'unsafe',dur)
            self.assertTrue((dur/r.read(dur/'latest.json')['file']).exists())


class ResultAuditTests(unittest.TestCase):
    def test_reject_duplicate_volumes_and_wrong_dice_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'output';out.mkdir()
            t=r.task(8,0,'nnunet')
            split=dict(train=[f'train{i}' for i in range(19)],val=[f'val{i}' for i in range(96)])
            r.write(root/'splits.json',[split])
            r.write(out/'protocol.json',dict(trainer_alias='nnunet',models=dict(nnunet=dict(
                objectives=dict(dice=dict(max_steps=151,consumed_training_flops=123))))))
            r.write(out/'config.json',dict(model_identity='nnunet',smoke_test=False))
            r.write(out/'completion.json',dict(seed=8,fold=0,updates=151,discarded_attempts=0,
                consumed_training_flops=123,budget_flops=r.CAP,best_validation_hard_dice=.8,
                final_training_hard_dice=.75))
            r.write(out/'initialization.json',dict(model_state_sha256='fixed'))
            def report(names):
                return dict(summary=dict(macro_hard_dice=.75),cases=[dict(case_name=n,
                    macro_hard_dice=.75,confusion=[[96**3-500000,100000],[100000,300000]]) for n in names])
            validation=report(split['val'])
            r.write(out/'validation_latest.json',validation)
            r.write(out/'training_final.json',report(split['train']))
            self.assertEqual(r.validate_output(root,out,t)['final_dice'],.75)
            validation['cases'][-1]=validation['cases'][0]
            r.write(out/'validation_latest.json',validation)
            with self.assertRaises(ValueError):r.validate_output(root,out,t)
            validation=report(split['val']);validation['summary']['macro_hard_dice']=.9
            r.write(out/'validation_latest.json',validation)
            with self.assertRaises(ValueError):r.validate_output(root,out,t)


if __name__=='__main__':unittest.main()
