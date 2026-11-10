"""Pilot gate and complete matched-pair confirmation checks."""
from pathlib import Path
import tempfile
import unittest
from scripts.cross_dataset_campaign.common import write
from scripts.mni_constraints.unetr_strength import tasks,selection,eligible,TARGETS


class StrengthSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.stage=Path(self.temp.name)
        self.plan=dict(tasks=tasks(),selection_rule='positive mean then confirm',selection_caveat='development')
        write(self.stage/'controls.json',[dict(arm=arm,seed=s,fold=f,validation_dice=.65)
              for arm in ('dice','dice_bands') for s in range(3) for f in range(3)])

    def tearDown(self):self.temp.cleanup()

    def finish(self,target,seed,gain):
        for t in self.plan['tasks']:
            if t['target']!=target or t['seed']!=seed:continue
            folder=self.stage/'runs'/t['id']
            write(folder/'cleanup.json',dict(status='complete'))
            write(folder/'validation_latest.json',dict(summary=dict(macro_hard_dice=.65+gain)))
            write(folder/'training_final.json',dict(summary=dict(macro_hard_dice=.70+gain)))
            write(folder/'calibration.json',dict(statistics={}))

    def test_all_pilots_required_then_six_confirmations(self):
        self.finish(.1,0,.01)
        state=selection(self.stage,self.plan)
        self.assertIsNone(state['selected']['target'])
        self.assertTrue(all(t['seed']==0 for t in eligible(self.stage,self.plan,state)))
        self.finish(.2,0,.02);self.finish(.4,0,-.01)
        state=selection(self.stage,self.plan)
        self.assertEqual(state['selected']['target'],.2)
        self.assertFalse(state['complete'])
        pending=eligible(self.stage,self.plan,state)
        self.assertEqual(len(pending),6)
        self.assertEqual({t['target'] for t in pending},{.2})
        self.finish(.2,1,.01);self.finish(.2,2,-.01)
        state=selection(self.stage,self.plan)
        self.assertTrue(state['complete'])
        self.assertFalse(next(r for r in state['summary'] if r['target']==.2)['consistent_nine_pair_improvement'])

    def test_no_improving_target_finishes_after_pilots(self):
        for target in TARGETS:self.finish(target,0,-.01)
        state=selection(self.stage,self.plan)
        self.assertTrue(state['complete'])
        self.assertEqual(eligible(self.stage,self.plan,state),[])


if __name__=='__main__':unittest.main()
