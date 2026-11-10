"""Check rejection, one-fold-at-a-time screening, and all-fold qualification."""
from pathlib import Path
import tempfile
import unittest
from scripts.cross_dataset_campaign.common import write
from scripts.mni_constraints.unet_fold_screen import tasks,selection,eligible


class FoldScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.stage=Path(self.tmp.name)
        write(self.stage/'known_seeds.json',[])
        self.plan=dict(tasks=tasks(),selection_rule='every final fold50',selection_caveat='development')

    def tearDown(self):self.tmp.cleanup()

    def finish(self,seed,fold,value,clean=True):
        d=self.stage/'runs'/f'MNI_basic_unet16_dice_f{fold}s{seed}'
        write(d/'completion.json',dict(status='complete'))
        write(d/'validation_latest.json',dict(summary=dict(macro_hard_dice=value)))
        if clean:write(d/'cleanup.json',dict(status='complete'))

    def pending(self):
        return [(r['seed'],r['fold']) for r in eligible(self.stage,self.plan,selection(self.stage,self.plan))]

    def test_restore_both_started_runs_then_screen_different_candidates(self):
        self.assertEqual(self.pending()[:4],[(11,1),(11,0),(12,1),(13,1)])
        self.finish(11,1,.39,clean=False)
        self.assertEqual(self.pending()[:2],[(12,1),(13,1)])
        self.assertEqual(selection(self.stage,self.plan)['decisions'][0]['status'],'rejected')

    def test_pass_advances_only_one_fold_and_failure_skips_rest(self):
        self.finish(11,1,.39)
        self.finish(12,1,.6)
        self.assertEqual(self.pending()[:2],[(12,0),(13,1)])
        self.finish(12,0,.49)
        self.assertNotIn((12,2),self.pending())
        self.assertFalse(selection(self.stage,self.plan)['complete'])

    def test_all_three_folds_required_and_completed_active_work_retained(self):
        self.finish(11,1,.4)
        self.finish(12,1,.6);self.finish(12,0,.6)
        self.assertFalse(selection(self.stage,self.plan)['complete'])
        self.assertEqual(self.pending()[0],(12,2))
        self.finish(12,2,.5)
        state=selection(self.stage,self.plan)
        self.assertTrue(state['complete']);self.assertEqual(self.pending(),[])
        self.assertEqual(state['selected']['basic_unet16'],[0,5,12])


if __name__=='__main__':unittest.main()
