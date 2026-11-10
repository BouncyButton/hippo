"""Check ordered continuation and the stop after the third working UNet seed."""
from pathlib import Path
import tempfile
import unittest
from scripts.cross_dataset_campaign.common import write
from scripts.mni_constraints.unet_seed10_recovery import tasks, selection, eligible


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.stage=Path(self.temp.name)
        known=[]
        for model in ('nnunet','swinunetr','unetr_vit_b16','basic_unet16'):
            for seed in (0,2,5):
                known.append(dict(model=model,seed=seed,
                    passes_mean50=not(model=='basic_unet16' and seed==2),
                    passes_every_fold50=not(model=='basic_unet16' and seed in (2,5))))
        write(self.stage/'known_seeds.json',known)
        self.plan=dict(tasks=tasks(),selection_rule='mean50',selection_caveat='development')

    def tearDown(self):self.temp.cleanup()

    def finish(self,seed,values):
        for fold,value in enumerate(values):
            p=self.stage/'runs'/f'MNI_basic_unet16_dice_f{fold}s{seed}'
            write(p/'cleanup.json',dict(status='complete'))
            write(p/'validation_latest.json',dict(summary=dict(macro_hard_dice=value)))

    def pending(self):
        return [(t['seed'],t['fold']) for t in eligible(self.stage,self.plan,selection(self.stage,self.plan))]

    def test_seed10_first_no_early_rejection_then11(self):
        self.assertEqual(self.pending(),[(10,0),(10,1),(10,2)])
        self.finish(10,[.1]);self.assertEqual(self.pending(),[(10,1),(10,2)])
        self.finish(10,[.1,.2,.3]);self.assertEqual(self.pending(),[(11,0),(11,1),(11,2)])

    def test_mean_pass_stops_with_three_for_all_models(self):
        self.finish(10,[.45,.55,.6]);state=selection(self.stage,self.plan)
        self.assertTrue(state['complete']);self.assertEqual(self.pending(),[])
        self.assertEqual(set(state['selected']['basic_unet16']),{0,5,10})
        for m in ('nnunet','swinunetr','unetr_vit_b16'):
            self.assertEqual(set(state['selected'][m]),{0,2,5})
        row=next(r for r in state['rows'] if r['seed']==10)
        self.assertFalse(row['passes_every_fold50'])


if __name__=='__main__':unittest.main()
