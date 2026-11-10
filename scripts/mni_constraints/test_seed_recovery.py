"""Selection and scheduling checks without data, GPUs or cluster access."""
from pathlib import Path
import tempfile
import unittest
from scripts.cross_dataset_campaign.common import write
from scripts.mni_constraints.seed_recovery import tasks, selection, eligible


class SeedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stage = Path(self.temp.name)
        self.plan = dict(tasks=tasks(), selection_rule='mean >=50', selection_caveat='development')
        known=[]
        for m,seeds in [('nnunet',[0,2]),('basic_unet16',[0]),('swinunetr',[0,1,2]),('unetr_vit_b16',[0,1,2])]:
            known.extend(dict(model=m,seed=s,passes_mean50=True,passes_every_fold50=True) for s in seeds)
        write(self.stage/'known_seeds.json', known)

    def tearDown(self):
        self.temp.cleanup()

    def finish(self, model, seed, values):
        for fold,value in enumerate(values):
            folder=self.stage/'runs'/f'MNI_{model}_dice_f{fold}s{seed}'
            write(folder/'validation_latest.json',dict(summary=dict(macro_hard_dice=value)))
            write(folder/'cleanup.json',dict(status='complete'))

    def test_no_early_rejection_or_fold_replacement(self):
        self.finish('nnunet',3,[.01])
        state=selection(self.stage,self.plan)
        pending=[t for t in eligible(self.stage,self.plan,state) if t['model']=='nnunet']
        self.assertEqual([(t['seed'],t['fold']) for t in pending],[(3,1),(3,2)])
        self.finish('nnunet',3,[.01,.02,.03])
        state=selection(self.stage,self.plan)
        self.assertFalse(next(r for r in state['rows'] if r['model']=='nnunet' and r['seed']==3)['passes_mean50'])
        self.assertEqual({t['seed'] for t in eligible(self.stage,self.plan,state) if t['model']=='nnunet'},{4})

    def test_model_specific_stop_and_two_unet_replacements(self):
        self.finish('nnunet',3,[.6,.6,.6])
        self.finish('basic_unet16',3,[.6,.6,.6])
        state=selection(self.stage,self.plan)
        self.assertEqual(state['selected']['nnunet'],[0,2,3])
        pending=eligible(self.stage,self.plan,state)
        self.assertEqual({(t['model'],t['seed']) for t in pending},{('basic_unet16',4)})
        self.finish('basic_unet16',4,[.6,.6,.6])
        self.assertTrue(selection(self.stage,self.plan)['complete'])
        self.assertEqual(eligible(self.stage,self.plan,selection(self.stage,self.plan)),[])

    def test_historical_mean_threshold_reports_weak_fold(self):
        self.finish('nnunet',3,[.4,.55,.55])
        state=selection(self.stage,self.plan)
        row=next(r for r in state['rows'] if r['model']=='nnunet' and r['seed']==3)
        self.assertTrue(row['passes_mean50'])
        self.assertFalse(row['passes_every_fold50'])
        self.assertEqual(row['validation_by_fold_pct'],[40.,55.00000000000001,55.00000000000001])


if __name__=='__main__':unittest.main()
