import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.colab_hfh.nnunet_seed0_long import make_pack, longer_archive
from scripts.colab_hfh.nnunet_seed0_long_worker import learning_rate, install, source
from scripts.colab_hfh.nnunet_seed0_balance_worker import plateau_lr
from scripts.colab_hfh.constraint_schedule import training_flops

class LongerPairTests(unittest.TestCase):
    def test_schedule_retains_initial_trajectory_and_refinement_window(self):
        for step in range(60):self.assertEqual(learning_rate(step),plateau_lr(step))
        for step in range(60,120):self.assertEqual(learning_rate(step),.07)
        self.assertGreater(learning_rate(150),0)
        self.assertEqual(learning_rate(180),0)

    def test_cap_pair_and_actual_trainer_integration(self):
        root=Path(__file__).resolve().parents[2]
        report=root/'reports/hfh_colab_exploration_20261111'
        base=json.loads((report/'bands_v1_payload.json').read_text())
        plan=json.loads((report/'constraint_matrix_plan.json').read_text())
        for arm,steps in [('dice',180),('dice_bands',179)]:
            p=make_pack(base,plan,arm,Path(__file__).parent)['protocols']['nnunet']['protocol.json']
            schedule=p['models']['nnunet']['objectives'][arm]
            self.assertEqual(p['budget_flops'],63952901334180)
            self.assertEqual(schedule['max_steps'],steps)
            self.assertLessEqual(training_flops(steps,schedule),p['budget_flops'])
            self.assertGreater(training_flops(steps+1,schedule),p['budget_flops'])
            self.assertEqual((p['seeds'],p['folds'],p['augmentation']),([0],[0],'none'))
        file=root/'scripts/cross_dataset_campaign/capped_core.py'
        tree=ast.parse(file.read_text());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
        ns={};exec(compile(ast.Module(body=[run],type_ignores=[]),str(file),'exec'),ns)
        trainer=SimpleNamespace(run=ns['run'],calibrate=lambda *a,**k:None,base=SimpleNamespace(),__file__=str(file))
        install(trainer,p,'dice_bands')
        self.assertIn('finish',trainer.run.__code__.co_names)
        self.assertTrue(any('training_update179.json' in str(getattr(c,'co_consts',())) for c in trainer.run.__code__.co_consts))
        self.assertEqual(trainer.base.learning_rate('nnunet',100,180,.1),.07)
        compile(source(),'long_worker','exec')
        self.assertTrue(callable(longer_archive()))

if __name__=='__main__':unittest.main()
