import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.colab_hfh.nnunet_long_constraints import ARMS, CAP, pack_for
from scripts.colab_hfh.nnunet_long_constraints_worker import install, source
from scripts.colab_hfh.constraint_schedule import training_flops


class LongConstraintTests(unittest.TestCase):
    def test_budget_and_real_trainer_integration(self):
        root = Path(__file__).resolve().parents[2]
        report = root/'reports/hfh_colab_exploration_20261111'
        base = json.loads((report/'bands_v1_payload.json').read_text())
        plan = json.loads((report/'constraint_matrix_plan.json').read_text())
        file = root/'scripts/cross_dataset_campaign/capped_core.py'
        tree = ast.parse(file.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        for arm in ARMS:
            p = pack_for(base, plan, arm)['protocols']['nnunet']['protocol.json']
            schedule = p['models']['nnunet']['objectives'][arm]
            self.assertEqual(schedule['max_steps'], 180)
            self.assertLessEqual(training_flops(180, schedule), CAP)
            self.assertGreater(training_flops(181, schedule), CAP)
            self.assertEqual(schedule['sdf_prefix_updates'], 60)
            self.assertEqual((p['seeds'], p['folds'], p['augmentation']), ([0], [0], 'none'))
            ns = {}; exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), 'exec'), ns)
            trainer = SimpleNamespace(run=ns['run'], calibrate=lambda *a, **k: None,
                                      base=SimpleNamespace(), __file__=str(file))
            install(trainer, p, arm)
            self.assertIn('matched_readiness_anchor', trainer.run.__code__.co_consts)
            self.assertEqual(trainer.base.validation_steps(180, 20)[-3:], [170, 179, 180])
            self.assertEqual(trainer.base.learning_rate('nnunet', 100, 180, .1), .07)
        compile(source(), 'constraint_worker', 'exec')


if __name__ == '__main__': unittest.main()
