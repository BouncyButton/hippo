import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

from scripts.colab_hfh.nnunet_search13_shared_policy import CAP,ARMS,SCHEDULE,candidates,make_protocol
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh.test_nnunet_search13 import PolicyTests,RuntimeTests


class SharedTests(unittest.TestCase):
    def test_all_13_combinations_get_more_exposure_without_more_flops(self):
        rows=candidates()
        self.assertEqual(len(rows),104)
        self.assertEqual(len({r['id'] for r in rows}),104)
        self.assertEqual({r['arm'] for r in rows},set(ARMS))
        self.assertEqual(CAP,31976450667090)
        for recipe in rows:
            arm=recipe['arm']
            ref=dict(active_objective=arm,models={'nnunet':{'objectives':{arm:dict(
                prefix_updates=60,sdf_prefix_updates=60,baseline_flops_per_update=355237172535,
                early_flops_per_update=355400000000,flops_per_update=355400000000,calibration_flops=1200000000000)}}})
            p=make_protocol(ref,recipe,Path('/content/search'))
            s=p['models']['nnunet']['objectives'][arm]
            self.assertEqual(p['lr_horizon_updates'],90)
            self.assertEqual(s['sdf_prefix_updates'],recipe['prefix'])
            self.assertLessEqual(training_flops(s['max_steps'],s),CAP)
            self.assertGreater(training_flops(s['max_steps']+1,s),CAP)
            self.assertEqual(training_flops(recipe['prefix'],s),355237172535*recipe['prefix'])
            self.assertGreaterEqual(s['max_steps']-recipe['prefix']-recipe['ramp']+1,15)
            if recipe['prefix']==20:self.assertGreaterEqual(s['max_steps']-20,65)

    def test_generated_sources(self):
        from scripts.colab_hfh.nnunet_search13_shared_worker import source
        for native in (True,False):compile(source(native),'shared_worker','exec')

    def test_shared_installer_uses_90_update_decay_and_earlier_anchors(self):
        from scripts.colab_hfh.nnunet_search13_shared_worker import install,install_native
        from scripts.colab_hfh.nnunet_warm_ready import READY_POLICY
        root=Path(__file__).resolve().parents[2]
        path=root/'scripts/cross_dataset_campaign/capped_core.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        for native in (True,False):
            ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),ns)
            trainer=SimpleNamespace(run=ns['run'],calibrate=lambda *a,**k:None,base=SimpleNamespace(),__file__=str(path))
            p=dict(seed0_plateau_schedule=SCHEDULE,lr_horizon_updates=90,augmentation='none',
                warm_ready_source_sha256={},readiness_policy=READY_POLICY,
                constraint_component_targets={'bands':.1},calibration_p95_cap=.5,
                constraint_joint_target=.1,constraint_joint_cap=.5)
            (install_native if native else install)(trainer,p,'dice' if native else 'dice_bands')
            self.assertIn('adherence',trainer.run.__code__.co_names)
            self.assertAlmostEqual(trainer.base.learning_rate('nnunet',75,90,.1),.07*.5**.9)
            self.assertGreater(trainer.base.learning_rate('nnunet',89,90,.1),0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SharedTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
    if '--runtime' in sys.argv:
        suite.addTest(RuntimeTests('test_real_loss_probe_and_other_native_heads_preserved'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
