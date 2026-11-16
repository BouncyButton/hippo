import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.colab_hfh import consistent_recipes as campaign


class ConsistentRecipeTests(unittest.TestCase):
    def run_campaign(self, score):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'bundle').mkdir()
            (root/'bundle/gpu_overlay.json').write_text('{"precision":"bf16"}')
            calls = []
            def worker(command, **kwargs):
                lr = float(command[command.index('--lr')+1])
                fold = int(command[command.index('--folds')+1])
                seed = int(command[command.index('--seeds')+1])
                calls.append((lr, fold, seed))
                value = score(lr, fold, seed)
                if isinstance(value, Exception):
                    raise value
                path = root/'recipe_diagnostics'/f'nnunet_lr{lr:g}_poly_cap'/'archives'/f'nnunet_f{fold}_s{seed}'/'result.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(model='nnunet',fold=fold,seed=seed,validation_dice=value)))
            with patch.object(campaign, 'RECIPES', {'nnunet':[0.1,0.05]}), \
                 patch.object(campaign.transport,'restore'), \
                 patch.object(campaign.transport,'publish_status'), \
                 patch.object(campaign.subprocess,'run',side_effect=worker), \
                 patch.object(campaign.sys,'argv',['campaign','--root',str(root)]):
                try:
                    campaign.main()
                except (subprocess.CalledProcessError, ValueError):
                    pass
            return json.loads((root/'consistent_recipes_status.json').read_text()), calls

    def test_threshold_requires_every_fixed_seed_and_fold(self):
        state, calls = self.run_campaign(lambda lr,f,s:.5)
        self.assertEqual(set((f,s) for _,f,s in calls), {(f,s) for f in range(3) for s in range(3)})
        self.assertEqual(state['selected_recipes']['nnunet']['cells'],9)

    def test_one_failed_cell_rejects_whole_recipe(self):
        state,calls=self.run_campaign(lambda lr,f,s:.499 if lr==.1 and (f,s)==(0,1) else .6)
        self.assertEqual([x for x in calls if x[0]==.1],[(.1,0,0),(.1,0,1)])
        self.assertEqual(state['selected_recipes']['nnunet']['initial_lr'],.05)
        self.assertEqual(len(state['attempts']),11)

    def test_runtime_failure_is_not_a_bad_seed(self):
        state,calls=self.run_campaign(lambda *args:subprocess.CalledProcessError(1,'worker'))
        self.assertEqual(state['status'],'runtime_error')
        self.assertEqual(state['attempts'],[])
        self.assertEqual(len(calls),1)

    def test_nan_cannot_qualify_a_recipe(self):
        state,calls=self.run_campaign(lambda *args:float('nan'))
        self.assertEqual(state['status'],'runtime_error')
        self.assertEqual(state['selected_recipes'],{})
        self.assertEqual(len(calls),1)


if __name__=='__main__':
    unittest.main()
