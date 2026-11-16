"""Scientific invariants for the diagnosis-led nnU-Net follow-up bank."""
import json
from pathlib import Path
import unittest

from scripts.colab_hfh.constraint_schedule import make_payload
from scripts.colab_hfh.constraint_search import recipes, diagnosis_recipes


class DiagnosisRecipeTests(unittest.TestCase):
    def test_existing_candidates_are_immutable_and_new_trials_remain_matched(self):
        root = Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
        plan = json.loads((root/'constraint_matrix_plan.json').read_text())
        base = json.loads((root/'bands_v1_payload.json').read_text())
        previous = json.loads((root/'constraints_analysis/recipe_bank_before_nnunet_diagnosis.json').read_text())
        count = 0
        for key, old in previous.items():
            model, arm = key.split('__')
            current = recipes(model, plan, arm)
            self.assertEqual(current[:len(old)], old, key)
            if model != 'nnunet':
                self.assertEqual(current, old, key)
                self.assertEqual(diagnosis_recipes(model, plan, arm), [])
            for spec in current[len(old):]:
                count += 1
                payload = make_payload(base, model, arm,
                    plan['cost_references'][model]['canonical_costs'], **spec, screening=False)
                protocol = payload['protocols'][model]['protocol.json']
                native = base['protocols'][model]['protocol.json']
                for field in ('initial_lrs', 'lr_horizon_updates', 'training_loss', 'precision', 'native_geometry'):
                    self.assertEqual(protocol[field], native[field])
                schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
                self.assertEqual(schedule['max_steps'], 90)
                self.assertLessEqual(schedule['consumed_training_flops'], native['budget_flops'])
                # Outside declared constraint metadata/objectives, model and
                # optimizer configuration must remain identical to the control.
                for name, config in native['models'].items():
                    expected = {k:v for k,v in config.items() if k != 'objectives'}
                    actual = {k:v for k,v in protocol['models'][name].items() if k != 'objectives'}
                    self.assertEqual(actual, expected)
                self.assertEqual(payload['references'], {k:v for k,v in base['references'].items()
                                 if k.startswith(model+'_f')})
                self.assertNotIn('min_pair_delta_pp', payload)
        self.assertEqual(count, 16)

    def test_band_focus_appends_only_two_moderate_side_candidates(self):
        root=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
        plan=json.loads((root/'constraint_matrix_plan.json').read_text())
        previous=json.loads((root/'constraints_analysis/recipe_bank_before_band_focus.json').read_text())
        changed=[]
        for key,old in previous.items():
            model,arm=key.split('__');current=recipes(model,plan,arm)
            self.assertEqual(current[:len(old)],old)
            if len(current)>len(old):
                self.assertEqual(len(current),len(old)+1)
                self.assertEqual(current[-1]['inner_weight'],.75)
                changed.append(key)
        self.assertEqual(set(changed),{'nnunet__dice_bands','nnunet__dice_bands_depth'})


if __name__ == '__main__':
    unittest.main()
