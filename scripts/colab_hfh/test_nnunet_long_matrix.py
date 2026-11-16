import json
from pathlib import Path
import unittest

from scripts.colab_hfh import nnunet_long_constraints as previous
from scripts.colab_hfh.nnunet_long_matrix import ARMS, NAME, pack_for, namespace
from scripts.colab_hfh.constraint_schedule import training_flops


class MatrixTests(unittest.TestCase):
    def test_complete_scope_and_matched_cap(self):
        root = Path(__file__).resolve().parents[2]
        report = root/'reports/hfh_colab_exploration_20261111'
        base = json.loads((report/'bands_v1_payload.json').read_text())
        plan = json.loads((report/'constraint_matrix_plan.json').read_text())
        costs = plan['cost_references']['nnunet']['canonical_costs']
        self.assertEqual(set(ARMS) | set(previous.ARMS) | {'dice'}, set(costs))
        self.assertFalse(set(ARMS) & set(previous.ARMS))
        for arm in ARMS:
            payload = pack_for(base, plan, arm)
            p = payload['protocols']['nnunet']['protocol.json']
            s = p['models']['nnunet']['objectives'][arm]
            self.assertEqual(p['name'], NAME+'_'+arm)
            self.assertEqual((payload['seeds'], payload['folds'], p['augmentation']), ([0],[0],'none'))
            self.assertEqual(p['matched_readiness_anchor'], previous.ANCHOR)
            self.assertEqual(s['prefix_updates'],60)
            self.assertEqual(s['sdf_prefix_updates'],60)
            self.assertIn(s['max_steps'], (179,180))
            self.assertLessEqual(training_flops(s['max_steps'],s), previous.CAP)
            self.assertGreater(training_flops(s['max_steps']+1,s), previous.CAP)
        self.assertEqual(previous.NAME, 'nnunet_long_constraints_v1')
        self.assertEqual(len(previous.ARMS),4)
        self.assertIn('complete_nine_remaining_arms_no_expansion',namespace()['run'].__code__.co_consts)


if __name__ == '__main__':unittest.main()
