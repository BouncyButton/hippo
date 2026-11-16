"""Scientific pairing and the full-matrix gate must fail closed."""
import ast
import copy
import unittest

from scripts.colab_hfh.medsam_constraints_policy import PAIRS, RECIPES, qualifies, matrix_tasks, make_protocol
from scripts.colab_hfh.medsam_constraints_worker import worker_source


class PolicyTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=s, fold=f, validation_dice=.8, delta_dice_pp=1.) for s, f in PAIRS]

    def test_every_original_pair_required(self):
        rows = self.rows()
        self.assertTrue(qualifies(rows))
        self.assertFalse(qualifies(rows[:-1]))
        rows[-1] = rows[0]
        self.assertFalse(qualifies(rows))

    def test_positive_mean_cannot_rescue_a_regression(self):
        rows = self.rows(); rows[0]['delta_dice_pp'] = -.01
        self.assertFalse(qualifies(rows))
        rows[0]['delta_dice_pp'] = 0
        self.assertFalse(qualifies(rows))

    def test_invalid_dice_and_nonfinite_gain_cannot_win(self):
        for key, value in [('validation_dice', .49), ('validation_dice', float('nan')),
                           ('delta_dice_pp', float('inf'))]:
            rows = self.rows(); rows[0][key] = value
            self.assertFalse(qualifies(rows))

    def test_matrix_has_all_remaining_pairs_and_no_repeats(self):
        tasks = matrix_tasks(RECIPES[0])
        keys = {(a, s, f) for r, a, s, f in tasks}
        self.assertEqual(len(keys), 108)
        self.assertEqual(len(tasks), 108)
        self.assertFalse(any(a in ('dice', 'dice_bands') for a, s, f in keys))
        for arm in {a for a, s, f in keys}:
            self.assertEqual({(s, f) for a, s, f in keys if a == arm}, set(PAIRS))

    def test_protocol_preserves_native_training_and_input(self):
        p = dict(arm='dice', updates=200, accumulation=2, learning_rate=5e-5,
                 final_learning_rate=5e-6, warmup_updates=20, augmentation='none',
                 seed=0, fold=0, train_cases=['a'], evaluation_cases=['b'], base_sha256='frozen')
        before = copy.deepcopy(p)
        schedule = [['a', i % 32] for i in range(400)]
        history = [dict(lr=5e-5/(i+1)) for i in range(200)]
        actual = make_protocol(p, schedule, history, RECIPES[2], 'dice_bands', 'sha')
        for key, value in p.items():
            if key != 'arm': self.assertEqual(actual[key], value)
        self.assertEqual(p, before)
        self.assertEqual(actual['frozen_sample_schedule'], schedule)
        self.assertEqual(actual['frozen_learning_rates'], [x['lr'] for x in history])
        with self.assertRaises(ValueError):
            make_protocol(p, schedule[:-1], history, RECIPES[0], 'dice_bands', 'sha')

    def test_arm_bridge_keeps_audited_training_and_evaluation(self):
        source = worker_source(); ast.parse(source)
        self.assertIn("'--arm',args.arm", source)
        self.assertIn("'arm':args.arm", source)
        self.assertIn('install_evaluation()', source)
        self.assertIn('writer.close(force=True)', source)
        self.assertIn('original.main()', source)
        self.assertNotIn("'--arm','dice'", source)

    def test_hfh_slice_spacing_is_preserved(self):
        import numpy as np
        from scripts.colab_hfh.medsam_constraints_geometry import physical_affine
        affine = np.eye(4); affine[:3, :3] = np.eye(3)[::-1]
        actual = physical_affine(affine)
        np.testing.assert_array_equal(np.linalg.norm(actual[:3, :3], axis=0), [2, 1, 1])
        np.testing.assert_array_equal(affine[:3, :3], np.eye(3)[::-1])
        with self.assertRaises(ValueError): physical_affine(actual)


if __name__ == '__main__': unittest.main()
