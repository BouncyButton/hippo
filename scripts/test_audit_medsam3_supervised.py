import copy
import unittest

from scripts.audit_medsam3_supervised import check_initial_losses


class InitialLossAuditTests(unittest.TestCase):
    def setUp(self):
        self.arms = ['baseline', 'bands', 'bands_edge']
        self.traces = {a: [dict(native=144.15442276000977,
                               bands=2.5802260637283325,
                               edge=0.2803417891263962)] for a in self.arms}
        self.traces['baseline'][0]['native'] = 144.15442657470703

    def test_observed_float32_rounding_is_recorded(self):
        result = check_initial_losses(self.traces, self.arms)
        self.assertEqual(result['differences_from_baseline']['bands']['native'], -3.814697265625e-06)

    def test_material_mismatch_in_any_component_is_rejected(self):
        for component in ('native', 'bands', 'edge'):
            with self.subTest(component=component):
                traces = copy.deepcopy(self.traces)
                traces['bands'][0][component] += .001
                with self.assertRaises(AssertionError):
                    check_initial_losses(traces, self.arms)

    def test_nonfinite_values_are_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            traces = copy.deepcopy(self.traces)
            traces['bands'][0]['native'] = value
            with self.assertRaises(AssertionError):
                check_initial_losses(traces, self.arms)


if __name__ == '__main__':
    unittest.main()
