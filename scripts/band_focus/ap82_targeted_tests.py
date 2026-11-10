"""Checks for shared-tissue accounting, gated selection, and loss compatibility."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from . import ap82_targeted_matrix as matrix
from .ap82_shared_tissue import paired_counts,aggregate
from .ap82_targeted_resume import SOURCES,extend_protocol
from . import ap82_tests as original

SCOPE=dict(stage1=original.FIRST,bases=original.BASES,coefficients=original.COEFFICIENTS)


def evidence():
    return dict(rows=[dict(objective=e['id'],background=e['background'],cases_with_shared_tissue=52,
        shared_swap_delta=-10,mean_case_shared_error_rate_delta=-.001,
        delta_foreground_dice=.0001,delta_outer_errors=-10,delta_ap_interface_chamfer_mm=-.01,
        delta_ap_interface_hd95_mm=-.01) for e in original.FIRST])


class TargetedTests(unittest.TestCase):
    def test_shared_voxels_corrections_and_exclusions(self):
        y=np.array([1,1,2,2,0]);b=np.array([2,2,1,0,1]);c=np.array([0,1,2,1,2])
        r=paired_counts(y,b,c)
        self.assertEqual((r['corrected'],r['introduced'],r['lost_baseline_swaps'],r['gained_candidate_swaps']),(2,0,1,1))
        self.assertEqual(aggregate([r])['shared_swap_delta'],-2)
        r=paired_counts(y,b,b);self.assertEqual(r['corrected']+r['introduced'],0)
        r=paired_counts(y,b,np.zeros_like(c));self.assertEqual(r['corrected'],0)
        self.assertIsNone(aggregate([r])['mean_case_shared_error_rate_delta'])
        # Total swaps can fall solely because all difficult tissue disappeared.
        self.assertEqual(r['lost_baseline_swaps'],3)
        with self.assertRaises(AssertionError):paired_counts(y,b,np.full_like(c,3))

    def test_selection_requires_shared_cut_and_outer_benefit(self):
        v=evidence();candidate=deepcopy(v['rows'][0])
        self.assertTrue(matrix.eligible(candidate))
        for key,value in [('shared_swap_delta',0),('mean_case_shared_error_rate_delta',0),
                          ('delta_foreground_dice',-.000001),('delta_outer_errors',1),
                          ('delta_ap_interface_chamfer_mm',0),('delta_ap_interface_hd95_mm',.001),
                          ('cases_with_shared_tissue',51),('background','dice')]:
            self.assertFalse(matrix.eligible(dict(candidate,**{key:value})),key)
        for row in v['rows']:row['shared_swap_delta']=1
        self.assertEqual(matrix.targeted_entries(SCOPE,v),([],None))
        v=evidence();parent=next(e for e in original.FIRST if e['id']=='ap82_s1_winner_b25_g100')
        next(r for r in v['rows'] if r['objective']==parent['id'])['mean_case_shared_error_rate_delta']=-.01
        self.assertEqual(matrix.select_parent(original.FIRST,v),parent)

    def test_bounded_mirrored_changes_preserve_original_runs(self):
        v=evidence();name='ap82_s1_winner_b25_g100'
        next(r for r in v['rows'] if r['objective']==name)['mean_case_shared_error_rate_delta']=-.01
        entries,parent=matrix.targeted_entries(SCOPE,v)
        self.assertEqual((len(entries),parent),(20,name))
        before=deepcopy(original.FIRST)
        p=extend_protocol(dict(runs=original.FIRST),entries)
        self.assertEqual(p['runs'][:30],before);self.assertEqual(original.FIRST,before)
        self.assertEqual(len({original.matrix.scientific_key(e) for e in p['runs']}),50)
        for a,b in zip(entries[::2],entries[1::2]):
            self.assertEqual((a['background'],b['background']),('winner','dice'))
            self.assertEqual(a['ap82'],b['ap82'])
            self.assertEqual(a['base_coefficients'],SCOPE['coefficients']['winner'])
            self.assertEqual(a['base_objective'],SCOPE['bases']['winner'])
            self.assertEqual(len(a['ap82_changes']),1)
            self.assertFalse(a.get('training'));self.assertFalse(b.get('training'))
        self.assertEqual(entries,json.loads(json.dumps(entries)))

    def test_targeted_objectives_finite_calibrated_and_frozen(self):
        v=evidence();name='ap82_s1_winner_b25_g100'
        next(r for r in v['rows'] if r['objective']==name)['mean_case_shared_error_rate_delta']=-.01
        entries,_=matrix.targeted_entries(SCOPE,v)
        with patch.object(original,'FIRST',entries),patch.object(original,'SECOND',[]):
            check=original.APTests();check.setUp()
            check.test_all_74_values_gradients_budgets_and_frozen_backgrounds()

    def test_exact_resume_contract(self):
        check=original.APTests();check.setUp()
        check.test_exact_restart_with_shared_old_history_and_both_backgrounds()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TargetedTests))
    path=Path(__file__).parent
    args.output.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        targeted_max=20,source_hashes={n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in SOURCES}),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
