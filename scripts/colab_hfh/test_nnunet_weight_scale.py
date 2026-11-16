"""Verify exact coefficient scaling, unchanged costs, and full-fold selection."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import unittest
from scripts.colab_hfh.nnunet_weight_scale_policy import MULTIPLIERS,CAP,scale_protocol,summarize
from scripts.colab_hfh.constraint_calibration import select_joint_weights
from scripts.colab_hfh.constraint_schedule import training_flops

PACKAGE=Path(os.environ.get('HFH_WEIGHT_PACKAGE',str(Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111/nnunet_weight_scale_a100_v1/package')))


class WeightTests(unittest.TestCase):
    def test_complete_grid_preserves_every_successful_timing_and_cap(self):
        d=json.loads((PACKAGE/'weight_design.json').read_text())
        self.assertEqual(len(d['runs']),351);self.assertEqual(len(d['groups']),39)
        self.assertEqual(len({r['name'] for r in d['runs']}),351)
        for spec in d['runs']:
            p=json.loads((PACKAGE/'protocols'/spec['name']/'protocol.json').read_text())
            original=json.loads((PACKAGE/'references'/spec['reference_name']/'protocol.json').read_text())
            self.assertEqual(p['models'],original['models']);self.assertEqual(p['activity_policy'],original['activity_policy'])
            self.assertEqual(p['prefix_anchor_updates'],d['groups'][spec['group']]['prefix'])
            self.assertEqual(p['constraint_warmup_updates'],original['constraint_warmup_updates'])
            s=p['models']['nnunet']['objectives'][spec['arm']]
            self.assertLessEqual(training_flops(s['max_steps'],s),CAP);self.assertGreater(training_flops(s['max_steps']+1,s),CAP)
        for key in d['groups']:
            self.assertEqual({(r['fold'],r['weight_multiplier']) for r in d['runs'] if r['group']==key},{(f,m) for f in range(3) for m in MULTIPLIERS})

    def test_all_archived_calibrations_scale_exactly(self):
        for directory in (PACKAGE/'references').iterdir():
            original=json.loads((directory/'protocol.json').read_text());cal=json.loads((directory/'calibration.json').read_text())
            for m in MULTIPLIERS:
                p=scale_protocol(original,m)
                weights,_=select_joint_weights(cal['cases'],p['active_objective'],targets=p['constraint_component_targets'],
                    component_cap=p['calibration_p95_cap'],joint_target=p['constraint_joint_target'],joint_cap=p['constraint_joint_cap'])
                for name,w in weights.items():self.assertTrue(math.isclose(w,m*cal['weights'][name],rel_tol=1e-12))

    def test_never_choose_partial_or_worse_strength(self):
        refs=[dict(native_gain_pp=1.,validation_dice=.75) for _ in range(3)]
        d=dict(groups={'g':dict(arm='dice_depth',seed=0,prefix=30,reference_results=refs)})
        partial=[dict(group='g',weight_multiplier=1.25,native_gain_pp=10.,validation_dice=.84)]
        self.assertEqual(summarize(partial,d)[0]['selected_multiplier'],1.)
        bad=[dict(group='g',weight_multiplier=1.5,native_gain_pp=-1.,validation_dice=.73) for _ in range(3)]
        self.assertEqual(summarize(bad,d)[0]['selected_multiplier'],1.)
        good=[dict(group='g',weight_multiplier=2.,native_gain_pp=x,validation_dice=.75+x/100) for x in (-1.,3.,4.)]
        self.assertEqual(summarize(good,d)[0]['selected_multiplier'],2.)
        tie=[dict(group='g',weight_multiplier=1.25,native_gain_pp=1.,validation_dice=.75) for _ in range(3)]
        self.assertEqual(summarize(tie,d)[0]['selected_multiplier'],1.)


if __name__=='__main__':unittest.main(verbosity=2)
