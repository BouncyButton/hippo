"""Check the timing intervention, reuse partition, and charged budget."""
import json
import os
from pathlib import Path
import unittest
from scripts.colab_hfh.build_nnunet_prefix30 import change_timing
from scripts.colab_hfh.constraint_schedule import training_flops

PACKAGE=Path(os.environ.get('HFH_PREFIX30_PACKAGE',str(Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111/nnunet_prefix30_shared_a100_v1/package')))


class TimingTests(unittest.TestCase):
    def test_partition_and_preserved_45(self):
        d=json.loads((PACKAGE/'confirmation_design.json').read_text())
        self.assertEqual(len(d['runs']),63);self.assertEqual(len(d['reused']),63)
        allrows=d['runs']+d['reused']
        self.assertEqual(len({r['name'] for r in allrows}),126)
        for f in range(3):
            for s in range(3):
                self.assertEqual(len([r for r in allrows if (r['fold'],r['seed'])==(f,s)]),14)
        self.assertEqual(d['selected_recipes']['dice_bands_sdf_depth']['prefix'],45)
        self.assertTrue(all(d['original_recipes'][r['arm']]['prefix']==20 for r in d['runs']))
        self.assertTrue(all(r['arm']=='dice' or d['original_recipes'][r['arm']]['prefix']>=30 for r in d['reused']))

    def test_only_timing_cost_and_metadata_change(self):
        d=json.loads((PACKAGE/'confirmation_design.json').read_text())
        for r in d['runs']:
            p=json.loads((PACKAGE/'protocols'/r['name']/'protocol.json').read_text())
            a=r['arm'];s=p['models']['nnunet']['objectives'][a]
            expected=dict(d['original_recipes'][a],prefix=30)
            self.assertEqual(p['search_recipe'],expected)
            self.assertEqual(s['prefix_updates'],30);self.assertEqual(s['sdf_prefix_updates'],30)
            self.assertEqual(s['constraint_warmup_updates'],20)
            self.assertEqual(p['constraint_component_targets'],{n:.05 for n in a.split('_')[1:]})
            self.assertLessEqual(training_flops(s['max_steps'],s),d['cap'])
            self.assertGreater(training_flops(s['max_steps']+1,s),d['cap'])
            self.assertGreaterEqual(s['max_steps']-30-s['constraint_warmup_updates']+1,15)
            # The intervention must never double-charge the existing activity probe.
            twice=change_timing(p)
            self.assertEqual(twice,p)


if __name__=='__main__':unittest.main(verbosity=2)
