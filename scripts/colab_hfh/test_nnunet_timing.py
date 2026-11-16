"""Timing-only changes, stopping logic, and full-fold decision safeguards."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from scripts.colab_hfh.nnunet_timing_policy import CAP,set_prefix,candidate_prefixes,summarize_groups
from scripts.colab_hfh.nnunet_timing_campaign import advance
from scripts.colab_hfh.constraint_schedule import training_flops

PACKAGE=Path(os.environ.get('HFH_TIMING_PACKAGE',str(Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111/nnunet_timing_seed_search_a100_v1/package')))


class TimingTests(unittest.TestCase):
    def test_design_and_budgets(self):
        d=json.loads((PACKAGE/'timing_design.json').read_text())
        self.assertEqual(len(d['groups']),39);self.assertEqual(d['initial_positive_groups'],27);self.assertEqual(d['failing_groups'],12)
        for spec in d['specs'].values():
            p=json.loads((PACKAGE/'protocols'/spec['name']/'protocol.json').read_text());a=spec['arm'];s=p['models']['nnunet']['objectives'][a]
            self.assertLessEqual(training_flops(s['max_steps'],s),CAP);self.assertGreater(training_flops(s['max_steps']+1,s),CAP)
            if a=='dice':continue
            original=d['original_recipes'][a]
            self.assertEqual(p['search_recipe'],dict(original,prefix=spec['prefix']))
            self.assertEqual(p['constraint_warmup_updates'],original['ramp'])
            self.assertEqual(s['sdf_prefix_updates'],spec['prefix']);self.assertEqual(s['prefix_updates'],spec['prefix'])
            self.assertGreaterEqual(s['max_steps']-spec['prefix']-original['ramp']+1,15)
            self.assertEqual(set_prefix(p,spec['prefix']),p)

    def fixture(self):
        names=[f'dice_sdf__p40__f{f}_s0' for f in range(3)]
        g={'x':dict(arm='dice_sdf',seed=0,status='searching',remaining_prefixes=[40,45],attempts=[dict(prefix=30,results=[dict(native_gain_pp=-1) for _ in range(3)])])}
        specs={n:dict(name=n) for n in names+[f'dice_sdf__p45__f{f}_s0' for f in range(3)]}
        return g,specs,names

    def test_never_select_partial_or_enqueue_duplicate(self):
        g,specs,names=self.fixture();self.assertEqual(len(advance(g,[],specs)),3)
        self.assertEqual(advance(g,[],specs),[])
        partial=[dict(name=names[i],native_gain_pp=10.) for i in range(2)]
        self.assertEqual(advance(g,partial,specs),[]);self.assertEqual(g['x']['status'],'searching')
        rows=partial+[dict(name=names[2],native_gain_pp=-21.)]
        self.assertEqual(len(advance(g,rows,specs)),3);self.assertEqual(g['x']['status'],'searching')

    def test_positive_mean_stops_even_with_reported_negative_fold(self):
        g,specs,names=self.fixture();advance(g,[],specs)
        rows=[dict(name=n,native_gain_pp=x) for n,x in zip(names,[1.,1.,-.5])]
        self.assertEqual(advance(g,rows,specs),[]);self.assertEqual(g['x']['status'],'positive')
        s=summarize_groups(g)[0];self.assertEqual(s['mean_gain_pp'],.5);self.assertEqual(s['worst_fold_gain_pp'],-.5)

    def test_finite_exhaustion(self):
        g,specs,names=self.fixture();g['x']['remaining_prefixes']=[]
        self.assertEqual(advance(g,[],specs),[]);self.assertEqual(g['x']['status'],'exhausted_without_positive_mean')


if __name__=='__main__':unittest.main(verbosity=2)
