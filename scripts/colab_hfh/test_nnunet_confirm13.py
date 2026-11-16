"""Confirmation scope, matched-control and failure-detection checks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.colab_hfh.nnunet_confirm13_campaign import CAP,validate_result,summarize
from scripts.colab_hfh.constraint_schedule import training_flops

PACKAGE=Path(os.environ.get('HFH_CONFIRM_PACKAGE',str(Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111/nnunet_confirm13_shared_a100_v1/package')))


class ConfirmationTests(unittest.TestCase):
    def test_complete_frozen_matrix_and_cap(self):
        design=json.loads((PACKAGE/'confirmation_design.json').read_text())
        self.assertEqual(len(design['runs']),112)
        self.assertEqual({(r['fold'],r['seed']) for r in design['runs']},{(f,s) for f in range(3) for s in range(3)}-{(0,0)})
        self.assertEqual(sum(r['arm']=='dice' for r in design['runs']),8)
        self.assertEqual(len({r['name'] for r in design['runs']}),112)
        for row in design['runs']:
            p=json.loads((PACKAGE/'protocols'/row['name']/'protocol.json').read_text())
            self.assertEqual(p['folds'],[row['fold']]);self.assertEqual(p['seeds'],[row['seed']])
            s=p['models']['nnunet']['objectives'][row['arm']]
            self.assertLessEqual(training_flops(s['max_steps'],s),CAP)
            self.assertGreater(training_flops(s['max_steps']+1,s),CAP)
            if row['arm']!='dice':
                self.assertEqual(p['search_recipe'],design['selected_recipes'][row['arm']])
                self.assertIn(row['native_name'],p['a100_native_anchor'])
                self.assertFalse(p['confirmation']['retuning'])

    def fixture(self,root,variant):
        name=variant+'__f0_s0'
        with zipfile.ZipFile(PACKAGE/'selection_archives'/(variant+'.zip')) as z:
            for item in z.namelist():
                if item.startswith('results/'):
                    suffix=item.removeprefix('results/')
                    for where in ('run','archive'):
                        target=root/'runs'/name/where/suffix;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
                elif item.startswith('protocol/'):
                    target=root/'package/protocols'/name/item.removeprefix('protocol/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
        p=json.loads((root/'package/protocols'/name/'protocol.json').read_text())
        return dict(name=name,variant=variant,arm=p['active_objective'],fold=0,seed=0,native_name='native__f0_s0',updates=p['models']['nnunet']['objectives'][p['active_objective']]['max_steps'])

    def test_archive_validator_accepts_known_results_and_rejects_wrong_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);native=self.fixture(root,'native')
            candidate=self.fixture(root,'dice_bands_edge_sdf_depth__early')
            self.assertEqual(validate_result(root,root/'package',native)['updates'],90)
            row=validate_result(root,root/'package',candidate)
            self.assertTrue(row['matched_native_prefix_verified']);self.assertGreater(row['native_gain_pp'],0)
            with self.assertRaisesRegex(ValueError,'Baseline recipe mismatch'):
                validate_result(root,root/'package',{**candidate,'seed':1})
            anchor=root/'runs/native__f0_s0/archive/native_update30.json'
            a=json.loads(anchor.read_text());a['model_state_sha256']='0'*64;anchor.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,'Exact paired native prefix'):
                validate_result(root,root/'package',candidate)

    def test_over_cap_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);native=self.fixture(root,'native')
            path=root/'package/protocols'/native['name']/'protocol.json'
            p=json.loads(path.read_text());p['budget_flops']*=2;path.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'Budget or attempt'):
                validate_result(root,root/'package',native)

    def test_summary_keeps_selection_fold_separate(self):
        rows=[dict(arm='dice_bands',fold=f,seed=s,native_gain_pp=g,validation_dice=.7,all_included_validation_penalties_improved=True)
              for f,s,g in [(0,1,9.),(0,2,9.),(1,0,-1.),(1,1,-1.),(1,2,-1.),(2,0,-1.),(2,1,-1.),(2,2,-1.)]]
        result=summarize(rows,{'selected_recipes':{'dice_bands':{'id':'frozen'}}})[0]
        self.assertEqual(result['other_folds']['mean_gain_pp'],-1)
        self.assertEqual(result['same_fold_new_seeds']['mean_gain_pp'],9)
        self.assertEqual(result['other_folds']['n'],6)


if __name__=='__main__':unittest.main(verbosity=2)
