import unittest
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from scripts.colab_hfh.prepare_bands import EXPORT
from scripts.colab_hfh.bands_search import candidate_payload, recipes, qualifies
from scripts.colab_hfh.bands_search import run
from scripts.colab_hfh.bands_campaign import failed_fold


class SearchTests(unittest.TestCase):
    def test_candidates_preserve_matched_controls_and_budget(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        for model in base['rates']:
            for target,prefix,ramp in recipes(model,base):
                p=candidate_payload(base,model,target,prefix,ramp)
                for key,reference in p['references'].items():
                    self.assertEqual(reference,base['references'][key])
                protocol=p['protocols'][model]['protocol.json']
                schedule=protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']
                control=base['protocols'][model]['protocol.json']
                self.assertEqual(protocol['initial_lrs'],control['initial_lrs'])
                self.assertEqual(protocol['lr_horizon_updates'],control['lr_horizon_updates'])
                self.assertEqual(schedule['max_steps'],control['models'][control['trainer_alias']]['objectives']['dice']['max_steps'])
                self.assertLessEqual(schedule['consumed_training_flops'],protocol['budget_flops'])
                self.assertEqual(schedule['consumed_total_flops'],schedule['consumed_training_flops']+schedule['calibration_flops'])

    def test_winner_requires_all_folds_and_all_nine_runs(self):
        rows=[dict(fold=f,seed=s,band_dice=.6,delta_dice_pp=1.) for f in range(3) for s in range(3)]
        self.assertTrue(qualifies(rows))
        self.assertFalse(qualifies(rows[:-1]))
        bad=deepcopy(rows)
        for r in bad:
            if r['fold']==2:r['delta_dice_pp']=-.1
        self.assertFalse(qualifies(bad))
        rows[0]['band_dice']=.49
        self.assertFalse(qualifies(rows))

    def test_fold_rejection_waits_for_all_three_fixed_seeds(self):
        rows=[dict(model='nnunet',fold=2,seed=s,band_dice=.6,delta_dice_pp=-1.) for s in range(3)]
        self.assertIsNone(failed_fold(rows[:2]))
        self.assertEqual(failed_fold(rows)['fold'],2)
        for row in rows:row['delta_dice_pp']=.1
        self.assertIsNone(failed_fold(rows))

    def test_fresh_runtime_restores_completed_search_without_training(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        state=dict(status='all_models_improved',attempts={},winners={})
        for model in base['rates']:
            rows=[dict(model=model,fold=f,seed=s,band_dice=.6,delta_dice_pp=1.,
                       baseline_archive_sha256=base['references'][f'{model}_f{f}_s{s}']['archive_sha256'])
                  for f in range(3) for s in range(3)]
            state['winners'][model]=dict(recipe='restored',results=rows)
        payload=json.dumps(state).encode()
        class FakeDrive:
            folder='remote:'
            def __init__(self,folder):pass
            def path(self,name):return name
            def upload(self,path):pass
            def call(self,*args):
                if args[0]=='lsjson':return json.dumps([dict(Name='search_status.json',Size=len(payload),Hashes={'MD5':hashlib.md5(payload).hexdigest()})])
                if args[0]=='copyto':Path(args[2]).write_bytes(payload);return ''
                raise AssertionError(args)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'bands_v1_payload.json').write_text(json.dumps(base))
            (root/'bands_v1').mkdir();(root/'bands_v1/campaign_status.json').write_text(json.dumps(dict(status='complete_all_36',results=[])))
            with patch.dict('sys.modules',{'bands_campaign_search':SimpleNamespace(Drive=FakeDrive)}),patch('scripts.colab_hfh.bands_search.subprocess.run') as launch:
                run(root);launch.assert_not_called()
            restored=json.loads((root/'bands_recipe_search/search_status.json').read_text())
            self.assertEqual(restored['status'],'all_models_improved')
            self.assertEqual(len(restored['winners']),4)


if __name__=='__main__':unittest.main()
