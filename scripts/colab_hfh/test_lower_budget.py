"""Matched-budget invariants and archive/control identity guards."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.colab_hfh.lower_budget_protocol import native_payload,archive_native
from scripts.colab_hfh.lower_budget_baseline_campaign import campaign_source
from scripts.colab_hfh.constraint_schedule import make_payload,training_flops
from scripts.colab_hfh.prepare_lower_budget import rescale_schedule

ROOT=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'bands_v1_payload.json').read_text())
        self.matrix=json.loads((ROOT/'constraint_matrix_plan.json').read_text())

    def test_native_truncates_compute_without_changing_the_trajectory(self):
        original=deepcopy(self.base);low=native_payload(self.base)
        self.assertEqual(self.base,original)
        p=low['protocols']['nnunet']['protocol.json'];old=self.base['protocols']['nnunet']['protocol.json']
        self.assertEqual(p['budget_flops'],23982338000317)
        schedule=p['models']['nnunet']['objectives']['dice']
        self.assertEqual(schedule['max_steps'],67)
        self.assertEqual(p['lr_horizon_updates'],90)
        for field in ('initial_lrs','training_loss','precision','native_geometry','augmentation','nnunet_sgd_momentum'):
            self.assertEqual(p[field],old[field])
        self.assertEqual(low['protocols']['nnunet']['splits_10patients.json'],self.base['protocols']['nnunet']['splits_10patients.json'])
        self.assertEqual(len(low['references']),9)
        self.assertEqual(set(low['folds']),{0,1,2});self.assertEqual(set(low['seeds']),{0,1,2})
        self.assertNotIn('min_pair_delta_pp',low)
        self.assertLessEqual(training_flops(67,schedule),p['budget_flops'])
        self.assertGreater(training_flops(68,schedule),p['budget_flops'])

    def test_transferred_constraints_keep_onset_and_fit_matched_cap(self):
        low=native_payload(self.base);costs=self.matrix['cost_references']['nnunet']['canonical_costs']
        specifications={'dice_bands':dict(prefix=22,ramp=9,target=.3,cap=.75,inner_weight=.9),
            'dice_bands_depth':dict(prefix=45,ramp=9,target=.4,cap=.75,inner_weight=.9,
                                   component_targets={'bands':.3,'depth':.01})}
        for arm,spec in specifications.items():
            payload=make_payload(low,'nnunet',arm,costs,**spec,screening=False)
            p=payload['protocols']['nnunet']['protocol.json'];s=p['models']['nnunet']['objectives'][arm]
            self.assertEqual(s['max_steps'],67)
            self.assertEqual(s['prefix_updates'],spec['prefix'])
            self.assertEqual(s['constraint_warmup_updates'],9)
            self.assertEqual(p['lr_horizon_updates'],90)
            self.assertLessEqual(training_flops(67,s),p['budget_flops'])
            self.assertGreater(training_flops(68,s),p['budget_flops'])
            self.assertNotIn('min_pair_delta_pp',payload)

    def test_lower_native_cannot_silently_change_lr_or_initialization(self):
        low=native_payload(self.base);p=low['protocols']['nnunet']['protocol.json']
        ref=self.base['references']['nnunet_f0_s0'];config=deepcopy(ref['configuration'])
        cost=p['models']['nnunet']['objectives']['dice']['flops_per_update']
        values={'protocol.json':p,'config.json':config,'initialization.json':ref['initialization'],
            'completion.json':dict(updates=67,consumed_training_flops=67*cost,calibration_flops=0),
            'validation_latest.json':{'summary':{'macro_hard_dice':.4}}}
        with patch('scripts.colab_hfh.lower_budget_protocol.read',side_effect=lambda path:values[path.name]),\
             patch('scripts.colab_hfh.lower_budget_protocol.write'),\
             patch('scripts.colab_hfh.runtime.archive'):
            result=archive_native(Path('/o'),Path('/d'),'nnunet',0,0,Path('/p'),ref)
            self.assertEqual(result['lower_budget_dice'],.4)  # Keep failures, too.
            config['initial_lr']=.001
            with self.assertRaisesRegex(ValueError,'trajectory differs'):
                archive_native(Path('/o'),Path('/d'),'nnunet',0,0,Path('/p'),ref)
            config['initial_lr']=ref['configuration']['initial_lr']
            values['initialization.json']={'model_state_sha256':'wrong'}
            with self.assertRaisesRegex(ValueError,'initialization differs'):
                archive_native(Path('/o'),Path('/d'),'nnunet',0,0,Path('/p'),ref)

    def test_template_uses_native_worker_and_separate_archiver(self):
        source=campaign_source();compile(source,'lower_budget_campaign','exec')
        self.assertIn("'scripts.colab_hfh.baseline'",source)
        self.assertIn('archive_native as archive',source)
        self.assertNotIn("'scripts.colab_hfh.bands_worker'",source)

    def test_half_budget_retains_native_recipe_and_all_nine_controls(self):
        low=native_payload(self.base,dict(version='nnunet_budget50_scaled_v1',ratio=[1,2]))
        p=low['protocols']['nnunet']['protocol.json']
        self.assertEqual(p['budget_flops'],15988225333545)
        self.assertEqual(p['models']['nnunet']['objectives']['dice']['max_steps'],45)
        self.assertEqual(p['lr_horizon_updates'],90)
        self.assertEqual(len(low['references']),9)
        original=self.base['protocols']['nnunet']['protocol.json']
        for key in ('initial_lrs','training_loss','precision','augmentation','nnunet_sgd_momentum'):
            self.assertEqual(p[key],original[key])
        self.assertEqual(native_payload(self.base),native_payload(self.base,dict(version='nnunet_budget75_v1',ratio=[3,4])))

    def test_half_budget_constraints_act_before_endpoint_with_same_strength(self):
        low=native_payload(self.base,dict(version='nnunet_budget50_scaled_v1',ratio=[1,2]))
        costs=self.matrix['cost_references']['nnunet']['canonical_costs']
        for arm,prefix in [('dice_bands',22),('dice_bands_depth',45)]:
            spec=dict(prefix=prefix,ramp=9,target=.3 if arm=='dice_bands' else .4,cap=.75,inner_weight=.9)
            if arm=='dice_bands_depth':spec['component_targets']={'bands':.3,'depth':.01}
            before=deepcopy(spec);scaled=rescale_schedule(spec,1,2)
            self.assertEqual(spec,before)
            self.assertEqual({k:v for k,v in scaled.items() if k not in ('prefix','ramp')},
                             {k:v for k,v in spec.items() if k not in ('prefix','ramp')})
            p=make_payload(low,'nnunet',arm,costs,**scaled,screening=False)['protocols']['nnunet']['protocol.json']
            schedule=p['models']['nnunet']['objectives'][arm]
            self.assertEqual(schedule['max_steps'],45)
            self.assertEqual(schedule['prefix_updates'],prefix//2)
            self.assertEqual(schedule['constraint_warmup_updates'],5)
            self.assertLess(schedule['prefix_updates']+5,45)
            self.assertLessEqual(training_flops(45,schedule),p['budget_flops'])
            self.assertGreater(training_flops(46,schedule),p['budget_flops'])

    def test_invalid_budget_fraction_rejected(self):
        for ratio in ([0,2],[2,2],[3,2],[1,0]):
            with self.assertRaises(ValueError):
                native_payload(self.base,dict(version='invalid',ratio=ratio))

    def test_user_selected_54_update_budget_and_constraint_timing(self):
        low=native_payload(self.base,dict(version='nnunet_budget60_scaled_v1',ratio=[3,5]))
        native=low['protocols']['nnunet']['protocol.json']
        self.assertEqual(native['budget_flops'],19185870400254)
        self.assertEqual(native['models']['nnunet']['objectives']['dice']['max_steps'],54)
        for arm,prefix,expected in [('dice_bands',22,13),('dice_bands_depth',45,27)]:
            spec=dict(prefix=prefix,ramp=9,target=.3 if arm=='dice_bands' else .4,cap=.75,inner_weight=.9)
            if arm=='dice_bands_depth':spec['component_targets']={'bands':.3,'depth':.01}
            scaled=rescale_schedule(spec,3,5)
            p=make_payload(low,'nnunet',arm,self.matrix['cost_references']['nnunet']['canonical_costs'],
                           **scaled,screening=False)['protocols']['nnunet']['protocol.json']
            schedule=p['models']['nnunet']['objectives'][arm]
            self.assertEqual((schedule['max_steps'],schedule['prefix_updates'],schedule['constraint_warmup_updates']),
                             (54,expected,6))
            self.assertEqual(p['lr_horizon_updates'],90)
            self.assertLessEqual(training_flops(54,schedule),p['budget_flops'])
            self.assertGreater(training_flops(55,schedule),p['budget_flops'])


if __name__=='__main__':unittest.main()
