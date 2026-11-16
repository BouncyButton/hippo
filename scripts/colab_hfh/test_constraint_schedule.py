import json
from pathlib import Path
import unittest

from scripts.colab_hfh.constraint_schedule import make_payload, training_flops
from scripts.colab_hfh.constraint_worker import worker_source
from scripts.colab_hfh.constraint_campaign import campaign_source
from scripts.colab_hfh.constraint_search import recipes

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/hfh_colab_exploration_20261111'


class ConstraintScheduleTests(unittest.TestCase):
    def test_every_candidate_fits_both_independent_ramps(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        plan=json.loads((EXPORT/'constraint_matrix_plan.json').read_text())
        for model in plan['models']:
            for arm in plan['arms'][1:]:
                for spec in recipes(model,plan,arm):
                    p=make_payload(base,model,arm,plan['cost_references'][model]['canonical_costs'],**spec)['protocols'][model]['protocol.json']
                    s=p['models'][p['trainer_alias']]['objectives'][arm]
                    self.assertLessEqual(s['prefix_updates']+s['constraint_warmup_updates'],s['max_steps'])
                    if 'sdf' in arm:
                        self.assertLessEqual(s['sdf_prefix_updates']+s['sdf_warmup_updates'],s['max_steps'])

    def test_every_model_arm_fits_cap_without_changing_native_recipe(self):
        base = json.loads((EXPORT/'bands_v1_payload.json').read_text())
        plan = json.loads((EXPORT/'constraint_matrix_plan.json').read_text())
        defaults = dict(basic_unet16=(281,56), unetr_vit_b16=(50,50), nnunet=(22,9), swinunetr=(0,50))
        for model, (prefix,ramp) in defaults.items():
            costs = plan['cost_references'][model]['canonical_costs']
            # The measured early objectives can include non-reporting arms.
            costs = json.loads((ROOT/'reports/adni_constraints_5pct_fold0_seed0_20261013/profiles'/f'{model}.json').read_text())['costs']
            old = base['protocols'][model]['protocol.json']
            for arm in plan['arms'][1:]:
                payload = make_payload(base, model, arm, costs, prefix=prefix, ramp=ramp, target=.1, cap=.5, screening=False)
                p = payload['protocols'][model]['protocol.json']
                s = p['models'][p['trainer_alias']]['objectives'][arm]
                self.assertLessEqual(training_flops(s['max_steps'],s),p['budget_flops'])
                self.assertGreater(training_flops(s['max_steps']+1,s),p['budget_flops'])
                if 'sdf' in arm:
                    self.assertGreaterEqual(s['sdf_prefix_updates'],prefix+ramp)
                    self.assertLessEqual(s['sdf_prefix_updates']+ramp,s['max_steps'])
                for key in ('initial_lrs','lr_horizon_updates','training_loss','precision','native_geometry'):
                    self.assertEqual(p[key],old[key])
                self.assertNotIn('min_pair_delta_pp',payload)
                self.assertNotIn('band_reset_momentum',p)
        compile(worker_source(), 'constraint_worker', 'exec')
        compile(campaign_source(), 'constraint_campaign', 'exec')

    def test_complementary_candidates_keep_all_nine_identical_and_fit(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        plan=json.loads((EXPORT/'constraint_matrix_plan.json').read_text())
        costs=plan['cost_references']['nnunet']['canonical_costs']
        for arm in ('dice_bands_depth','dice_bands_sdf','dice_bands_edge_sdf_depth'):
            specs=recipes('nnunet',plan,arm)
            self.assertEqual(len(specs),13 if 'edge' in arm else (12 if arm=='dice_bands_depth' else 11))
            for spec in specs[4:6]:
                payload=make_payload(base,'nnunet',arm,costs,**spec)
                protocol=payload['protocols']['nnunet']['protocol.json']
                schedule=protocol['models']['nnunet']['objectives'][arm]
                self.assertNotIn('band_reset_momentum',protocol)
                self.assertEqual(set(payload['folds']),{0,1,2})
                self.assertEqual(set(payload['seeds']),{0,1,2})
                self.assertLessEqual(training_flops(schedule['max_steps'],schedule),protocol['budget_flops'])
                self.assertEqual(protocol['band_inner_weight'],spec['inner_weight'])
            for spec in specs[6:8]:
                payload=make_payload(base,'nnunet',arm,costs,**spec)
                protocol=payload['protocols']['nnunet']['protocol.json']
                self.assertEqual(protocol['constraint_component_targets'],spec['component_targets'])
                self.assertGreater(spec['component_targets']['bands'],spec['component_targets'].get('depth',spec['component_targets'].get('sdf')))

    def test_cost_selected_late_edge_candidates_retain_ninety_updates(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        plan=json.loads((EXPORT/'constraint_matrix_plan.json').read_text())
        for arm in plan['arms']:
            if 'edge' not in arm:continue
            for spec in (s for s in recipes('nnunet',plan,arm) if s['prefix']>=45):
                p=make_payload(base,'nnunet',arm,plan['cost_references']['nnunet']['canonical_costs'],**spec)['protocols']['nnunet']['protocol.json']
                self.assertEqual(p['models']['nnunet']['objectives'][arm]['max_steps'],90)
                self.assertEqual(p['lr_horizon_updates'],90)

    def test_preserved_band_strength_on_observed_training_gradients(self):
        import zipfile
        from scripts.colab_hfh.constraint_calibration import select_joint_weights
        for model in ('nnunet','swinunetr'):
            path=EXPORT/'constraints_analysis'/f'{model}__dice_bands_edge__r05'/f'{model}_f0_s1.zip'
            with zipfile.ZipFile(path) as archive:
                calibration=json.loads(archive.read('results/calibration.json'))
            weights,stats=select_joint_weights(calibration['cases'],'dice_bands_edge',
                targets={'bands':.3,'edge':.01},component_cap=.75,joint_target=.4,joint_cap=.75)
            self.assertAlmostEqual(stats['bands']['weighted_median_ratio'],.3,places=10)
            self.assertLessEqual(stats['joint']['worst_ramp_median_ratio'],.4)
            self.assertLessEqual(stats['joint']['worst_ramp_p95_ratio'],.75)

    def test_late_sdf_calibration_preserves_activation_and_optimization_cost(self):
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        plan=json.loads((EXPORT/'constraint_matrix_plan.json').read_text())
        for model in ('nnunet','swinunetr'):
            specs=recipes(model,plan,'dice_sdf')
            def schedule(spec):
                p=make_payload(base,model,'dice_sdf',plan['cost_references'][model]['canonical_costs'],**spec)['protocols'][model]['protocol.json']
                return p['models'][p['trainer_alias']]['objectives']['dice_sdf']
            before=schedule(specs[0])
            for spec in specs[4:6]:
                after=schedule(spec)
                self.assertEqual(after['sdf_prefix_updates'],before['sdf_prefix_updates'])
                self.assertEqual(after['sdf_warmup_updates'],before['sdf_warmup_updates'])
                self.assertEqual(after['max_steps'],before['max_steps'])
                self.assertGreater(after['prefix_updates'],before['prefix_updates'])
                self.assertEqual(training_flops(after['max_steps'],after),training_flops(before['max_steps'],before))

    def test_online_probe_is_charged_inside_cap_without_native_schedule_change(self):
        from scripts.colab_hfh.constraint_online_sdf import probe_allowance
        base=json.loads((EXPORT/'bands_v1_payload.json').read_text())
        plan=json.loads((EXPORT/'constraint_matrix_plan.json').read_text())
        costs=plan['cost_references']['swinunetr']['canonical_costs']
        specs=recipes('swinunetr',plan,'dice_sdf')
        for old,new in zip(specs[4:6],specs[6:8]):
            fixed=make_payload(base,'swinunetr','dice_sdf',costs,**old)['protocols']['swinunetr']['protocol.json']
            online=make_payload(base,'swinunetr','dice_sdf',costs,**new)['protocols']['swinunetr']['protocol.json']
            before=fixed['models'][fixed['trainer_alias']]['objectives']['dice_sdf']
            after=online['models'][online['trainer_alias']]['objectives']['dice_sdf']
            self.assertEqual(after['sdf_prefix_updates'],before['sdf_prefix_updates'])
            self.assertEqual(after['flops_per_update']-before['flops_per_update'],probe_allowance())
            self.assertEqual(online['lr_horizon_updates'],fixed['lr_horizon_updates'])
            self.assertLessEqual(training_flops(after['max_steps'],after),online['budget_flops'])
            self.assertGreater(training_flops(after['max_steps']+1,after),online['budget_flops'])


if __name__ == '__main__': unittest.main()
