"""Verify baseline-only search and transfer of exact compute allowances."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_unet48_budget_selection as pilot

REPO = Path(__file__).resolve().parents[1]


class SelectionChecks(unittest.TestCase):
    def setUp(self):
        self.plan = dict(initial_updates=400, min_updates=50, max_updates=3200,
                         max_pilot_runs=10, target_min=.70, target_max=.75)

    def test_target_uses_endpoint_inclusive(self):
        for score in (.70, .725, .75):
            self.assertIsNone(pilot.next_updates(self.plan, [dict(updates=400, final_dice=score)]))
        self.assertEqual(pilot.next_updates(self.plan, [dict(updates=400, final_dice=.65, best_dice=.73)]), 800)

    def test_bracketing_and_overshoot(self):
        self.assertEqual(pilot.next_updates(self.plan, []), 400)
        self.assertEqual(pilot.next_updates(self.plan, [dict(updates=400, final_dice=.80)]), 200)
        self.assertEqual(pilot.next_updates(self.plan, [dict(updates=400, final_dice=.65), dict(updates=800, final_dice=.80)]), 600)

    def test_invalid_or_exhausted_search(self):
        for rows in ([dict(updates=3200, final_dice=.60)],
                     [dict(updates=50, final_dice=.80)],
                     [dict(updates=400, final_dice=.80), dict(updates=800, final_dice=.65)]):
            with self.assertRaises(RuntimeError):
                pilot.next_updates(self.plan, rows)
        with self.assertRaises(ValueError):
            pilot.next_updates(self.plan, [dict(updates=400, final_dice=float('nan'))])

    def test_other_models_cannot_exceed_same_budget(self):
        for updates in (200, 400, 600, 800):
            budget = updates * 78849850205
            for cost in (155012136022, 355293896301):
                s = pilot.allowance(budget, cost)
                self.assertLessEqual(s['consumed_flops'], budget)
                self.assertGreater(s['consumed_flops'] + cost, budget)
                self.assertEqual(s['patient_pass_epochs'], s['max_updates'] / 10)

    def test_generated_protocol_collision_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, 'generated candidate'):
            pilot.validate_origin_manifest({'experiments/unet48_budget_candidates/candidate04/protocol.json': 'sha'})
        pilot.validate_origin_manifest({'scripts/train_unet48_budget_pilot.py': 'sha'})

    def test_failed_staging_exception_cannot_hide_source_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            name = 'experiments/unet48_budget_candidates/candidate04/protocol.json'
            path = source / name
            path.parent.mkdir(parents=True)
            pilot.write_json(path, {'selection_plan_sha256': 'old-plan'})
            actual = pilot.digest(path)
            code = source / 'trainer.py'; code.write_text('original')
            expected = {name: 'historical-generated-sha', 'trainer.py': pilot.digest(code)}
            prior = {'plan_sha256': 'old-plan', 'generated_protocol_exceptions': {name: actual}}
            pilot.verify_imported_source(source, expected, prior)
            code.write_text('changed')
            with self.assertRaises(AssertionError):
                pilot.verify_imported_source(source, expected, prior)
            with self.assertRaises(AssertionError):
                pilot.verify_imported_source(source, expected, {'plan_sha256': 'old-plan', 'generated_protocol_exceptions': {'trainer.py': pilot.digest(code)}})
            pilot.write_json(path, {'selection_plan_sha256': 'different-plan'})
            with self.assertRaises(AssertionError):
                pilot.verify_imported_source(source, {name: 'old'}, {'plan_sha256': 'old-plan', 'generated_protocol_exceptions': {name: pilot.digest(path)}})

    def test_import_binds_original_plan_history_and_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            directory = source / 'plan'
            directory.mkdir(parents=True)
            output = root / 'pilots/candidate00/run'
            output.mkdir(parents=True)
            costs = dict(unet_width48=78849850205)
            plan = dict(baseline_costs=costs, fold=0, seed=0, lr_policy='fixed')
            template = dict(recipe='same')
            pilot.write_json(directory / 'plan.json', plan)
            pilot.write_json(directory / 'template_protocol.json', template)
            pilot.write_json(directory / 'protocol.json', dict(selection_plan_sha256=pilot.digest(directory / 'plan.json')))
            pilot.write_json(output / 'completion.json', {})
            row = dict(updates=3200, final_dice=.45, output=str(output), protocol_dir=str(directory),
                       completion_sha256=pilot.digest(output / 'completion.json'))
            pilot.write_json(root / 'pilot_history.json', [row])
            pilot.write_json(root / 'runtime_origin_manifest.json', {'plan/plan.json': pilot.digest(directory / 'plan.json')})
            plan['imported_search'] = dict(root=str(root), plan_relative_path='plan/plan.json',
                                          plan_sha256=pilot.digest(directory / 'plan.json'),
                                          history_sha256=pilot.digest(root / 'pilot_history.json'),
                                          completion_sha256=[row['completion_sha256']])
            with patch.object(pilot, 'verify_completed', return_value=(.45, {})):
                rows = pilot.imported_history(plan, template)
                self.assertEqual(rows[0]['imported_root'], str(root))
                expanded = {**self.plan, 'max_updates': 12800}
                self.assertEqual(pilot.next_updates(expanded, rows), 6400)
                (root / 'pilot_history.json').write_text('[]')
                with self.assertRaises(AssertionError):
                    pilot.imported_history(plan, template)

    def test_controller_runs_only_dice_and_records_search_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; stage = root / 'plan'
            (source / 'scripts').mkdir(parents=True); stage.mkdir()
            fixture = REPO / 'experiments/unet48_budget_selection_20261005'
            for name in ('plan.json', 'template_protocol.json', 'splits_10patients.json', 'split_provenance.json', 'nnunet_architecture.json'):
                (stage / name).write_bytes((fixture / name).read_bytes())
            adapter = source / 'scripts/train_unet48_budget_pilot.py'; adapter.write_text('fixture adapter')
            data = root / 'data.pkl'; data.write_bytes(b'fixture data')
            template = json.loads((stage / 'template_protocol.json').read_text())
            template.update(pkl_sha256=pilot.digest(data), architecture_adapter_sha256=pilot.digest(adapter))
            pilot.write_json(stage / 'template_protocol.json', template)
            plan = json.loads((stage / 'plan.json').read_text())
            plan['template_protocol_sha256'] = pilot.digest(stage / 'template_protocol.json')
            pilot.write_json(stage / 'plan.json', plan)
            pilot.write_json(root / 'runtime_origin_manifest.json', {})
            calls = []
            def child(command, **kwargs):
                calls.append(command)
                output = Path(command[command.index('--output') + 1]); output.mkdir(parents=True)
                directory = Path(command[command.index('--protocol-dir') + 1])
                p = json.loads((directory / 'protocol.json').read_text())
                pilot.write_json(output / 'completion.json', {'best_validation_hard_dice': .80, 'consumed_total_flops': p['budget_flops']})
            scores = iter((.65, .73))
            def verify(output, *args):
                return next(scores), json.loads((output / 'completion.json').read_text())
            with patch.object(pilot.subprocess, 'run', child), patch.object(pilot, 'verify_completed', verify), contextlib.redirect_stdout(io.StringIO()):
                pilot.run(SimpleNamespace(root=root, plan=stage / 'plan.json', pkl=data, threads=1))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(c[c.index('--objective') + 1] == 'dice' for c in calls))
            selected = json.loads((root / 'budget_selection.json').read_text())
            self.assertEqual(selected['updates'], 800)
            self.assertEqual(selected['final_dice'], .73)
            self.assertEqual(selected['additional_search_flops'], 400 * 78849850205)
            self.assertEqual(selected['pilot_total_flops'], 1200 * 78849850205)
            self.assertEqual(set(selected['baseline_allowances']), {'unet_width48', 'swinunetr', 'nnunet'})


if __name__ == '__main__':
    unittest.main()
