"""Check common-compute milestones and completed-run tamper/budget guards."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import run_budget_baselines as campaign
from train_budget_baseline import validation_updates


class BudgetChecks(unittest.TestCase):
    def test_common_targets_round_down_without_exceeding_any_cap(self):
        budget = 94603780000000
        targets = [budget * k // 30 for k in range(1, 31)] + [13481038223811]
        for model, cost, expected in [('unet', 78836480841, 1200), ('swinunetr', 155012136022, 610), ('nnunet', 355293896301, 266)]:
            entry = {'flops_per_update': cost}
            if model == 'unet':
                entry = {'objectives': {'dice': entry}}
            protocol = {'budget_flops': budget, 'models': {model: entry}, 'validation_target_flops': targets}
            steps = validation_updates(protocol, model)
            self.assertEqual(steps[-1], expected)
            self.assertLessEqual(expected * cost, budget)
            self.assertGreater((expected + 1) * cost, budget)
            for target in targets:
                self.assertIn(target // cost, steps)
                self.assertLessEqual((target // cost) * cost, target)
            self.assertEqual(validation_updates(protocol, model, max_steps=1), [1])
            protocol['validation_target_flops'] = [budget + 1]
            with self.assertRaises(ValueError):
                validation_updates(protocol, model)

    def test_completed_baseline_rejects_extra_updates_and_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; directory = source / 'protocol'
            directory.mkdir(parents=True); (source / 'scripts').mkdir()
            output = root / 'swinunetr/fold0/seed0/run'; output.mkdir(parents=True)
            (source / 'scripts/train_compute_matched_baselines.py').write_text('frozen trainer')
            (source / 'scripts/train_budget_baseline.py').write_text('frozen adapter')
            protocol = {'budget_flops': 100, 'models': {'swinunetr': {'flops_per_update': 30}},
                        'validation_target_flops': [50, 100], 'pkl_sha256': 'data'}
            campaign.write_json(directory / 'protocol.json', protocol)
            split = {'train': ['train'], 'val': ['validation']}
            campaign.write_json(directory / 'splits_10patients.json', [split])
            config = {'model': 'swinunetr', 'fold': 0, 'seed': 0, 'smoke_test': False,
                      'protocol_sha256': campaign.digest(directory / 'protocol.json'),
                      'dataset_sha256': 'data', 'script_sha256': campaign.digest(source / 'scripts/train_compute_matched_baselines.py'),
                      'max_steps': 3, 'budget_flops': 100, 'train_cases': split['train'], 'validation_cases': split['val']}
            campaign.write_json(output / 'config.json', config)
            campaign.write_json(output / 'budget_ledger.json', {'charged_attempts': 3, 'charged_training_flops': 90, 'config_sha256': campaign.digest(output / 'config.json')})
            validation = {'cases': [{'case_name': 'validation'}], 'summary': {'macro_hard_dice': .7, 'anterior_hard_dice': .6, 'posterior_hard_dice': .8}}
            for name in ['validation_best.json', 'validation_latest.json']:
                campaign.write_json(output / name, validation)
            for name in ['checkpoint_best.pt', 'checkpoint_latest.pt']:
                (output / name).write_bytes(b'bound checkpoint')
            with (output / 'metrics.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['charged_attempts', 'macro_hard_dice']); writer.writeheader()
                writer.writerows([{'charged_attempts': 1, 'macro_hard_dice': .3}, {'charged_attempts': 3, 'macro_hard_dice': .7}])
            campaign.write_json(output / 'runtime_recipe.json', {'adapter_sha256': campaign.digest(source / 'scripts/train_budget_baseline.py'), 'protocol_sha256': config['protocol_sha256'], 'validation_updates': [1, 3]})
            done = {'status': 'complete', 'model': 'swinunetr', 'fold': 0, 'seed': 0,
                    'training_attempts_charged': 3, 'updates': 3, 'discarded_attempts': 0,
                    'consumed_training_flops': 90, 'unused_training_flops': 10, 'selected_step': 3, 'best_validation_hard_dice': .7}
            for key, name in [('config_sha256','config.json'),('metrics_sha256','metrics.csv'),('checkpoint_best_sha256','checkpoint_best.pt'),('checkpoint_latest_sha256','checkpoint_latest.pt'),('budget_ledger_sha256','budget_ledger.json'),('validation_best_sha256','validation_best.json')]:
                done[key] = campaign.digest(output / name)
            campaign.write_json(output / 'completion.json', done)
            self.assertEqual(campaign.verify_run(root, 'swinunetr', directory)['final_validation_macro_dice'], .7)
            done['training_attempts_charged'] = 4
            campaign.write_json(output / 'completion.json', done)
            with self.assertRaisesRegex(ValueError, 'budget mismatch'):
                campaign.verify_run(root, 'swinunetr', directory)
            done['training_attempts_charged'] = 3
            campaign.write_json(output / 'completion.json', done)
            (output / 'checkpoint_latest.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'Changed completed artifact'):
                campaign.verify_run(root, 'swinunetr', directory)


if __name__ == '__main__':
    unittest.main()
