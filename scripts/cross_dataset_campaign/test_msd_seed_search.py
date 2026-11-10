import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cross_dataset_campaign import msd_seed_search as search
from scripts.cross_dataset_campaign.common import read, write, CAP


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stage = Path(self.tmp.name)
        self.values = {}
        mock = patch.object(search, 'score', side_effect=lambda stage, item: self.values[item['id']])
        mock.start()
        self.addCleanup(mock.stop)

    def finish(self, item, dice=.5, fingerprint=None):
        root = self.stage/'runs'/item['id']
        write(root/'cleanup.json', {'status': 'complete'})
        write(root/'initialization.json', {'model_state_sha256': fingerprint or item['model']})
        self.values[item['id']] = dice

    def test_two_workers_and_no_early_qualification(self):
        for wave in range(7):
            result = search.decision(self.stage, 10)
            self.assertEqual(result['outcome'], 'next')
            self.assertEqual(len(result['tasks']), 2)
            for item in result['tasks']:
                self.finish(item)
        result = search.decision(self.stage, 10)
        self.assertEqual(result['outcome'], 'qualified')
        self.assertEqual(len(result['results']), 12)
        self.assertEqual(result['minimum_dice'], .5)

    def test_failed_full_run_rejects_even_if_other_model_passes(self):
        for item in search.decision(self.stage, 10)['tasks']:
            self.finish(item)
        pair = search.decision(self.stage, 10)['tasks']
        self.finish(pair[0], .4999)
        self.finish(pair[1], .9)
        result = search.decision(self.stage, 10)
        self.assertEqual(result['outcome'], 'rejected')
        self.assertEqual(result['threshold'], .5)

    def test_screen_rejection_and_partial_wave(self):
        pair = search.decision(self.stage, 10)['tasks']
        self.finish(pair[0], .28)
        self.assertEqual(search.decision(self.stage, 10)['tasks'], [pair[1]])
        self.finish(pair[1], .389)
        self.assertEqual(search.decision(self.stage, 10)['outcome'], 'rejected')

    def test_initialization_mismatch_cannot_qualify(self):
        for item in search.tasks():
            if item['seed'] == 10:
                self.finish(item, fingerprint=item['model']+str(item['fold']))
        with self.assertRaisesRegex(ValueError, 'Initialization differs'):
            search.decision(self.stage, 10)

    def test_recipe_preserves_original_lr_horizon_and_full_budget(self):
        parent = search.ROOT/'reports'/search.PARENT/'frozen/protocols'
        for model in search.MODELS:
            original = read(parent/f'MSD_{model}_f0s7/protocol.json')
            full = search.recipe(original, search.task(10, model), 'hash')
            self.assertEqual(full['models'], original['models'])
            self.assertEqual(full['budget_flops'], CAP)
            screen = search.recipe(original, search.task(10, model, phase='screen'), 'hash')
            self.assertEqual(screen['budget_flops'], CAP//2)
            for key in original:
                if 'lr' in key:
                    self.assertEqual(screen[key], original[key])
        self.assertEqual(len(search.tasks()), 448)
        self.assertEqual({t['seed'] for t in search.tasks()}, set(range(10, 42)))


if __name__ == '__main__':
    unittest.main()
