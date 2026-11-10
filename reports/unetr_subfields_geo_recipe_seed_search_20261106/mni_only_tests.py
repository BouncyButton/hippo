import unittest
from scripts.unetr_subfields.seed_search import selection, disabled_tasks, tasks


def rows(dataset, seed, scores):
    return [dict(dataset=dataset, seed=seed, fold=fold, final_dice=score)
            for fold, score in enumerate(scores)]


class SeedSelectionTests(unittest.TestCase):
    def test_requires_full_three_fold_mean(self):
        result = selection(rows('MNI', 0, [.7, .7]))
        self.assertEqual(result['groups']['MNI']['qualifying_seeds'], [])
        result = selection(rows('MNI', 0, [.5, .4, .45]))
        self.assertEqual(result['groups']['MNI']['qualifying_seeds'], [0])

    def test_bad_first_fold_does_not_reject_recoverable_mean(self):
        result = selection(rows('COBRA', 0, [.11864]))
        self.assertFalse(result['groups']['COBRA']['attempted_seeds'][0]['rejected_by_upper_bound'])
        result = selection(rows('COBRA', 0, [.1, .2]))
        disabled = disabled_tasks({'tasks': tasks()}, result)
        self.assertIn('COBRA_unetr_vit_b16_dice_f2s0', disabled)
        self.assertNotIn('MNI_unetr_vit_b16_dice_f2s0', disabled)

    def test_stops_each_dataset_independently(self):
        observed = sum([rows('MNI', seed, [.6, .5, .4]) for seed in [0, 2, 4]], [])
        result = selection(observed)
        self.assertEqual(result['groups']['MNI']['qualifying_seeds'], [0, 2, 4])
        self.assertFalse(result['target_met'])
        disabled = disabled_tasks({'tasks': tasks()}, result)
        self.assertIn('MNI_unetr_vit_b16_dice_f0s5', disabled)
        self.assertNotIn('COBRA_unetr_vit_b16_dice_f0s5', disabled)
        result = selection(observed + sum([rows('COBRA', seed, [.5, .5, .5]) for seed in [3, 5, 6]], []))
        self.assertTrue(result['target_met'])

    def test_deferred_dataset_stays_disabled_and_does_not_block_completion(self):
        observed = sum([rows('MNI', seed, [.6, .5, .4]) for seed in [0, 1, 2]], [])
        result = selection(observed, active_datasets=['MNI'])
        self.assertTrue(result['target_met'])
        self.assertEqual(result['deferred_datasets'], ['COBRA'])
        disabled = disabled_tasks({'tasks': tasks()}, result)
        self.assertEqual(set(disabled), {task['id'] for task in tasks()})
        partial = selection(rows('MNI', 0, [.6, .5, .4]), active_datasets=['MNI'])
        disabled = disabled_tasks({'tasks': tasks()}, partial)
        self.assertTrue(all(task['id'] in disabled for task in tasks() if task['dataset'] == 'COBRA'))
        self.assertNotIn('MNI_unetr_vit_b16_dice_f0s1', disabled)
        self.assertFalse(partial['target_met'])

    def test_scope_survives_repeated_summary_updates(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.unetr_subfields import seed_search
        from scripts.cross_dataset_campaign.common import read, write
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            write(stage / 'control.json', dict(pause=False, active_datasets=['MNI'], disabled_tasks=[]))
            def original(stage, plan, status, active):
                write(stage / 'status.json', dict(runs=rows('MNI', 0, [.6, .5, .4])))
            with patch.object(seed_search, 'ORIGINAL_SUMMARIZE', original):
                for _ in range(2):
                    seed_search.summarize(stage, {'tasks': tasks()}, 'running')
                    control = read(stage / 'control.json')
                    self.assertEqual(control['active_datasets'], ['MNI'])
                    self.assertTrue(all(t['id'] in control['disabled_tasks'] for t in tasks() if t['dataset'] == 'COBRA'))
                    self.assertNotIn('MNI_unetr_vit_b16_dice_f0s1', control['disabled_tasks'])

    def test_duplicate_folds_rejected(self):
        with self.assertRaises(ValueError):
            selection(rows('MNI', 0, [.5]) * 2)

    def test_interrupted_run_is_first_and_task_ids_unique(self):
        plan = tasks()
        self.assertEqual(plan[0]['id'], 'COBRA_unetr_vit_b16_dice_f0s1')
        self.assertEqual(plan[0]['protocol'], 'COBRA_resume')
        self.assertEqual(len(plan), len({r['id'] for r in plan}))


if __name__ == '__main__':
    unittest.main()
