"""Scientific split, selection, crop geometry and archive contracts."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.colab_hfh.policy import cells, decision
from scripts.colab_hfh.prepare import crop, EXPORT
from scripts.colab_hfh.runtime import read, write, digest, verify_archive


class SelectionTests(unittest.TestCase):
    def test_all_twelve_required_inclusive_boundary(self):
        completed = {cell:.5 for cell in cells('baselines')}
        self.assertEqual(decision('baselines', completed), ('accepted', None))
        completed.pop(cells('baselines')[-1])
        self.assertEqual(decision('baselines', completed)[0], 'pending')

    def test_any_model_or_fold_failure_rejects_immediately(self):
        for failed in cells('baselines'):
            completed = {}
            for cell in cells('baselines'):
                completed[cell] = .499999 if cell == failed else .8
                status, next_cell = decision('baselines', completed)
                if cell == failed:
                    self.assertEqual((status, next_cell), ('rejected', None))
                    break

    def test_corruption_is_not_bad_seed(self):
        for value in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                decision('baselines', {cells('baselines')[0]:value})

    def test_three_shared_seeds_and_finite_exhaustion(self):
        accepted, attempted = [], []
        for seed in range(32):
            completed = {}
            while decision('baselines', completed)[0] == 'pending':
                _, cell = decision('baselines', completed)
                attempted.append((seed, cell))
                completed[cell] = .4 if seed in (0, 2) else .6
            if decision('baselines', completed)[0] == 'accepted':accepted.append(seed)
            if len(accepted) == 3:break
        self.assertEqual(accepted, [1, 3, 4])
        self.assertEqual(sum(s == 0 for s, _ in attempted), 1)
        self.assertEqual(sum(s == 2 for s, _ in attempted), 1)
        self.assertEqual(len(attempted), 38)


class GeometryTests(unittest.TestCase):
    def test_thick_slices_are_not_interpolated(self):
        image = np.broadcast_to(np.arange(50)[None,:,None], (80,50,80)).copy()
        label = np.zeros(image.shape, dtype=np.uint8)
        label[35:44,21:29,35:44] = 1
        x, y, info = crop(image, label, np.array([1.,2.,1.]), 1)
        self.assertEqual(x.shape, (64,32,64))
        self.assertTrue(np.array_equal(x[32,:,32], np.arange(32)+info['source_origin_voxels'][1]))
        self.assertEqual(int(y.sum()), int((label == 1).sum()))

    def test_outside_field_is_rejected(self):
        image = np.ones((100,50,100), dtype=np.float32)
        label = np.ones_like(image, dtype=np.uint8)
        with self.assertRaises(ValueError):crop(image, label, np.array([1.,2.,1.]), 1)

    def test_real_frozen_data_and_protocols(self):
        bundle = EXPORT/'bundle'
        audit = read(bundle/'data_audit.json')
        self.assertEqual(len(audit['cases']), 50)
        by_case = {r['case']:r for r in audit['cases']}
        for row in audit['cases']:
            path = bundle/'data'/(row['case']+'.npz')
            self.assertEqual(digest(path), row['sha256'])
            with np.load(path) as data:
                self.assertEqual(data['image'].shape, (64,32,64))
                self.assertTrue(np.isfinite(data['image']).all())
                self.assertEqual(set(np.unique(data['label'])), {0,1})
        validation_people = []
        for fold, selection in enumerate(audit['selections']):
            self.assertEqual(len(selection['train']), 10)
            self.assertEqual(len(selection['val']), 10)
            self.assertFalse(set(selection['train_subjects']) & set(selection['val_subjects']))
            validation_people += selection['val_subjects']
            self.assertEqual(sum(by_case[c]['acquisition']=='fine' for c in selection['train']), 4)
            for seed in range(3):
                p = read(bundle/'medsam_protocols'/f'f{fold}_s{seed}.json')
                self.assertEqual(p['train_cases'], selection['train'])
                self.assertEqual(p['evaluation_cases'], selection['val'])
                self.assertEqual(p['updates'], 200)
        self.assertEqual(len(set(validation_people)), 15)
        for folder in (bundle/'protocols').iterdir():
            p = read(folder/'protocol.json'); model = p['models'][p['trainer_alias']]
            schedule = model['objectives']['dice']
            self.assertEqual(p['budget_flops'], 31976450667090)
            self.assertEqual(schedule['max_steps'], p['budget_flops']//schedule['flops_per_update'])
            self.assertEqual(set(model['objectives']), {'dice'})
            self.assertEqual(p['weight_ce'], int(folder.name=='nnunet'))

    def test_notebook_python_cells_parse(self):
        from scripts.colab_hfh import prepare
        with tempfile.TemporaryDirectory() as tmp, patch.object(prepare, 'EXPORT', Path(tmp)):
            bundle = Path(tmp)
            write(bundle/'manifest.json', {})
            package = bundle/'example.zip'
            package.write_bytes(b'fixture')
            for kind in ('baselines', 'medsam3'):
                prepare.notebook(bundle, package, kind)
            notebooks = sorted(bundle.glob('*.ipynb'))
            self.assertEqual(len(notebooks), 2)
            for path in notebooks:
                for cell in read(path)['cells']:
                    if cell['cell_type'] == 'code':
                        ast.parse(''.join(cell['source']))


class ArchiveTests(unittest.TestCase):
    def test_tampered_results_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write(folder/'result.json', dict(validation_dice=.8))
            write(folder/'archive_manifest.json', {'result.json':digest(folder/'result.json')})
            self.assertEqual(verify_archive(folder)['validation_dice'], .8)
            write(folder/'result.json', dict(validation_dice=.9))
            with self.assertRaises(ValueError):verify_archive(folder)


if __name__ == '__main__':
    unittest.main()
