"""CPU checks for the MNI adaptation and its frozen Colab artifact."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.colab_mni_medsam3.data import select_support, Volumes
from scripts.colab_mni_medsam3.train import SupportSlabs
from scripts.cross_dataset_campaign.common import digest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/'reports/medsam3_mni_colab_pilot_20261108/bundle'


class PilotTests(unittest.TestCase):
    def test_nested_subject_support_no_leakage(self):
        folds = json.loads((BUNDLE/'saved_folds.json').read_text())
        previous = set()
        for count in (1, 2, 5, 10, 20):
            s = select_support(folds, count)
            current = set(s['train_cases'])
            self.assertTrue(previous <= current)
            self.assertEqual(len(current), 2*count)
            self.assertFalse(set(s['support_subjects']) & {c.rsplit('_',1)[0] for c in s['evaluation_cases']})
            self.assertEqual(len(s['evaluation_cases']), 10)
            previous = current
        for count in (0, 21):
            with self.assertRaises(ValueError): select_support(folds, count)

    def test_native_arrays_four_classes_and_schedule(self):
        p = json.loads((BUNDLE/'protocol.json').read_text())
        p.update(data=str(BUNDLE/'data'), arm='dice')
        volumes = Volumes(p)
        slabs = SupportSlabs(volumes, p, None)
        self.assertEqual(len(slabs.inventory), 2)
        for case in p['train_cases']:
            _, label = volumes.load(case)
            self.assertEqual(set(np.unique(label)), {0,1,2,3})
            self.assertEqual(set(slabs.supports[case]), {0,1,2,3})
            self.assertEqual(label.shape, (40,56,48))
            for c in range(4):
                np.testing.assert_array_equal(slabs.supports[case][c].foreground.numpy(), label.transpose(2,1,0)==c)
            self.assertIsNone(volumes.load(case, labels=False)[1])
        self.assertEqual(len(slabs.schedule(p)), 400)
        self.assertEqual(slabs.schedule(p), slabs.schedule(p))
        self.assertTrue({c for c,z in slabs.schedule(p)} <= set(p['train_cases']))

    def test_frozen_recipe_and_notebook(self):
        p = json.loads((BUNDLE/'protocol.json').read_text())
        parent = json.loads((ROOT/'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/frozen/parent_protocol.json').read_text())
        for name in ('updates','accumulation','warmup_updates','learning_rate','final_learning_rate',
                     'augmentation','base_sha256','lora_sha256'):
            self.assertEqual(p[name], parent[name])
        self.assertEqual((p['fold'],p['seed'],p['train_subjects']), (0,0,1))
        for name, sha in json.loads((BUNDLE/'manifest.json').read_text()).items():
            self.assertEqual(digest(BUNDLE/name), sha, name)
        n = json.loads((BUNDLE.parent/'MedSAM3_MNI_one_subject_pilot.ipynb').read_text())
        for c in n['cells']:
            if c['cell_type']=='code': ast.parse(c['source'])
        self.assertEqual(n['metadata']['colab']['gpuType'], 'A100')


if __name__ == '__main__': unittest.main()
