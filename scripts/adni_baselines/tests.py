"""Check subject isolation and binary metrics independently of training."""
import json
import unittest

import pandas as pd
import torch

from .core import ROOT, PreparedADNI, evaluate, select_split


class ADNIChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ROOT/'datasets/Dataset103_ADNI'
        cls.df = pd.read_pickle(root/'adni_hippocampus_full.pkl',compression='gzip')
        cls.folds = json.loads((root/'splits_final.json').read_text())

    def test_grouping_and_preprocessing(self):
        split, provenance = select_split(self.df,self.folds)
        self.assertEqual(split,select_split(self.df.sample(frac=1,random_state=7),self.folds)[0])
        self.assertEqual(len(provenance['train_subjects']),5)
        self.assertFalse(set(provenance['train_subjects']) & set(provenance['validation_subjects']))
        dataset = PreparedADNI(self.df,split['train']+split['val'])
        self.assertEqual(len(dataset),64)
        for item in dataset:
            self.assertEqual(tuple(item['image'].shape),(1,64,64,64))
            self.assertEqual(torch.unique(item['label']).tolist(),[0,1])
            self.assertTrue(torch.isfinite(item['image']).all())

    def test_cross_partition_subject_is_rejected(self):
        folds = json.loads(json.dumps(self.folds))
        folds[0]['val'].append(folds[0]['train'][0])
        with self.assertRaisesRegex(ValueError,'Subject leakage'):
            select_split(self.df,folds)

    def test_binary_metric_oracle(self):
        label = torch.zeros(1,4,4,4,dtype=torch.long)
        label[:,1:3,1:3,1:3] = 1
        record = dict(image=label.float(),label=label,case_name='test_L',subject_id='test')

        class Oracle(torch.nn.Module):
            def __init__(self,extra=False):
                super().__init__(); self.extra=extra

            def forward(self,x):
                fg=x.clone()
                if self.extra:
                    fg[:,:,0,0,0]=1
                return torch.cat((1-fg,fg),dim=1)*50

        perfect=evaluate(Oracle(),[record],torch.device('cpu'),False)
        extra=evaluate(Oracle(True),[record],torch.device('cpu'),False)
        self.assertEqual(perfect['summary']['hippocampus_hard_dice'],1.)
        self.assertAlmostEqual(extra['summary']['hippocampus_hard_dice'],16/17,places=6)
        self.assertNotIn('anterior_hard_dice',perfect['summary'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
