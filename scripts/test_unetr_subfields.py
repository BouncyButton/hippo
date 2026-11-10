"""Check subject separation, label preservation, resume and verified cleanup."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.unetr_subfields import core
from scripts.unetr_subfields.contracts import ROOT, NAME, DATASETS, CAP, subject, baseline_schedule
from scripts.unetr_subfields.prepare import select_splits
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign import storage


class SubfieldChecks(unittest.TestCase):
    def test_splits_are_subject_disjoint_and_seed_independent(self):
        for name, spec in DATASETS.items():
            original = read(ROOT / 'datasets' / spec['folder'] / 'splits_final.json')
            splits, provenance = select_splits(original, spec)
            self.assertEqual(splits, select_splits(original, spec)[0])
            for split in splits:
                self.assertFalse({subject(c) for c in split['train']} & {subject(c) for c in split['val']})
                self.assertEqual(len(split['train']), spec['train_subjects'] * 2)
            if name == 'COBRA':
                self.assertEqual([p['validation_subjects'] for p in provenance],
                                 [['hippocampus_cobra_2'], ['hippocampus_cobra_5'], ['hippocampus_cobra_3']])
            damaged = deepcopy(original)
            damaged[0]['train'].append(damaged[0]['val'][0])
            with self.assertRaises(ValueError):
                select_splits(damaged, spec)

    def test_budget_never_allows_an_extra_update(self):
        for name, spec in DATASETS.items():
            profile = read(ROOT / 'reports' / NAME / 'profiles' / f'{name}.json')
            cost = profile['update_flops']
            schedule = baseline_schedule(cost, spec['train_subjects'] * 2)
            self.assertLessEqual(schedule['max_steps'] * cost, CAP)
            self.assertGreater((schedule['max_steps'] + 1) * cost, CAP)

    def test_preprocessing_preserves_edge_labels(self):
        for name, spec in DATASETS.items():
            image = np.ones(spec['native_shape'], np.float32)
            label = np.zeros(spec['native_shape'], np.uint8)
            label[0, 0, 0] = 1
            label[-1, -1, -1] = len(spec['classes']) - 1
            frame = pd.DataFrame([dict(subject_id='case_L', participant_id='case', image_data=image, label_data=label)])
            item = core.PreparedSubfields(frame, ['case_L'], spec)[0]
            self.assertEqual(tuple(item['image'].shape), (1, *spec['spatial_size']))
            self.assertEqual(int((item['label'] > 0).sum()), 2)

    def test_resume_matches_uninterrupted_and_archive_precedes_cleanup(self):
        torch.set_num_threads(2)
        stage = ROOT / 'experiments' / NAME
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory(prefix='subfields_test_', dir=ROOT / 'reports') as directory:
                root = Path(directory)
                folder = root / 'protocol'
                shutil.copytree(stage / 'protocols' / dataset, folder)
                protocol = read(folder / 'protocol.json')
                original = read(folder / 'splits_10patients.json')[0]
                split = dict(train=original['train'][:1], val=original['val'][:1])
                write(folder / 'splits_10patients.json', [split])
                classes = len(protocol['classes'])
                tiny = lambda *_: torch.nn.Conv3d(1, classes, kernel_size=1)
                protocol['models']['unet']['parameters'] = sum(p.numel() for p in tiny().parameters())
                schedule = protocol['models']['unet']['objectives']['dice']
                # Explicit test fixture, not a production FLOP measurement.
                for key in ['baseline_flops_per_update', 'early_flops_per_update', 'flops_per_update']:
                    schedule[key] = 17
                schedule['max_steps'] = 2
                protocol.update(budget_flops=34, validation_target_flops=[17, 34], train_count=1,
                                splits_sha256=digest(folder / 'splits_10patients.json'), folds=[0], seeds=[0])
                write(folder / 'protocol.json', protocol)
                output = root / 'interrupted'
                args = argparse.Namespace(output=output, protocol_dir=folder, model='unet', objective='dice',
                    fold=0, seed=0, pkl=stage / 'data' / f'{dataset}.pkl', smoke_steps=0,
                    device='cpu', threads=2, resume=True)
                with patch.object(core, 'build_model', tiny):
                    core.bind(protocol)
                    with patch.object(core.trainer, 'stopping', side_effect=[False, True]):
                        with self.assertRaises(SystemExit) as stopped:
                            core.trainer.run(args)
                        self.assertEqual(stopped.exception.code, 75)
                    self.assertFalse((output / 'completion.json').exists())
                    with patch.object(core.trainer, 'stopping', return_value=False):
                        core.trainer.run(args)
                    resumed = torch.load(output / 'checkpoint_latest.pt', map_location='cpu')['model']
                    direct_args = deepcopy(args)
                    direct_args.output = root / 'direct'
                    with patch.object(core.trainer, 'stopping', return_value=False):
                        core.trainer.run(direct_args)
                    direct = torch.load(direct_args.output / 'checkpoint_latest.pt', map_location='cpu')['model']
                    for name in direct:
                        self.assertTrue(torch.equal(direct[name], resumed[name]), name)
                done = read(output / 'completion.json')
                self.assertEqual(done['updates'], 2)
                self.assertEqual(done['consumed_training_flops'], 34)
                self.assertEqual(done['discarded_attempts'], 0)
                task = dict(id='isolated_test', kind='conventional', dataset=dataset)
                manifest = storage.prepare(output, task)
                with self.assertRaises(FileNotFoundError):
                    storage.retire(output)
                self.assertTrue((output / 'checkpoint_latest.pt').exists())
                archive = root / 'durable_archive'
                archive.mkdir()
                for name in manifest['metadata']:
                    shutil.copyfile(output / name, archive / name)
                storage.verify_archive(archive, manifest)
                write(output / 'archive_acknowledgment.json', dict(status='verified',
                    manifest_sha256=digest(output / 'archive_manifest.json'), source_verified=True,
                    durable_archive=str(archive)))
                storage.retire(output)
                self.assertFalse((output / 'checkpoint_latest.pt').exists())
                self.assertFalse((output / 'checkpoint_best.pt').exists())
                self.assertTrue((archive / 'training_final.json').exists())
                self.assertTrue((stage / 'data' / f'{dataset}.pkl').exists())


if __name__ == '__main__':
    unittest.main()
