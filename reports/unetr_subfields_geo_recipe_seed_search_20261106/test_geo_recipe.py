"""Run against the frozen Dice+CE source, including its baseline-recognition patch."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import torch
from monai.losses import DiceCELoss
from scripts.unetr_subfields import core, geo_search, geo_recipe
from scripts.unetr_subfields.geo_recipe import training_loss
from scripts.cross_dataset_campaign import storage
from scripts.cross_dataset_campaign.common import read, write, digest

STAGE = Path(__file__).resolve().parents[2] / 'experiments' / geo_search.NAME


class DiceCEChecks(unittest.TestCase):
    def test_loss_and_gradients_match_equal_weight_dice_ce(self):
        torch.manual_seed(9)
        logits = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        labels = torch.randint(0, 4, (1, 1, 4, 4, 4))
        actual = training_loss(logits, labels, 'dice_ce', {}, .2)
        expected = DiceCELoss(**geo_recipe.LOSS_ARGUMENTS)(logits, labels)
        self.assertTrue(torch.allclose(actual, expected))
        g1 = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        g2 = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(torch.allclose(g1, g2, atol=1e-7))

    def test_actual_entrypoint_resume_and_objective_identity(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='hippo_dice_ce_test_') as directory:
            folder = Path(directory) / 'protocol'
            shutil.copytree(STAGE / 'protocols/MNI', folder)
            protocol = read(folder / 'protocol.json')
            split = read(folder / 'splits_10patients.json')[0]
            write(folder / 'splits_10patients.json', [dict(train=split['train'][:1], val=split['val'][:1])])
            tiny = lambda *_: torch.nn.Sequential(torch.nn.Conv3d(1, 4, kernel_size=1), torch.nn.Dropout(.1))
            protocol['models']['unet']['parameters'] = 8
            schedule = protocol['models']['unet']['objectives']['dice_ce']
            for key in ('baseline_flops_per_update', 'early_flops_per_update', 'flops_per_update'):
                schedule[key] = 17
            schedule['max_steps'] = 2
            protocol.update(budget_flops=34, validation_target_flops=[17, 34], train_count=1,
                folds=[0], seeds=[31], splits_sha256=digest(folder / 'splits_10patients.json'))
            write(folder / 'protocol.json', protocol)
            args = argparse.Namespace(output=Path(directory) / 'resumed', protocol_dir=folder,
                model='unet', objective='dice_ce', fold=0, seed=31, pkl=STAGE / 'data/MNI.pkl',
                smoke_steps=0, device='cpu', threads=2, resume=True)
            with patch.object(geo_recipe, 'build_model', tiny):
                with patch.object(core.trainer, 'stopping', side_effect=[False, True]):
                    with self.assertRaises(SystemExit) as stopped:
                        geo_search.train(args)
                    self.assertEqual(stopped.exception.code, 75)
                with patch.object(core.trainer, 'stopping', return_value=False):
                    geo_search.train(args)
                    direct = deepcopy(args); direct.output = Path(directory) / 'direct'
                    geo_search.train(direct)
            task = dict(kind='conventional', dataset='MNI', augmentation='xy_d4_v1')
            manifest = storage.prepare(args.output, task)
            with self.assertRaises(FileNotFoundError):
                storage.retire(args.output)
            archive = Path(directory) / 'archive'
            archive.mkdir()
            for name in manifest['metadata']:
                shutil.copyfile(args.output / name, archive / name)
            storage.verify_archive(archive, manifest)
            done = read(args.output / 'completion.json')
            self.assertEqual(done['objective'], 'dice_ce')
            self.assertEqual(done['updates'], 2)
            self.assertEqual(done['discarded_attempts'], 0)
            self.assertEqual(done['consumed_training_flops'], 34)
            self.assertEqual(read(args.output / 'calibration.json')['status'], 'not_required')
            self.assertEqual(read(args.output / 'runtime_recipe.json')['loss_arguments']['lambda_ce'], 1.)
            resumed = torch.load(args.output / 'checkpoint_latest.pt', map_location='cpu')['model']
            uninterrupted = torch.load(direct.output / 'checkpoint_latest.pt', map_location='cpu')['model']
            for key in resumed:
                self.assertTrue(torch.equal(resumed[key], uninterrupted[key]))
            write(args.output / 'archive_acknowledgment.json', dict(status='verified',
                manifest_sha256=digest(args.output / 'archive_manifest.json'), source_verified=True,
                durable_archive='/Users/test/durable_archive'))
            storage.retire(args.output)
            self.assertFalse((args.output / 'checkpoint_latest.pt').exists())
            self.assertEqual(read(args.output / 'cleanup.json')['status'], 'complete')

    def test_paired_d4_preserves_labels_and_rng(self):
        label = torch.arange(8*8*3).reshape(1,1,8,8,3) % 4
        image = label.float() + .25
        state = torch.get_rng_state().clone()
        for index in range(8):
            x,y = geo_recipe.apply_transform(image,label,index)
            self.assertTrue(torch.equal(x,y.float()+.25))
            self.assertTrue(torch.equal(torch.bincount(y.flatten()),torch.bincount(label.flatten())))
            self.assertEqual(x.shape,image.shape)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(geo_recipe.model_kwargs(4,(64,64,64))['dropout_rate'],.1)



if __name__ == '__main__':
    unittest.main()
