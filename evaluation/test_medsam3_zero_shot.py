"""Checks of inference refinement safeguards, independent of model weights."""
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import json
import io
import contextlib
import os
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from evaluation.medsam3_zero_shot import refine, normalize_image
from scripts.medsam3_preflight import quota_free_bytes


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self):
        rng = np.random.default_rng(0)
        p = np.full((20, 20, 20), .1, dtype=np.float32)
        p[5:15, 5:15, 5:15] = .9
        image = np.clip(.15 + .65 * (p > .5) + rng.normal(0, .04, p.shape), 0, 1).astype(np.float32)
        return image, p

    def test_band_is_frozen_and_logit_change_is_bounded(self):
        image, p = self.sample()
        hard = p > .5
        struct = ndimage.generate_binary_structure(3, 1)
        band = ndimage.binary_dilation(hard, struct, iterations=2) & ~ndimage.binary_erosion(hard, struct, iterations=2)
        for edge in (False, True):
            out, info = refine(image, p, edge=edge, steps=5)
            np.testing.assert_array_equal(out[~band], p[~band])
            delta = np.log(out/(1-out)) - np.log(p/(1-p))
            self.assertLessEqual(np.abs(delta).max(), 2.00001)
            self.assertEqual(info['status'], 'complete')
            self.assertTrue(np.isfinite(out).all())

    def test_empty_prediction_stays_empty(self):
        image, p = self.sample()
        p[:] = 0
        out, info = refine(image, p, edge=True)
        np.testing.assert_array_equal(out, p)
        self.assertTrue(info['status'].startswith('no_op'))

    def test_inputs_not_mutated_and_repeatable(self):
        image, p = self.sample()
        original = p.copy()
        a, _ = refine(image, p, steps=3)
        b, _ = refine(image, p, steps=3)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(p, original)

    def test_reject_invalid_input(self):
        image, p = self.sample()
        p[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            refine(image, p)
        with self.assertRaises(ValueError):
            normalize_image(np.ones((3, 3, 3)))

    def test_quota_uses_tightest_applicable_limit(self):
        usage = '1009 1009 group pool_home 9.67TiB/∞ 38.82M/∞\n3160552 1395 user pool_home 88.78GiB/93.13GiB 177k/∞'
        self.assertAlmostEqual(quota_free_bytes(usage)/1024**3, 4.35)
        capped = usage.replace('9.67TiB/∞', '9.99GiB/10.00GiB')
        self.assertAlmostEqual(quota_free_bytes(capped)/1024**3, .01)
        with self.assertRaises(RuntimeError):
            quota_free_bytes('unexpected format')

    def test_full_artifact_flow_opens_reference_only_after_prediction(self):
        from evaluation import medsam3_zero_shot as runner
        image, p = self.sample()
        original_load = nib.load
        original_refine = runner.refine
        original_check_output = runner.subprocess.check_output
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path, label_path = root/'image.nii.gz', root/'label.nii.gz'
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            weights = root/'weights.pt'
            weights.write_bytes(b'synthetic test only')
            saved_predictions = []
            for index, truth in enumerate([(p > .5).astype(np.uint8), np.zeros(p.shape, dtype=np.uint8)]):
                nib.save(nib.Nifti1Image(truth, np.eye(4)), label_path)
                output = root/f'output{index}'
                def guarded_load(path, *args, **kwargs):
                    if Path(path) == label_path:
                        self.assertEqual(len(list(output.glob('*_mask.nii.gz'))), 3)
                        self.assertEqual(len(list(output.glob('*_probability.nii.gz'))), 3)
                    return original_load(path, *args, **kwargs)
                argv = ['runner', '--upstream', tmp, '--base-weights', str(weights),
                        '--lora-weights', str(weights), '--image', str(image_path),
                        '--label', str(label_path), '--output', str(output)]
                with mock.patch('sys.argv', argv), mock.patch.object(runner, 'load_model', return_value=(object(), object())), \
                     mock.patch.object(runner, 'infer_volume', return_value=(p.copy(), [])), \
                     mock.patch.object(runner, 'refine', side_effect=lambda im, prob, edge, device: original_refine(im, prob, edge=edge, steps=3)), \
                     mock.patch.object(nib, 'load', side_effect=guarded_load), \
                     mock.patch.object(runner.subprocess, 'check_output', side_effect=lambda command, **kw: 'test-revision\n' if command[0] == 'git' else original_check_output(command, **kw)), \
                     mock.patch.dict(os.environ, {'MPLCONFIGDIR': str(root/'matplotlib')}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                report = json.loads((output/'results.json').read_text())
                self.assertEqual(report['metrics']['medsam3']['dice'], 1. if index == 0 else 0.)
                self.assertTrue((output/'comparison.png').is_file())
                saved_predictions.append([original_load(path).get_fdata() for path in sorted(output.glob('*_probability.nii.gz'))])
            for a, b in zip(*saved_predictions):
                np.testing.assert_array_equal(a, b)

    def test_storage_falls_back_without_using_beegfs(self):
        from scripts import medsam3_cluster_job as worker
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(Path, 'is_dir', return_value=True), \
             mock.patch.object(os, 'access', return_value=True), \
             mock.patch.object(Path, 'stat', lambda p: SimpleNamespace(st_dev=2 if str(p) == '/beegfs' else 1)), \
             mock.patch.object(worker.shutil, 'disk_usage', side_effect=lambda p: SimpleNamespace(free=(2 if str(p) == '/tmp' else 10)*1024**3)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.local_storage(Path('/beegfs')), Path('/dev/shm'))
        with mock.patch.object(Path, 'is_dir', return_value=True), \
             mock.patch.object(os, 'access', return_value=True), \
             mock.patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=1)), \
             mock.patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*1024**3)):
            with self.assertRaises(RuntimeError):
                worker.local_storage(Path('/beegfs'))

    def test_base_loader_allows_only_disabled_interactive_extras(self):
        from evaluation.medsam3_zero_shot import load_detector_state
        model = torch.nn.Linear(2, 1)
        state = {'detector.'+k: torch.ones_like(v) for k, v in model.state_dict().items()}
        state['detector.backbone.vision_backbone.sam2_convs.0.weight'] = torch.ones(1)
        report = load_detector_state(model, state)
        self.assertEqual(report['loaded_detector_tensors'], 2)
        self.assertEqual(len(report['ignored_disabled_interactive_tensors']), 1)
        missing = dict(state); del missing['detector.bias']
        with self.assertRaises(RuntimeError):
            load_detector_state(model, missing)
        wrong = dict(state); wrong['detector.unknown.weight'] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            load_detector_state(model, wrong)


if __name__ == '__main__':
    unittest.main()
