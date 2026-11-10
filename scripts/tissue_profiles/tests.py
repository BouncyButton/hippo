"""Geometric semantics, loss gradients, restart recovery and native-model checks."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.depth_supervision.losses import components
from scripts.depth_supervision.train import Patients, model_factory, autocast
from scripts.depth_supervision.utils import atomic_json
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from .profiles import make_geometry, tissue_mass, profile_distance, profile_loss, ProfileCache
from .multiscale import DEFAULT_SCALES, make_strip_geometry, strip_mass, multiscale_distance, multiscale_loss, MultiscaleCache
from . import train
from .informed import native_plane_channels, InformedSwin, NativePlaneCache
from .mri_views import MRIOnlyViewSwin, MRIOnlyEncoder, MRI_ARMS, NativeViewAtlas, native_chart_inputs


class TinyPatients:
    def __init__(self, dataframe, names, metadata):
        generator = torch.Generator().manual_seed(56)
        self.items = [dict(case_name=name, image=torch.randn(1, 4, 4, 4, generator=generator),
                           label=torch.randint(0, 3, (1, 4, 4, 4), generator=generator), affine=np.eye(4))
                      for name in names]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]


class TinyProfiles:
    def __init__(self, *args):
        self.geometry = make_geometry([4]*3, [4]*3, np.eye(4), [1.5]*3, np.eye(3))
    def get(self, item):
        return self.geometry


class TinyDiagnostics:
    def __init__(self, *args):
        pass
    def get(self, item, fields=False):
        return dict(bin_ids=np.zeros((4, 4, 4), dtype=int))


class TinyMultiscale:
    def __init__(self, *args):
        self.geometry = [make_strip_geometry([4]*3, [4]*3, np.eye(4), [1.5]*3, np.eye(3), **s)
                         for s in DEFAULT_SCALES]
    def get(self, item):
        return self.geometry


class TinyBranch(torch.nn.Module):
    def __init__(self, device, arm):
        super().__init__()
        self.native = torch.nn.Conv3d(1, 3, 1)
        self.encoder = torch.nn.Conv3d(1, 3, 1)
        self.fusion = torch.nn.Conv3d(3, 3, 1)
        self.to(device)
    def forward(self, image, atlas):
        return self.native(image)+self.fusion(self.encoder(image)*atlas['support'])


class TinyAtlas:
    def __init__(self, *args):
        pass
    def get(self, item):
        return dict(support=1.)


class TinyInformed(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.native = torch.nn.Conv3d(4, 3, 1).to(device)
    def forward(self, image, atlas):
        return self.native(torch.cat([image, atlas['native_channels'][None].to(image)], 1))


class TinyPlaneCache:
    def __init__(self, *args):
        pass
    def get(self, item):
        channels = native_plane_channels([4]*3, item['affine'], [1.5]*3, np.eye(3), np.arange(64))
        return dict(native_channels=torch.from_numpy(channels))


def cancellation_example():
    shape = [12, 12, 4]
    truth = torch.zeros(2, *shape)
    truth[0, 1:11, 1, 1] = 1.
    wrong = torch.zeros_like(truth)
    wrong[0, 1:11, 9, 1] = 1.
    args = (shape, shape, np.eye(4), np.zeros(3), np.eye(3))
    original = make_geometry(*args)
    scales = [make_strip_geometry(*args, **s) for s in DEFAULT_SCALES]
    return dict(missing_voxels=10, extra_voxels=10,
                pooled_loss=float(profile_distance(tissue_mass(wrong, original), tissue_mass(truth, original))),
                coarse_loss=float(multiscale_distance(wrong, truth, scales[:1])),
                fine_loss=float(multiscale_distance(wrong, truth, scales[1:])),
                multiscale_loss=float(multiscale_distance(wrong, truth, scales)))


class ProfileTests(unittest.TestCase):
    def test_native_view_sampling_keeps_chart_membership_and_voxel_positions(self):
        shape = (16, 18, 20)
        image = np.random.default_rng(4).normal(size=shape).astype(np.float32)
        memberships = [np.ravel_multi_index(np.array(p).T, shape) for p in
                       ([[2, 3, 4], [4, 8, 10], [6, 6, 7]], [[12, 13, 14], [11, 9, 12]])]
        pack = native_chart_inputs(image, memberships)
        np.testing.assert_array_equal(np.sort(pack['roi_indices']), np.sort(np.concatenate(memberships)))
        for i, indices in enumerate(memberships):
            volume = torch.from_numpy(pack[f'native_{i}_input'])[None]
            grid = torch.from_numpy(pack[f'native_{i}_grid'])[None, :, None, None]
            sampled = torch.nn.functional.grid_sample(volume, grid, align_corners=True)[0, :, :, 0, 0]
            expected = torch.from_numpy(image.ravel()[indices].astype(np.float16).astype(np.float32))
            torch.testing.assert_close(sampled[0], expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(sampled[3], torch.ones(len(indices)), atol=1e-5, rtol=0)
            self.assertEqual(float(volume[:, 1:3].abs().sum()), 0.)
            self.assertEqual(float(volume[:, 3].sum()), len(indices))
            self.assertTrue(all(s % 8 == 0 for s in volume.shape[2:]))

    def test_mri_view_controls_match_parameters_and_initialization(self):
        for kind in ('context', 'attention'):
            torch.manual_seed(0)
            native = MRIOnlyViewSwin(torch.device('cpu'), 'native_mri_'+kind)
            for view in ('plane', 'unfold'):
                torch.manual_seed(0)
                alternative = MRIOnlyViewSwin(torch.device('cpu'), view+'_mri_'+kind)
                self.assertEqual(sum(p.numel() for p in native.parameters()), sum(p.numel() for p in alternative.parameters()))
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(value, alternative.state_dict()[key], atol=0, rtol=0)
            self.assertEqual(native.settings['view'], 'native')

    def test_informed_initialization_preserves_the_native_model(self):
        torch.manual_seed(0)
        original = model_factory(torch.device('cpu'))
        torch.manual_seed(0)
        informed = InformedSwin(torch.device('cpu'))
        expanded = {name+'.weight' for name in informed.expanded_inputs}
        for name, value in original.state_dict().items():
            new = informed.native.state_dict()[name]
            if name in expanded:
                torch.testing.assert_close(value, new[:, :1], rtol=0, atol=0)
                self.assertEqual(float(new[:, 1:].abs().sum()), 0.)
            else:
                torch.testing.assert_close(value, new, rtol=0, atol=0)
        self.assertTrue(all(q.requires_grad for q in informed.parameters()))

    def test_mri_only_encoder_ignores_position_channels_and_trains_mri(self):
        from scripts.unwrapped_swin.models import ChartEncoder
        torch.manual_seed(0)
        model = MRIOnlyEncoder(ChartEncoder(context=True))
        packed = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        with torch.no_grad():
            packed[:, 3] = 1
        output = model(packed)
        changed = packed.detach().clone()
        changed[:, 1:3] = torch.randn_like(changed[:, 1:3])*100
        torch.testing.assert_close(output, model(changed), rtol=0, atol=0)
        output.square().mean().backward()
        self.assertGreater(float(packed.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(packed.grad[:, 1:3].abs().sum()), 0)
        self.assertTrue(all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters()))
        self.assertTrue(all(model.get_submodule(name).in_channels == 1 for name in model.reduced_inputs))

    def test_native_plane_coordinates_preserve_roi_and_signed_physical_positions(self):
        # Anisotropic, translated and rotated native grid, both plane sides.
        shape = (7, 8, 9)
        affine = np.array([[0., -2., 0., 14.], [1., 0., 0., -3.], [0., 0., 3., -9.], [0., 0., 0., 1.]])
        basis = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        center = np.array([6., 0., 0.])
        points = np.array([[1, 2, 2], [5, 6, 6], [3, 4, 3]])
        indices = np.ravel_multi_index(points.T, shape)
        channels = native_plane_channels(shape, affine, center, basis, indices)
        expected = ((points @ affine[:3, :3].T+affine[:3, 3]-center) @ basis)
        np.testing.assert_allclose(channels[0].ravel()[indices], expected[:, 0]/32)
        np.testing.assert_allclose(channels[1].ravel()[indices], expected[:, 2]/16)
        self.assertTrue((channels[1] > 0).any() and (channels[1] < 0).any())
        self.assertEqual(int(channels[2].sum()), 3)
        self.assertTrue((channels[:2, channels[2] == 0] == 0).all())
        with self.assertRaises(ValueError):
            native_plane_channels(shape, affine, center, basis, [indices[0], indices[0]])

    def test_native_plane_cache_ignores_labels_and_chart_fields_and_rejects_padding(self):
        from scripts.depth_supervision.utils import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = root/'atlas/5pct'
            atlas.mkdir(parents=True)
            path = atlas/'case.npz'
            shape = (8, 8, 8)
            def write(indices):
                # No chart fields exist: this path cannot consume an alternate view.
                np.savez(path, center=np.array([3.5]*3), basis=np.eye(3), roi_indices=indices)
                atomic_json(atlas/'manifest.json', dict(complete=True, protocol_sha256='frozen',
                                                       artifacts={path.name: digest(path)}))
            write(np.ravel_multi_index(np.array([[2, 2, 2], [5, 5, 5]]).T, shape))
            p = dict(atlas_protocol_sha256='frozen', native_geometry={'case': dict(native_shape=[4]*3)})
            cache = NativePlaneCache(root, p, '5pct')
            item = dict(case_name='case', image=torch.zeros(1, *shape), affine=np.eye(4))
            a = cache.get(item)['native_channels']
            b = cache.get(dict(item, label=torch.ones(1, *shape)))['native_channels']
            np.testing.assert_array_equal(a, b)
            self.assertEqual(float(a[2].sum()), 2.)
            write(np.array([0]))
            with self.assertRaisesRegex(ValueError, 'padding'):
                NativePlaneCache(root, p, '5pct').get(item)

    def test_informed_dice_only_never_calibrates_and_resumes_exactly(self):
        with patch.object(train, 'calibrate', side_effect=AssertionError('Dice-only calibrated a constraint')):
            self.check_restart('none', 'informed')

    def test_informed_pooled_resume(self):
        self.check_restart('pooled', 'informed')

    def test_informed_multiscale_resume(self):
        self.check_restart('multiscale', 'informed')

    def test_mri_only_dice_resume_without_calibration(self):
        with patch.object(train, 'calibrate', side_effect=AssertionError('MRI-only Dice calibrated a constraint')):
            self.check_restart('none', 'plane_mri_context')

    def test_physical_mass_and_padding(self):
        affine = np.diag([1., 2., 3., 1.])
        g = make_geometry([7]*3, [2, 3, 4], affine, [3., 6., 9.], np.eye(3))
        x = torch.ones(2, 7, 7, 7, requires_grad=True)
        masses = tissue_mass(x, g)
        torch.testing.assert_close(masses.sum((1, 2)), torch.tensor([144., 144.]))
        masses.sum().backward()
        self.assertEqual(int((x.grad[0] > 0).sum()), 24)
        self.assertEqual(float(x.grad[:, 0].sum()), 0.)
        self.assertEqual(sum(g['lengths']), 24)

    def test_distance_increases_with_displacement_and_preserves_mass(self):
        target = torch.zeros(2, 2, 12)
        target[0, 0, 2] = 10.
        def moved(index):
            pred = torch.zeros_like(target)
            pred[0, 0, index] = 10.
            return profile_distance(pred, target)
        self.assertEqual(float(moved(2)), 0.)
        self.assertGreater(float(moved(8)), float(moved(3)))
        self.assertGreater(float(profile_distance(target*.5, target)), 0.)
        self.assertGreater(float(profile_distance(target*1.5, target)), 0.)
        self.assertGreater(float(profile_distance(target.flip(0), target)), 0.)
        self.assertGreater(float(profile_distance(target.flip(1), target)), 0.)

    def test_reversal_invariance_and_no_empty_band_reweighting(self):
        target = torch.rand(2, 3, 11)
        pred = torch.rand_like(target)
        value = profile_distance(pred, target)
        torch.testing.assert_close(value, profile_distance(pred.flip(-1), target.flip(-1)))
        empty = torch.zeros(2, 4, 11)
        torch.testing.assert_close(value, profile_distance(torch.cat((pred, empty), 1), torch.cat((target, empty), 1)))
        # Duplicating tissue and discrepancies doubles both mass and scale.
        torch.testing.assert_close(value, profile_distance(2*pred, 2*target))

    def test_gradients_missed_extra_swaps_and_empty_target(self):
        truth = torch.zeros(2, 2, 10)
        truth[0, 0, 2] = 1.
        truth[1, 0, 7] = 1.
        pred = truth.clone()
        pred[0, 0, 2] = .2
        pred[0, 1, 5] = .4
        pred.requires_grad_()
        profile_distance(pred, truth).backward()
        self.assertLess(float(pred.grad[0, 0, 2]), 0.)
        self.assertGreater(float(pred.grad[0, 1, 5]), 0.)
        self.assertTrue(torch.isfinite(profile_distance(pred, truth*0)))
        g = make_geometry([4]*3, [4]*3, np.eye(4), [1.5]*3, np.eye(3))
        logits = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
        label = torch.randint(0, 3, (1, 1, 4, 4, 4))
        loss = profile_loss(logits, label, g)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        self.assertEqual(set(components(logits, label, None, 'dice')), {'dice'})

    def test_restart_before_and_after_calibration_and_completion_skip(self):
        self.check_restart('pooled')

    def test_multiscale_restart_before_and_after_calibration_and_completion_skip(self):
        self.check_restart('multiscale')

    def test_branch_multiscale_training_calibration_and_resume(self):
        self.check_restart('multiscale', 'plane_position')

    def check_restart(self, variant, arm='native'):
        p = dict(identity_sha='test', splits={'5pct': dict(train=['a', 'b'], val=['v'])}, native_geometry={},
                 pkl='unused', learning_rate=.01, weight_decay=1e-5, step_size=2, lr_gamma=.5,
                 geometry={}, profile_geometry=dict(band_mm=2., longitudinal_mm=2.),
                 multiscale_geometry=DEFAULT_SCALES, max_epochs=7,
                 calibration=dict(prefix_epochs=5, ramp_epochs=5, target_ratio=.1, cap_ratio=.5, max_cases=32),
                 early_stopping=dict(patience=8, min_epochs=25, min_delta=.0005))
        entry = dict(id='test', seed=0, fraction='5pct', profile_variant=variant, arm=arm)
        factory = lambda device: torch.nn.Conv3d(1, 3, 1).to(device)
        with tempfile.TemporaryDirectory() as directory, patch.object(train, 'model_factory', factory), \
             patch.object(train, 'Patients', TinyPatients), patch.object(train, 'ProfileCache', TinyProfiles), \
             patch.object(train, 'MultiscaleCache', TinyMultiscale), \
             patch.object(train, 'DualViewSwin', TinyBranch), patch.object(train, 'FrozenAtlas', TinyAtlas), \
             patch.object(train, 'MRIOnlyViewSwin', TinyBranch), \
             patch.object(train, 'InformedSwin', TinyInformed), patch.object(train, 'NativePlaneCache', TinyPlaneCache), \
             patch.object(train, 'GeometryCache', TinyDiagnostics), patch.object(train, '_load_pkl_dataframe', return_value=None):
            a = Path(directory)/'continuous'
            train.run(p, a, entry, device_override='cpu')
            config = json.loads((a/'runs/test/config.json').read_text())
            self.assertEqual(config['auxiliary_branch'], arm not in ('native', 'informed'))
            if variant == 'none':
                self.assertIsNone(config['calibration'])
                self.assertEqual(config['objective'], 'Dice only')
            for stopped in (5, 6):
                b = Path(directory)/f'resumed_{stopped}'
                train.run(p, b, entry, stop_after=stopped, device_override='cpu')
                with self.assertRaisesRegex(ValueError, 'Resume identity'):
                    train.run(p, b, dict(entry, seed=1), device_override='cpu')
                rb = train.run(p, b, entry, device_override='cpu')
                sa = torch.load(a/'runs/test/best.pt', weights_only=False)
                sb = torch.load(b/'runs/test/best.pt', weights_only=False)
                self.assertEqual(sa['epoch'], sb['epoch'])
                for key in sa['model']:
                    torch.testing.assert_close(sa['model'][key], sb['model'][key], rtol=0, atol=0)
                self.assertEqual(rb['optimizer_updates'], 14)
                with (a/'runs/test/metrics.csv').open() as fa, (b/'runs/test/metrics.csv').open() as fb:
                    ha, hb = list(csv.DictReader(fa)), list(csv.DictReader(fb))
                for ra, row in zip(ha, hb):
                    self.assertEqual({k: v for k, v in ra.items() if k not in ('elapsed_seconds', 'epoch_seconds')},
                                     {k: v for k, v in row.items() if k not in ('elapsed_seconds', 'epoch_seconds')})
                with patch.object(train, 'model_factory', side_effect=AssertionError('Completed run retrained')):
                    self.assertEqual(train.run(p, b, entry, device_override='cpu'), rb)

    def test_opposite_edge_cancellation_is_detected(self):
        result = cancellation_example()
        self.assertEqual(result['pooled_loss'], 0.)
        self.assertGreater(result['coarse_loss'], 0.)
        self.assertGreater(result['fine_loss'], 0.)
        self.assertAlmostEqual(result['multiscale_loss'], .5*(result['coarse_loss']+result['fine_loss']))

    def test_fine_scale_exposes_coarse_cancellation_and_retains_limit(self):
        args = ([8]*3, [8]*3, np.eye(4), np.zeros(3), np.eye(3))
        scales = [make_strip_geometry(*args, **s) for s in DEFAULT_SCALES]
        truth = torch.zeros(2, 8, 8, 8)
        truth[0, 2, 0, 0] = 1.
        wrong = torch.zeros_like(truth)
        wrong[0, 2, 2, 0] = 1.
        self.assertEqual(float(multiscale_distance(wrong, truth, scales[:1])), 0.)
        self.assertGreater(float(multiscale_distance(wrong, truth, scales[1:])), 0.)
        wrong.requires_grad_()
        multiscale_distance(wrong, truth, scales).backward()
        self.assertLess(float(wrong.grad[0, 2, 0, 0]), 0.)
        self.assertGreater(float(wrong.grad[0, 2, 2, 0]), 0.)
        # Distinct voxels inside the same finest cell remain indistinguishable.
        within = torch.zeros_like(truth)
        within[0, 2, 1, 0] = 1.
        self.assertEqual(float(multiscale_distance(within, truth, scales)), 0.)
        labels = truth.argmax(0, keepdim=True)+1
        labels = torch.where(truth.sum(0, keepdim=True) > 0, labels, 0)[None]
        probabilities = torch.cat((1-within.sum(0, keepdim=True), within), 0)
        dice = components(probabilities.clamp_min(1e-8).log()[None], labels, None, 'dice')['dice']
        self.assertGreater(float(dice), 0.)

    def test_multiscale_native_mass_and_single_global_normalization(self):
        shape, native = [10]*3, [3, 5, 7]
        affine = np.diag([1., 2., 3., 1.])
        scales = [make_strip_geometry(shape, native, affine, np.zeros(3), np.eye(3), **s)
                  for s in DEFAULT_SCALES]
        x = torch.ones(2, *shape, requires_grad=True)
        for g in scales:
            self.assertEqual(len(np.unique(g['indices'])), 105)
            self.assertEqual(sum(g['lengths']), 105)
            mass = strip_mass(x, g)
            torch.testing.assert_close(mass.sum((1, 2, 3)), torch.tensor([630., 630.]))
            grad = torch.autograd.grad(mass.sum(), x, retain_graph=True)[0]
            self.assertEqual(int((grad[0] > 0).sum()), 105)
            # Appending empty strips changes neither numerator nor denominator.
            target = mass.detach()
            wrong = target*.75
            empty = torch.zeros(2, g['n_depth'], 3, g['n_long'])
            expected = profile_distance(wrong.flatten(1, 2), target.flatten(1, 2))
            actual = profile_distance(torch.cat((wrong, empty), 2).flatten(1, 2),
                                      torch.cat((target, empty), 2).flatten(1, 2))
            torch.testing.assert_close(expected, actual)

    def test_twelve_run_report_preserves_matched_controls(self):
        from .report import report, baseline_folder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'current'
            prior = Path(directory)/'prior'
            p = dict(identity_sha='test', prior_root=str(prior), runs=[],
                     branch_baseline_root=str(Path(directory)/'branch_controls'),
                     view_pairs={'5pct': ['plane_position', 'unfold_position'],
                                 'full': ['plane_attention', 'unfold_attention']})
            metrics = dict(macro_dice=.8, foreground_dice=.9, missed=2, extra=3, swaps=1)
            for fraction in ('5pct', 'full'):
                for arm in ['native', *p['view_pairs'][fraction]]:
                    folders = [baseline_folder(p, fraction, arm)]
                    for variant in ('pooled', 'multiscale'):
                        suffix = 'profile' if variant == 'pooled' else variant
                        arm_suffix = '' if arm == 'native' else '_'+arm
                        entry = dict(id=f'{fraction}_dice_{suffix}_s0{arm_suffix}', fraction=fraction,
                                     profile_variant=variant, arm=arm)
                        p['runs'].append(entry)
                        folders.append(root/'runs'/entry['id'])
                    for i, folder in enumerate(folders):
                        folder.mkdir(parents=True)
                        values = dict(metrics, macro_dice=.8+i*.01)
                        atomic_json(folder/'completion.json', dict(best_epoch=7, completed_epochs=9, validation=values))
                        atomic_json(folder/'validation_best.json', dict(cases=[dict(case_name='case', **values)]))
            result = report(p, root)
            self.assertEqual(result['new_completed'], 12)
            self.assertEqual(len(result['rows']), 18)
            self.assertEqual(len(result['paired_deltas']), 22)
            self.assertAlmostEqual(result['paired_deltas']['full_multiscale_minus_pooled'][0]['macro_dice'], .01)
            self.assertAlmostEqual(result['paired_deltas']['full_plane_attention_multiscale_minus_pooled'][0]['macro_dice'], .01)
            self.assertAlmostEqual(result['paired_deltas']['5pct_multiscale_unfold_minus_plane'][0]['macro_dice'], 0.)
            # Append the ten requested controls and check both partial/final reports.
            p['mri_view_pairs'] = {'5pct': ['plane_mri_context', 'unfold_mri_context'],
                                   'full': ['plane_mri_attention', 'unfold_mri_attention']}
            for fraction in ('5pct', 'full'):
                for arm, variants in [('informed', ['none', 'pooled', 'multiscale']),
                                      *[(a, ['none']) for a in p['mri_view_pairs'][fraction]]]:
                    for variant in variants:
                        label = {'none': 'dice', 'pooled': 'dice_profile', 'multiscale': 'dice_multiscale'}[variant]
                        entry = dict(id=f'{fraction}_{label}_s0_{arm}', fraction=fraction,
                                     profile_variant=variant, arm=arm)
                        p['runs'].append(entry)
                        before = report(p, root)['new_completed']
                        folder = root/'runs'/entry['id']
                        folder.mkdir(parents=True)
                        values = dict(metrics, macro_dice=.84)
                        atomic_json(folder/'completion.json', dict(best_epoch=7, completed_epochs=9, validation=values))
                        atomic_json(folder/'validation_best.json', dict(cases=[dict(case_name='case', **values)]))
                        self.assertEqual(report(p, root)['new_completed'], before+1)
            result = report(p, root)
            self.assertEqual(result['new_completed'], 22)
            self.assertEqual(len(result['rows']), 28)
            self.assertEqual(len(result['paired_deltas']), 68)
            self.assertAlmostEqual(result['paired_deltas']['5pct_none_informed_minus_native'][0]['macro_dice'], .04)
            self.assertAlmostEqual(result['paired_deltas']['full_plane_mri_attention_minus_position_branch'][0]['macro_dice'], .04)
            self.assertAlmostEqual(result['paired_deltas']['full_mri_unfold_minus_plane'][0]['macro_dice'], 0.)
            p['native_mri_arms'] = {'5pct': 'native_mri_context', 'full': 'native_mri_attention'}
            for fraction in ('5pct', 'full'):
                for arm, variants in [*[(a, ['pooled', 'multiscale']) for a in p['mri_view_pairs'][fraction]],
                                      (p['native_mri_arms'][fraction], ['none', 'pooled', 'multiscale'])]:
                    for variant in variants:
                        label = {'none': 'dice', 'pooled': 'dice_profile', 'multiscale': 'dice_multiscale'}[variant]
                        entry = dict(id=f'{fraction}_{label}_s0_{arm}', fraction=fraction, profile_variant=variant, arm=arm)
                        p['runs'].append(entry)
                        folder = root/'runs'/entry['id']
                        folder.mkdir(parents=True)
                        values = dict(metrics, macro_dice=.83, surface_distance_mean_mm=.4, hd95_mm=1.2)
                        atomic_json(folder/'completion.json', dict(best_epoch=7, completed_epochs=9,
                                                                 parameter_count=123, training_seconds=8., validation=values))
                        atomic_json(folder/'validation_best.json', dict(cases=[dict(case_name='case', **values)]))
                        report(p, root)
            result = report(p, root)
            self.assertEqual(result['new_completed'], 36)
            self.assertEqual(len(result['rows']), 42)
            self.assertAlmostEqual(result['paired_deltas']['full_none_native_aux_minus_ordinary'][0]['macro_dice'], .03)
            self.assertAlmostEqual(result['paired_deltas']['full_multiscale_plane_mri_attention_minus_native_aux'][0]['macro_dice'], 0.)
            self.assertIn('full_unfold_attention_multiscale_position_minus_mri', result['paired_deltas'])
            self.assertIn('full_dice_multiscale_s0_native_mri_attention_minus_dice', result['paired_deltas'])
            self.assertEqual(result['rows'][-1]['parameter_count'], 123)
            self.assertEqual(result['rows'][-1]['training_seconds'], 8.)


def model_smoke(device_name, item=None, geometry=None, variant='pooled', arm='native', atlas=None):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device(device_name)
    if arm == 'informed':
        model = InformedSwin(device)
    elif arm in MRI_ARMS:
        model = MRIOnlyViewSwin(device, arm)
    else:
        model = model_factory(device) if arm == 'native' else train.DualViewSwin(device, arm)
    if item is None:
        item = dict(image=torch.randn(1, 64, 64, 64), label=torch.zeros(1, 64, 64, 64, dtype=torch.long))
        item['label'][:, 24:32, 24:40, 24:40] = 1
        item['label'][:, 32:40, 24:40, 24:40] = 2
        args = ([64]*3, [48]*3, np.eye(4), [31.5]*3, np.eye(3))
        geometry = (make_geometry(*args) if variant == 'pooled' else
                    [make_strip_geometry(*args, **s) for s in DEFAULT_SCALES])
        if arm == 'informed':
            roi = np.zeros((64, 64, 64), bool)
            roi[16:48, 16:48, 16:48] = True
            atlas = dict(native_channels=native_plane_channels([64]*3, np.eye(4), [31.5]*3,
                                                               np.eye(3), np.flatnonzero(roi)))
        elif arm.startswith('native_mri_'):
            if atlas is None:
                raise ValueError('Native-view smoke requires real frozen chart memberships')
            atlas = native_chart_inputs(item['image'][0].numpy(),
                                       [atlas[f'plane_{i}_indices'] for i in range(int(atlas['charts']))])
    _, loss_function = train.profile_implementation(variant)
    model.train()
    with autocast(device):
        image = item['image'][None].to(device)
        logits = model(image) if arm == 'native' else model(image, atlas)
    labels = item['label'][None].to(device)
    dice = components(logits, labels, None, 'dice')['dice']
    profile, profile_gradient = logits.new_zeros(()), torch.zeros_like(logits)
    if loss_function is not None:
        profile = loss_function(logits, labels, geometry)
        profile_gradient = torch.autograd.grad(profile, logits, retain_graph=True)[0]
        assert torch.isfinite(profile_gradient).all() and profile_gradient.abs().sum() > 0
        # Stable segment reductions must give identical values and gradients.
        repeated = loss_function(logits, labels, geometry)
        torch.testing.assert_close(profile, repeated, atol=0, rtol=0)
        torch.testing.assert_close(profile_gradient, torch.autograd.grad(repeated, logits, retain_graph=True)[0], atol=0, rtol=0)
    loss = dice+.1*profile
    loss.backward()
    gradients = [q.grad for q in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    native = model if arm == 'native' else model.native
    modules = [native.swinViT, native.encoder1, native.decoder1, native.out]
    if arm not in ('native', 'informed'):
        modules.extend([model.encoder, model.fusion])
    for module in modules:
        assert sum(float(q.grad.abs().sum()) for q in module.parameters()) > 0
    channel_gradients = {}
    if arm == 'informed':
        for name in model.expanded_inputs:
            layer = native.get_submodule(name)
            per_channel = layer.weight.grad.abs().sum((0, 2, 3, 4))
            assert bool((per_channel > 0).all())
            channel_gradients[name] = per_channel.tolist()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, foreach=False)
    optimizer.step()
    assert all(torch.isfinite(q).all() for q in model.parameters())
    target = torch.stack((labels[0, 0] == 1, labels[0, 0] == 2)).float()
    if variant == 'pooled':
        torch.testing.assert_close(tissue_mass(target, geometry).sum(), target.sum()*geometry['voxel_mm3'])
    elif variant == 'multiscale':
        for g in geometry:
            torch.testing.assert_close(strip_mass(target, g).sum(), target.sum()*g['voxel_mm3'])
    return dict(passed=True, device=device_name, variant=variant, arm=arm, full_native_network_trained=True,
                branch_and_fusion_gradients_checked=arm not in ('native', 'informed'),
                input_channel_gradients=channel_gradients,
                mri_only_auxiliary=arm in MRI_ARMS,
                parameter_count=sum(q.numel() for q in model.parameters()), dice=float(dice.detach()),
                profile=float(profile.detach()), profile_gradient_rms=float(profile_gradient.square().mean().sqrt()),
                target_mass_preserved=variant != 'none', repeated_profile_exact=variant != 'none',
                peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--fraction', choices=['5pct', 'full'])
    parser.add_argument('--variant', choices=['none', 'pooled', 'multiscale'], default='pooled')
    parser.add_argument('--arm', choices=['native', 'informed', 'plane_position', 'unfold_position', 'plane_attention', 'unfold_attention', *MRI_ARMS], default='native')
    parser.add_argument('--atlas', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.root:
        p = json.loads((args.root/'protocol.json').read_text())
        dataframe = _load_pkl_dataframe(p['pkl'])
        patients = Patients(dataframe, p['splits'][args.fraction]['train'], p['native_geometry'])
        cache_factory, _ = train.profile_implementation(args.variant)
        cache = cache_factory(args.root, p, args.fraction) if cache_factory else None
        # Validate every training case's native tissue coverage before training.
        for item in patients.items:
            if cache is None:
                continue
            g = cache.get(item)
            target = torch.stack((item['label'][0] == 1, item['label'][0] == 2)).float()
            if args.variant == 'pooled':
                torch.testing.assert_close(tissue_mass(target, g).sum(), target.sum()*g['voxel_mm3'])
            else:
                for scale in g:
                    torch.testing.assert_close(strip_mass(target, scale).sum(), target.sum()*scale['voxel_mm3'])
        selected, atlas = patients[0], None
        if args.arm == 'informed':
            atlas_cache = NativePlaneCache(args.root, p, args.fraction)
            for item in patients.items:
                atlas_cache.get(item)
            atlas = atlas_cache.get(selected)
        elif args.arm.startswith('native_mri_'):
            atlas_cache = NativeViewAtlas(args.root, p, args.fraction)
            largest_size = -1
            for item in patients.items:
                candidate = atlas_cache.get(item)
                size = sum(np.prod(candidate[f'native_{i}_input'].shape[1:]) for i in range(int(candidate['charts'])))
                if size > largest_size:
                    selected, atlas, largest_size = item, candidate, size
        elif args.arm != 'native':
            atlas_cache = train.FrozenAtlas(args.root, p, args.fraction)
            view = 'plane' if args.arm.startswith('plane_') else 'unfold'
            largest = max(atlas_cache.manifest['reports'],
                          key=lambda r: sum(np.prod(a['grid_shape']) for a in r['charts_audit'] if a['view'] == view))
            # The largest chart may be a validation case: this is only a disposable
            # architecture/memory check; actual training reloads its own partition.
            selected = Patients(dataframe, [largest['case']], p['native_geometry'])[0]
            atlas = atlas_cache.get(selected)
        result = model_smoke('cuda', selected, cache.get(selected) if cache else None, args.variant, args.arm, atlas)
        result.update(case=selected['case_name'], training_geometries_verified=len(patients))
    elif args.smoke:
        atlas = None
        if args.arm not in ('native', 'informed'):
            if args.atlas is None:
                raise ValueError('A real frozen atlas is required for branch smoke checks')
            with np.load(args.atlas, allow_pickle=False) as archive:
                atlas = {k: archive[k] for k in archive.files}
        result = model_smoke('cpu', variant=args.variant, arm=args.arm, atlas=atlas)
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProfileTests)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        result = dict(passed=outcome.wasSuccessful(), tests=outcome.testsRun,
                      cancellation_example=cancellation_example())
        if not result['passed']:
            raise SystemExit(1)
    from scripts.depth_supervision.utils import digest
    result['tested_sources'] = {str(path.relative_to(Path(__file__).resolve().parents[2])): digest(path)
                                for path in Path(__file__).resolve().parent.glob('*.py')}
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
