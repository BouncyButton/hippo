"""Geometry correspondence and differentiable branch smoke checks."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import trimesh
from torch.nn import functional as F
from .geometry import build_atlas, disk_pieces, flatten_pieces, open_disk, raster
from .surface import boundary_loops, flatten
from .models import ARMS, ChartEncoder, DualViewSwin


class GeometryTests(unittest.TestCase):
    def test_failed_stage_stops_and_records_failure(self):
        from .worker import finish_stage
        state = dict(state='running', stages=[], current_stage='5pct_atlas')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'stopped without skipping'):
                finish_stage(Path(directory), state, '5pct_atlas', 1, .1)
            saved = json.loads((Path(directory) / 'status.json').read_text())
            self.assertEqual(saved['state'], 'failed')
            self.assertEqual(saved['failed_stage'], '5pct_atlas')

    def test_repair_reuses_only_native_training_identity(self):
        from .train import run_identity
        p = dict(identity_sha='repaired', native_training_identity_sha='original',
                 splits={'5pct': dict(train=['a'], val=['b'])},
                 guide_splits={'5pct': [dict(train=['a'], predict=['b'])]})
        for arm in ('dice', 'depth'):
            self.assertEqual(run_identity(p, '5pct', arm)['protocol_sha256'], 'original')
        self.assertEqual(run_identity(p, '5pct', 'dice', 0)['protocol_sha256'], 'original')
        self.assertEqual(run_identity(p, '5pct', 'unfold_local')['protocol_sha256'], 'repaired')

    def test_compressed_chart_partitions_preserve_source_triangles(self):
        # A long capped tube with a small opening compresses its far cap when
        # mapped to a convex disk, the same failure as the noisy guide charts.
        rings = 24
        angle = np.arange(3) * 2 * np.pi / 3
        vertices = np.array([[np.cos(a), np.sin(a), z] for z in range(rings) for a in angle])
        faces = []
        for z in range(rings - 1):
            for k in range(3):
                a, b = 3*z+k, 3*z+(k+1)%3
                faces.extend([[a, b, a+3], [b, b+3, a+3]])
        faces.append([3*rings-3, 3*rings-2, 3*rings-1])
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        with self.assertRaisesRegex(AssertionError, 'Flipped or collapsed'):
            flatten(mesh, np.array([0., 0., 1.]))
        charts = flatten_pieces(mesh, np.array([0., 0., 1.]), np.array([1., 0., 0.]))
        self.assertGreater(len(charts), 1)
        self.assertEqual(sum(len(c['mesh'].faces) for c in charts), len(mesh.faces))
        self.assertAlmostEqual(sum(c['mesh'].area for c in charts), mesh.area)
        self.assertTrue(all(c['checks']['flipped_triangles'] == 0 for c in charts))
        original = sorted(tuple(t.ravel()) for t in mesh.triangles)
        rebuilt = sorted(tuple(t.ravel()) for c in charts for t in c['mesh'].triangles)
        self.assertEqual(original, rebuilt)

    def test_closed_flat_component_is_opened_without_losing_faces(self):
        # Two coincident sides of a flat closed component, as in the failed guide.
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 1], [3, 1, 0]])
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        self.assertEqual(mesh.euler_number, 2)
        self.assertEqual(len(boundary_loops(mesh)), 0)
        pieces = disk_pieces(mesh, np.array([0., 1., 0.]))
        self.assertAlmostEqual(sum(p.area for p in pieces), mesh.area)
        self.assertEqual(sum(len(p.faces) for p in pieces), len(mesh.faces))
        for piece in pieces:
            opened = open_disk(piece)
            self.assertEqual(opened.euler_number, 1)
            self.assertEqual(len(boundary_loops(opened)), 1)
            _, _, checks = flatten(opened, np.array([0., 0., 1.]))
            self.assertEqual(checks['flipped_triangles'], 0)

    def test_raster_constant_and_return_gradient(self):
        q = np.indices((5, 6, 7)).reshape(3, -1).T.astype(float)
        values = np.c_[np.ones(len(q)) * 3, q[:, 0], q[:, 2]]
        volume, grid, audit = raster(q, values, np.ones(3))
        x = torch.tensor(volume[None], requires_grad=True)
        g = torch.tensor(grid[None, :, None, None])
        y = F.grid_sample(x, g, align_corners=True)[0, :, :, 0, 0]
        np.testing.assert_allclose(y.detach().numpy()[:3].T, values, atol=1e-5)
        y[0].sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertEqual(audit['return_support_fraction'], 1.)

    def test_cut_shell_preserves_source_and_depth(self):
        xyz = np.indices((24, 32, 24)).transpose(1, 2, 3, 0)
        q = xyz - [12, 16, 12]
        fg = ((q[..., 0] / 5) ** 2 + (q[..., 1] / 10) ** 2 + (q[..., 2] / 6) ** 2) < 1
        image = (xyz[..., 0] + xyz[..., 1] / 10).astype(np.float32)
        pack, audit = build_atlas(image, fg.astype(int), np.eye(4), np.ones_like(fg), margin_mm=2)
        self.assertEqual(audit['charts'], 2)
        self.assertTrue(audit['all_source_faces_retained'])
        self.assertFalse(audit['geometry_uses_annotation'])
        indices = np.array(np.unravel_index(pack['roi_indices'], image.shape)).T
        depth = np.abs(((indices - pack['center']) @ pack['basis'])[:, 2])
        np.testing.assert_allclose(pack['unfolded_points'][:, 2], depth, atol=1e-5)
        for i in range(int(pack['charts'])):
            self.assertEqual(pack[f'plane_{i}_input'].shape, pack[f'unfold_{i}_input'].shape)
            np.testing.assert_array_equal(pack[f'plane_{i}_indices'], pack[f'unfold_{i}_indices'])

    def test_shared_chart_encoder_gradients(self):
        for context in (False, True):
            model = ChartEncoder(context)
            a = torch.randn(1, 4, 8, 16, 16, requires_grad=True)
            a.data[:, -1] = 1
            b = a.detach().clone().requires_grad_(True)
            loss = model(a).square().mean() + model(b).abs().mean()
            loss.backward()
            self.assertGreater(float(a.grad.abs().sum()), 0)
            self.assertGreater(float(b.grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


def real_model_smoke(atlas, device='cpu'):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    model = DualViewSwin(torch.device(device), 'unfold_attention')
    from scripts.depth_supervision.train import autocast
    model.eval()
    with torch.no_grad(), autocast(torch.device(device)):
        reference = model.native(x)
        reconstructed = model.native.out(model.native_features(x))
        torch.testing.assert_close(reconstructed, reference, rtol=0, atol=0)
    model.train()
    with autocast(torch.device(device)):
        logits = model(x, atlas)
    logits.float().square().mean().backward()
    assert logits.shape == (1, 3, 64, 64, 64)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()) > 0
    assert sum(float(p.grad.abs().sum()) for p in model.fusion.parameters()) > 0
    return dict(parameters=sum(p.numel() for p in model.parameters()), shape=list(logits.shape),
                auxiliary_gradient_nonzero=True, fusion_gradient_nonzero=True,
                native_feature_path_matches_monai=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
