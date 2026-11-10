"""CPU checks of the real frozen evaluation loop and opt-in feature reuse."""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.colab_medsam3 import fast_eval as fe, runtime
from scripts.colab_medsam3 import parallel
from scripts.colab_medsam3.prepare_fast_eval import fast_notebook

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle/engine.py'


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.images, self.texts = 0, 0

    def forward_image(self, samples):
        self.images += 1
        features = samples.float()*self.weight
        return dict(vision_features=features, backbone_fpn=[features], other=None)

    def forward_text(self, captions, input_boxes=None, additional_text=None, device='cuda'):
        self.texts += 1
        return dict(language_features=torch.tensor(float(len(captions[0]))))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()


class Transform:
    enabled = False
    def __init__(self): self.calls = 0
    def clear(self): pass
    def __call__(self, value):
        self.calls += 1
        return value.float()


def forward(model, processor, pixels, labels=None, criterion=None):
    fields = []
    for text in ('anterior hippocampus', 'posterior hippocampus'):
        value = processor.transform(torch.from_numpy(pixels.copy()))
        features = model.backbone.forward_image(value)
        encoded = model.backbone.forward_text([text], device='cpu')
        # Exercise alias preservation and downstream mutation of cached values.
        assert features['vision_features'] is features['backbone_fpn'][-1]
        features['vision_features'].add_(encoded['language_features']/10)
        fields.append(features['backbone_fpn'][-1].clone())
        encoded['language_features'].add_(100)
    return torch.stack([torch.zeros_like(fields[0]), *fields]).log_softmax(0), None


def rng():
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())


def set_rng(state):
    torch.set_rng_state(state['torch']); random.setstate(state['python']); np.random.set_state(state['numpy'])


def frozen_engine():
    tree = ast.parse(FROZEN.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    namespace = dict(torch=torch, np=np, hashlib=hashlib, read=runtime.read, write=runtime.write, stopped=lambda _: False)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(FROZEN), 'exec'), namespace)
    return SimpleNamespace(evaluate=namespace['evaluate'], rng_state=rng, set_rng=set_rng,
        read=runtime.read, write=runtime.write, digest=runtime.digest, copy_verified=runtime.copy_verified,
        binding=lambda *_: dict(bundle_sha256='frozen', environment=dict(gpu='CPU test')))


class FastTests(unittest.TestCase):
    def fixture(self):
        return Model().eval(), SimpleNamespace(transform=Transform()), np.arange(16, dtype=np.uint8).reshape(4, 4)

    def test_feature_counts_exact_values_and_mutation_isolation(self):
        model, processor, pixels = self.fixture()
        reuse = fe.FeatureReuse(model, processor, forward)
        before = model.backbone.weight.clone()
        with torch.inference_mode():
            for image in (pixels, pixels+7, pixels):
                reference = forward(model, processor, image)[0]
                actual = reuse(image)[0]
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertEqual(reuse.counts, dict(image_encodes=3, image_hits=3, text_encodes=2, text_hits=4, transform_hits=3))
        torch.testing.assert_close(model.backbone.weight, before, rtol=0, atol=0)
        self.assertNotIn('forward_image', vars(model.backbone))
        self.assertNotIn('forward_text', vars(model.backbone))
        self.assertIsInstance(processor.transform, Transform)

    def test_training_and_gradients_are_rejected(self):
        model, processor, pixels = self.fixture()
        reuse = fe.FeatureReuse(model, processor, forward)
        with self.assertRaisesRegex(RuntimeError, 'inference'):
            reuse(pixels)
        with torch.inference_mode():
            model.train()
            with self.assertRaisesRegex(RuntimeError, 'inference'):
                reuse(pixels)
            model.eval().requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, 'frozen'):
                reuse(pixels)

    def test_exception_restores_hooks(self):
        model, processor, pixels = self.fixture()
        transform = processor.transform
        def bad(*args): raise RuntimeError('broken decoder')
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'broken decoder'):
            fe.FeatureReuse(model, processor, bad)(pixels)
        self.assertIs(processor.transform, transform)
        self.assertNotIn('forward_image', vars(model.backbone))
        self.assertNotIn('forward_text', vars(model.backbone))

    def test_different_prompt_images_cannot_reuse_features(self):
        model, processor, pixels = self.fixture()
        def mismatched(model, processor, pixels):
            model.backbone.forward_image(torch.ones(2, 2))
            return model.backbone.forward_image(torch.zeros(2, 2))
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'different images'):
            fe.FeatureReuse(model, processor, mismatched)(pixels)

    def test_cache_does_not_cross_model_instances(self):
        model, processor, pixels = self.fixture()
        with torch.inference_mode():
            fe.FeatureReuse(model, processor, forward)(pixels)
            other = Model().eval()
            fe.FeatureReuse(other, processor, forward)(pixels)
        self.assertEqual(other.backbone.images, 1)
        self.assertEqual(other.backbone.texts, 2)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root/'work/task'
        self.output.mkdir(parents=True)
        self.model, self.processor, self.pixels = FastTests().fixture()
        self.engine = frozen_engine()
        self.original = SimpleNamespace(forward_plane=forward)
        self.receipt = self.root/'receipt.json'

    def tearDown(self): self.temp.cleanup()

    def audit(self, **kw):
        return fe.EvaluationAudit(self.engine, self.original, self.model, self.processor,
            self.output, self.receipt, dict(checkpoint_sha256='checkpoint'), **kw)

    def run_volume(self, audit, case='v', depth=4):
        audit.begin_volume(case, depth)
        with torch.inference_mode():
            return [audit.plane(self.model, self.processor, self.pixels+i)[0] for i in range(depth)]

    def test_full_volume_then_sentinel_and_receipt_reuse(self):
        audit = self.audit(minimum_gain=0.)
        self.run_volume(audit)
        self.assertEqual(audit.mode, 'fast')
        self.assertEqual(audit.data['full_volume']['checked'], 4)
        self.run_volume(audit, 'next')
        self.assertEqual(audit.data['checked_slices'], 5)
        self.assertEqual(audit.data['sentinels'][0]['case'], 'next')
        again = self.audit()
        self.assertEqual(again.mode, 'fast')
        changed = fe.EvaluationAudit(self.engine, self.original, self.model, self.processor,
            self.output, self.receipt, dict(checkpoint_sha256='other'))
        self.assertEqual(changed.mode, 'audit')

    def test_numerical_failure_returns_original(self):
        audit = self.audit(minimum_gain=0.)
        def wrong(_): return torch.ones(3, 4, 4), None
        class Wrong:
            counts = {}
            def __call__(self, _): return wrong(None)
            def clear(self): pass
        audit.fast = Wrong()
        results = self.run_volume(audit)
        self.assertEqual(audit.mode, 'original')
        with torch.inference_mode():
            for i, result in enumerate(results):
                torch.testing.assert_close(result, forward(self.model, self.processor, self.pixels+i)[0], rtol=0, atol=0)

    def test_no_speed_gain_keeps_original(self):
        audit = self.audit(minimum_gain=1e9)
        self.run_volume(audit)
        self.assertEqual(audit.mode, 'original')
        self.assertIn('No measured', audit.data['fallback_reason'])

    def test_stochastic_forward_falls_back_and_preserves_reference_rng(self):
        def stochastic(*args):
            torch.rand(1)
            return forward(*args)
        self.original.forward_plane = stochastic
        audit = self.audit(minimum_gain=0.)
        audit.begin_volume('v', 4)
        torch.manual_seed(9)
        with torch.inference_mode():
            stochastic(self.model, self.processor, self.pixels)
            expected = torch.get_rng_state()
            torch.manual_seed(9)
            audit.plane(self.model, self.processor, self.pixels)
        self.assertTrue(torch.equal(expected, torch.get_rng_state()))
        self.assertEqual(audit.mode, 'original')
        self.assertFalse(audit.data['rng_agreement'])

    def test_sentinel_failure_reverts_before_fast_volume(self):
        audit = self.audit(minimum_gain=0.)
        self.run_volume(audit)
        cached = next(iter(audit.fast.text.values()))
        with torch.inference_mode(): cached['language_features'].add_(10)
        self.run_volume(audit, 'next')
        self.assertEqual(audit.mode, 'original')

    def test_frozen_loop_preserves_partial_results_metrics_and_hashes(self):
        images = {case: (np.arange(64).reshape(4, 4, 4)/64).astype(np.float32) for case in ('done', 'support', 'val')}
        loads = []
        def load(_root, case):
            loads.append(case)
            return None, images[case]
        original = SimpleNamespace(forward_plane=forward, load_image=load,
            read_labels=lambda _r, case, _s: np.ones(images[case].shape, dtype=np.uint8),
            metrics=lambda truth, pred: dict(correct=int((truth == pred).sum())))
        protocol = dict(train_cases=['done', 'support'], evaluation_cases=['val'])
        prior = dict(case='done', split='training_support', correct=42)
        runtime.write(self.output/'evaluation.json', dict(records=[prior]))
        expected = self.engine.evaluate(original, self.model, self.processor, self.root, protocol, self.output)
        runtime.write(self.output/'evaluation.json', dict(records=[prior]))
        (self.output/'checkpoint_latest.pt').write_bytes(b'fixed checkpoint')
        runtime.write(self.output/'protocol.json', protocol)
        loads.clear()
        before_state = rng()
        RealAudit = fe.EvaluationAudit
        class AlwaysMeasure(RealAudit):
            def __init__(self, *args, **kw): super().__init__(*args, **kw, minimum_gain=0.)
        original_loop = self.engine.evaluate
        with patch.object(fe, 'EvaluationAudit', AlwaysMeasure):
            fe.install(self.engine, self.root/'durable')
            # Stop between volumes, then resume through the same frozen loop.
            original_loop.__globals__['stopped'] = lambda _: len(runtime.read(self.output/'evaluation.json')['records']) >= 2
            with self.assertRaises(SystemExit) as stop:
                self.engine.evaluate(original, self.model, self.processor, self.root, protocol, self.output)
            self.assertEqual(stop.exception.code, 75)
            self.assertEqual([r['case'] for r in runtime.read(self.output/'evaluation.json')['records']], ['done', 'support'])
            original_loop.__globals__['stopped'] = lambda _: False
            actual = self.engine.evaluate(original, self.model, self.processor, self.root, protocol, self.output)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], prior)
        self.assertEqual(loads, ['support', 'val'])
        self.assertTrue(fe.equal_state(before_state, rng()))
        self.assertIs(original.forward_plane, forward)
        self.assertIs(original.load_image, load)
        receipts = list((self.root/'durable/evaluation_extension/runs/task').glob('*.json'))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(runtime.read(receipts[0])['evaluated_volumes'], 3)
        self.assertEqual(runtime.read(receipts[0])['status'], 'fast')
        self.assertEqual((self.output/'checkpoint_latest.pt').read_bytes(), b'fixed checkpoint')


class PackageTests(unittest.TestCase):
    def test_complete_notebook_opt_in_and_unchanged_results(self):
        nb = fast_notebook({'medsam3_msd_data.zip': 'data', 'sam3.pt': 'base'}, 'package')
        code = '\n'.join(c['source'] for c in nb['cells'] if c['cell_type'] == 'code')
        for item in nb['cells']:
            if item['cell_type'] == 'code': ast.parse(item['source'])
        self.assertIn("'--fast-evaluation'", code)
        self.assertIn('results_matched200_noaug_108_v1', code)
        self.assertIn('/content/hippo_medsam3_parallel_fast_eval_v1', code)
        self.assertIn("campaign('preflight')", code)
        self.assertIn('MAX_NEW_RUNS = 108', code)
        self.assertNotIn('/content/hippo_medsam3_parallel_v1', code)

    def test_real_worker_installs_extension_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            engine = folder/'engine.py'
            engine.write_text('''import hashlib, json, sys
from pathlib import Path
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def evaluate(*args): pass
def train(): return 'original training'
def main():
    folder = Path(sys.argv[sys.argv.index('--durable')+1])
    (folder/'result.json').write_text(json.dumps(dict(evaluation=evaluate.__name__, training=train())))
if __name__ == '__main__': main()
''')
            for enabled, expected in (('0', 'evaluate'), ('1', 'audited_evaluate')):
                result = subprocess.run([sys.executable, parallel.__file__, '_worker', str(folder/'ready.json'),
                    str(engine), '--durable', str(folder)], env=dict(os.environ, HIPPO_AUDITED_FAST_EVAL=enabled),
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                value = json.loads((folder/'result.json').read_text())
                self.assertEqual(value, dict(evaluation=expected, training='original training'))


if __name__ == '__main__': unittest.main()
