"""Offline candidate checks; these do not establish real SAM3 GPU equivalence."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import typing
import unittest

import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F

from scripts.colab_medsam3.fast_eval import FeatureReuse
from scripts.colab_medsam3.investigate_performance import BUNDLE, same
from scripts.colab_medsam3.performance_candidates import forward_batch, packed_cpu_snapshot, split_semantic_outputs


def native_definitions():
    """Execute frozen data/metric definitions without importing the CUDA model."""
    filename = BUNDLE/'upstream/sam3/model/data_misc.py'
    spec = importlib.util.spec_from_file_location('audit_native_data_misc', filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = dict(vars(module), **vars(typing))
    namespace.update(torch=torch, F=F, PILImage=PILImage, dataclass=dataclass)
    filename = BUNDLE/'upstream/sam3/train/data/sam3_image_dataset.py'
    names = {'InferenceMetadata', 'FindQuery', 'FindQueryLoaded', 'Object', 'Image', 'Datapoint'}
    definitions = [n for n in ast.parse(filename.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(filename), 'exec'), namespace)
    filename = BUNDLE/'upstream/sam3/train/data/collator.py'
    definitions = [n for n in ast.parse(filename.read_text()).body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(filename), 'exec'), namespace)
    for relative, names in (
        ('source/scripts/medsam3_supervised_constraints.py', {'semantic_log_fields'}),
        ('source/scripts/train_medsam3_multiclass.py', {'exclusive_log_probabilities'}),
        ('source/scripts/train_medsam3_few_shot.py', {'move_to_device'}),
    ):
        filename = BUNDLE/relative
        definitions = [n for n in ast.parse(filename.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(filename), 'exec'), namespace)
    return namespace


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        parameters = [torch.nn.Parameter(torch.arange(12.).reshape(3, 4)/10), torch.nn.Parameter(torch.ones(5))]
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=.01)
        for parameter in parameters: parameter.grad = torch.full_like(parameter, .2)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        return parameters, optimizer

    def test_bare_optimizer_state_is_not_an_immutable_snapshot(self):
        parameters, optimizer = self.fixture()
        state = optimizer.state_dict()
        fixed = packed_cpu_snapshot(state)
        before = fixed['state'][0]['exp_avg'].clone()
        optimizer.state[parameters[0]]['exp_avg'].add_(1)
        self.assertFalse(torch.equal(state['state'][0]['exp_avg'], before))
        self.assertTrue(torch.equal(fixed['state'][0]['exp_avg'], before))

    def test_packed_values_and_rng_exact_without_mutating_original(self):
        _, optimizer = self.fixture()
        state = optimizer.state_dict()
        before = deepcopy(state)
        rng = torch.get_rng_state()
        actual = packed_cpu_snapshot(state)
        self.assertTrue(same(before, actual))
        self.assertTrue(same(before, state))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        first, second = actual['state'][0]['exp_avg'], actual['state'][0]['exp_avg_sq']
        second_before = second.clone()
        first.add_(9)
        self.assertTrue(torch.equal(second, second_before))
        self.assertTrue(same(state, before))

    def test_unsupported_storage_aliases_are_rejected(self):
        value = torch.arange(12.)
        with self.assertRaisesRegex(ValueError, 'alias audit'):
            packed_cpu_snapshot(dict(left=value[:6], right=value[3:9]))
        actual = packed_cpu_snapshot(dict(left=value, right=value))
        self.assertIs(actual['left'], actual['right'])

    def test_both_formats_resume_identical_next_optimizer_update(self):
        parameters, optimizer = self.fixture()
        state = dict(adapter=[p.detach().clone() for p in parameters], optimizer=optimizer.state_dict(), rng=torch.get_rng_state())
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for modern in (True, False):
                path = Path(directory)/f'{modern}.pt'
                torch.save(state, path, _use_new_zipfile_serialization=modern)
                recovered = torch.load(path, map_location='cpu', weights_only=False)
                self.assertTrue(same(state, recovered))
                resumed = [torch.nn.Parameter(value.clone()) for value in recovered['adapter']]
                opt = torch.optim.AdamW(resumed, lr=5e-5, weight_decay=.01)
                opt.load_state_dict(recovered['optimizer'])
                for parameter in resumed: parameter.grad = torch.full_like(parameter, .37)
                torch.nn.utils.clip_grad_norm_(resumed, 1.)
                opt.step()
                results.append(dict(adapter=[p.detach().clone() for p in resumed], optimizer=opt.state_dict()))
        self.assertTrue(same(results[0], results[1]))

    def test_atomic_rename_preserves_recovery_hardlink_in_both_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for modern in (True, False):
                latest, snapshot, temporary = root/'latest.pt', root/'snapshot.pt', root/'temporary.pt'
                torch.save(dict(step=84, value=torch.arange(4.)), latest, _use_new_zipfile_serialization=modern)
                os.link(latest, snapshot)
                before = snapshot.read_bytes()
                torch.save(dict(step=85, value=torch.ones(4)), temporary, _use_new_zipfile_serialization=modern)
                temporary.replace(latest)
                self.assertEqual(snapshot.read_bytes(), before)
                self.assertEqual(torch.load(snapshot, weights_only=False)['step'], 84)
                self.assertEqual(torch.load(latest, weights_only=False)['step'], 85)
                snapshot.unlink()


class MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.image_calls = 0
    def forward_image(self, samples):
        self.image_calls += 1
        return dict(features=samples[:, 0, 0, 0].clone())
    def forward_text(self, captions, input_boxes=None, additional_text=None, device='cpu'):
        return dict(features=torch.tensor([0.25 if text.startswith('anterior') else -.25 for text in captions]))


class MockModel(torch.nn.Module):
    """Only tests batch routing. It is deliberately not a SAM3 performance proxy."""
    def __init__(self):
        super().__init__(); self.backbone = MockBackbone()
    def forward(self, batch):
        image = self.backbone.forward_image(batch.img_batch)['features']
        text = self.backbone.forward_text(batch.find_text_batch, device='cpu')['features']
        stage = batch.find_inputs[0]
        v = image[stage.img_ids]+text[stage.text_ids]
        masks = v[:, None, None, None]+torch.arange(3*5*5).view(1, 3, 5, 5)/32
        output = dict(pred_masks=masks, pred_logits=v[:, None, None]+torch.arange(3).view(1, 3, 1)/8,
                      presence_logit_dec=v[:, None])
        return SimpleNamespace(output=[[output]])


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.native = native_definitions()

    def fixture(self):
        n = self.native
        def datapoint(pixels, labels, class_id, processor):
            self.assertIsNone(labels)
            tensor = processor.transform(torch.from_numpy(pixels.copy()).expand(3, *pixels.shape))
            query = n['FindQueryLoaded'](query_text=('anterior hippocampus' if class_id == 1 else 'posterior hippocampus'),
                image_id=0, object_ids_output=[], is_exhaustive=True, input_bbox=None, input_points=None,
                inference_metadata=n['InferenceMetadata'](0, 0, 1, pixels.shape, -1, -1))
            return n['Datapoint'](find_queries=[query], images=[n['Image'](tensor, [], pixels.shape)])
        original = SimpleNamespace(datapoint=datapoint, move_to_device=n['move_to_device'],
            semantic_log_fields=n['semantic_log_fields'], exclusive_log_probabilities=n['exclusive_log_probabilities'])
        return original, MockModel().eval(), SimpleNamespace(transform=lambda tensor: tensor.float(), resolution=8)

    def test_actual_collator_preserves_slice_and_prompt_indices(self):
        original, _, processor = self.fixture()
        points = [original.datapoint(np.full((8, 8), i, dtype=np.uint8), None, 1, processor) for i in range(4)]
        batch = self.native['collate_fn_api'](points, 'input', with_seg_masks=True)['input']
        self.assertEqual(batch.find_inputs[0].img_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(batch.find_inputs[0].text_ids.tolist(), [0, 0, 0, 0])
        self.assertEqual(batch.find_text_batch, ['anterior hippocampus'])
        self.assertEqual(batch.img_batch[:, 0, 0, 0].tolist(), [0, 1, 2, 3])

    def test_batch_1_2_4_and_uneven_tail_keep_original_postprocessing(self):
        original, model, processor = self.fixture()
        images = [np.full((8, 8), i, dtype=np.uint8) for i in range(7)]
        call = lambda m, p, batch: forward_batch(original, m, p, batch, collate=self.native['collate_fn_api'], device='cpu')
        with torch.inference_mode():
            reference = [call(model, processor, [im])[0][0] for im in images]
            for size in (1, 2, 4):
                cached = FeatureReuse(model, processor, call)
                actual = []
                for offset in range(0, len(images), size): actual.extend(cached(images[offset:offset+size])[0])
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                    self.assertTrue(torch.equal(a.exp().argmax(0), b.exp().argmax(0)))
                self.assertEqual(cached.counts['image_encodes'], (len(images)+size-1)//size)
                self.assertEqual(cached.counts['image_hits'], (len(images)+size-1)//size)

    def test_batched_outputs_cannot_silently_drop_or_mix_slices(self):
        output = dict(pred_masks=torch.zeros(2, 3, 4, 4), pred_logits=torch.zeros(2, 3, 1), presence_logit_dec=torch.zeros(1, 1))
        with self.assertRaisesRegex(ValueError, 'presence_logit_dec'):
            split_semantic_outputs(output, 2)

    def test_candidate_cannot_enter_training(self):
        original, model, processor = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'evaluation only'):
            forward_batch(original, model.train(), processor, [np.zeros((8, 8), dtype=np.uint8)])


if __name__ == '__main__': unittest.main()
