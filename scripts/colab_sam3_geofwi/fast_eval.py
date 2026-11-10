"""Checkpoint-local, audited feature reuse around the frozen evaluation loop.

Training, calibration, image sizes, per-prompt decoders and volume metrics are
untouched. A full-volume comparison gates reuse, followed by one checked slice
per volume. Failed checks return to the original evaluator automatically.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

RTOL, ATOL = 1e-5, 1e-7


def clone_tree(value, memo=None):
    """Protect cached features from downstream mutation, retaining aliases."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif isinstance(value, dict):
        result = {k: clone_tree(v, memo) for k, v in value.items()}
    elif isinstance(value, list):
        result = [clone_tree(v, memo) for v in value]
    elif isinstance(value, tuple):
        result = tuple(clone_tree(v, memo) for v in value)
    elif value is None or isinstance(value, (str, int, float, bool)):
        return value
    else:
        raise TypeError(f'Unsupported feature type: {type(value).__name__}')
    memo[id(value)] = result
    return result


@contextmanager
def replace_attribute(obj, name, value):
    existed, previous = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


class FeatureReuse:
    """Cache the single salt prompt only within one frozen evaluation model."""
    def __init__(self, model, processor, forward):
        self.model, self.processor, self.forward = model, processor, forward
        self.text = {}
        self.counts = dict(text_encodes=0, text_hits=0)

    def clear(self):
        self.text.clear()

    def __call__(self, pixels):
        if torch.is_grad_enabled() or any(m.training for m in self.model.modules()):
            raise RuntimeError('Text reuse requires eval mode and inference mode')
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Text reuse requires frozen evaluation parameters')
        backbone = self.model.backbone
        original = backbone.forward_text
        def forward_text(captions, input_boxes=None, additional_text=None, device='cuda'):
            if input_boxes is not None or additional_text is not None or not isinstance(captions, (list, tuple)):
                return original(captions, input_boxes=input_boxes, additional_text=additional_text, device=device)
            key = (tuple(captions), str(device))
            if key in self.text:
                self.counts['text_hits'] += 1
                return clone_tree(self.text[key])
            value = original(captions, device=device)
            if len(self.text) < 1:
                self.text[key] = clone_tree(value)
            self.counts['text_encodes'] += 1
            return value
        with replace_attribute(backbone, 'forward_text', forward_text):
            return self.forward(self.model, self.processor, pixels)


class EvaluationAudit:
    def __init__(self, engine, original, model, processor, output, receipt, identity, minimum_gain=1.02):
        self.engine, self.original, self.model, self.processor = engine, original, model, processor
        self.output, self.receipt, self.identity = output, receipt, identity
        self.forward = original.forward_plane
        self.fast = FeatureReuse(model, processor, self.forward)
        self.minimum_gain = minimum_gain
        self.case, self.slice, self.depth = None, 0, 0
        self.mode = 'audit'
        self.data = dict(identity=identity, status='audit', full_volume=None, checked_slices=0,
                         max_log_probability_difference=0., reference_seconds=0., candidate_seconds=0.,
                         timed_slices=0, prediction_agreement=True, rng_agreement=True,
                         sentinels=[], tolerance=dict(rtol=RTOL, atol=ATOL), fallback_reason=None)
        if receipt.exists():
            old = engine.read(receipt)
            if old.get('identity') == identity and old.get('status') in ('fast', 'original'):
                self.data = old
                self.mode = old['status']
                print(f'EVAL: reusing {self.mode} decision for this exact checkpoint.', flush=True)

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def save(self):
        self.data.update(status=self.mode, counts=self.fast.counts, updated_at=time.time())
        self.engine.write(self.receipt, self.data)

    def begin_volume(self, case, depth):
        self.case, self.slice, self.depth = case, 0, depth
        if self.mode == 'audit':
            self.data['full_volume'] = dict(case=case, slices=depth, checked=0)
            print(f'EVAL AUDIT {case}: checking all {depth} slices; metrics use original predictions.', flush=True)

    def fallback(self, reason):
        self.mode = 'original'
        self.data['fallback_reason'] = reason
        self.fast.clear()
        self.save()
        print(f'EVAL: using original path. {reason}', flush=True)

    def checked(self, pixels):
        before = self.engine.rng_state()
        self.sync(); start = time.perf_counter()
        reference = self.forward(self.model, self.processor, pixels)
        self.sync(); reference_seconds = time.perf_counter()-start
        after = self.engine.rng_state()
        self.engine.set_rng(before)
        try:
            self.sync(); start = time.perf_counter()
            candidate = self.fast(pixels)
            self.sync(); candidate_seconds = time.perf_counter()-start
            candidate_state = self.engine.rng_state()
            if not equal_state(before, after) or not equal_state(after, candidate_state):
                self.data['rng_agreement'] = False
                raise ValueError('Evaluation consumed RNG; feature reuse was disabled.')
            if reference[1] is not None or candidate[1] is not None:
                raise ValueError('Evaluation unexpectedly returned a training loss.')
            a, b = reference[0], candidate[0]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Nonfinite audit output')
            difference = float((a-b).abs().max())
            self.data['max_log_probability_difference'] = max(self.data['max_log_probability_difference'], difference)
            # Compare the same exp/argmax used by the frozen volume evaluator.
            if not torch.equal(a.exp().argmax(0), b.exp().argmax(0)):
                self.data['prediction_agreement'] = False
                raise ValueError('Predicted labels changed.')
            torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
            self.data['checked_slices'] += 1
            if self.mode == 'audit':
                self.data['full_volume']['checked'] += 1
                # Exclude first two slices from timing to avoid cache/compile warm-up.
                if self.slice >= 2:
                    self.data['reference_seconds'] += reference_seconds
                    self.data['candidate_seconds'] += candidate_seconds
                    self.data['timed_slices'] += 1
            else:
                self.data['sentinels'].append(dict(case=self.case, slice=self.slice, max_difference=difference))
        except Exception as error:
            self.fallback(f'{type(error).__name__}: {str(error)[:1200]}')
        finally:
            self.engine.set_rng(after)
        # All pilot and sentinel predictions remain the original predictions.
        return reference

    def plane(self, model, processor, pixels, labels=None, criterion=None):
        if model is not self.model or processor is not self.processor or labels is not None or criterion is not None:
            raise ValueError('Evaluation wrapper called outside its fixed checkpoint/prompt scope.')
        if self.mode == 'audit' or (self.mode == 'fast' and self.slice == 0):
            result = self.checked(pixels)
        elif self.mode == 'fast':
            before = self.engine.rng_state()
            try:
                result = self.fast(pixels)
            except Exception as error:
                self.engine.set_rng(before)
                self.fallback(f'{type(error).__name__}: {str(error)[:1200]}')
                result = self.forward(model, processor, pixels)
        else:
            result = self.forward(model, processor, pixels)
        self.slice += 1
        if self.mode == 'audit' and self.slice == self.depth:
            ratio = self.data['reference_seconds']/max(self.data['candidate_seconds'], 1e-12)
            self.data['observed_slice_speedup'] = ratio
            if self.data['timed_slices'] and ratio >= self.minimum_gain:
                self.mode = 'fast'
                self.save()
                print(f'EVAL FAST: full-volume labels/log-probabilities passed; observed slice speedup {ratio:.2f}x.', flush=True)
            else:
                self.fallback(f'No measured slice-time improvement (observed {ratio:.2f}x).')
        # Local progress is cheap; durable receipts are updated once per volume.
        self.engine.write(self.output/'evaluation_runtime.json', dict(case=self.case, slice=self.slice,
            slices=self.depth, backend=self.mode, checkpoint_sha256=self.identity['checkpoint_sha256']))
        if self.slice == self.depth:
            self.save()
        return result
