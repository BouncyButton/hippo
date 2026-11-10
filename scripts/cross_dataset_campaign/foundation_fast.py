"""Equivalent runtime optimizations for the six-run foundation comparison."""
from collections import OrderedDict
from contextlib import contextmanager
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import numpy as np
import torch

from scripts.cross_dataset_campaign import foundation as original
from scripts.cross_dataset_campaign.common import read, write, digest, stopping

REFERENCE_PLANE = original.forward_plane
REFERENCE_SLAB = types.FunctionType(original.forward_slab.__code__,
    {**original.forward_slab.__globals__, 'forward_plane': REFERENCE_PLANE})
REFERENCE_LOAD = original.load_foundation
REFERENCE_EVALUATE = original.evaluate
REFERENCE_SUPPORT = original.SupportSlabs


class TransformCache:
    """Bounded CPU cache of deterministic transforms; callers receive a copy."""
    def __init__(self, transform, max_bytes=8 * 1024**3):
        self.transform, self.max_bytes = transform, max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.enabled = True

    def __call__(self, value):
        if not self.enabled:
            return self.transform(value)
        key = (tuple(value.shape), str(value.dtype),
               hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).digest())
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key].clone()
        self.misses += 1
        result = self.transform(value)
        if result.device.type != 'cpu':
            raise ValueError('Preprocessing cache must remain on CPU')
        cost = result.numel() * result.element_size()
        while self.entries and self.bytes + cost > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.numel() * removed.element_size()
        if cost <= self.max_bytes:
            self.entries[key] = result.clone()
            self.bytes += cost
        return result

    def clear(self):
        self.entries.clear()
        self.bytes = 0


def gram_matrix(gradients):
    """Keep each original FP64 dot and Python summation order; transfer once."""
    terms, lengths = [], []
    for left in gradients:
        for right in gradients:
            values = [(a.detach().double()*b.detach().double()).sum()
                      for a, b in zip(left, right) if a is not None and b is not None]
            lengths.append(len(values))
            terms.extend(values)
    values = torch.stack(terms).cpu().tolist() if terms else []
    result, offset = [], 0
    for size in lengths:
        result.append(sum(values[offset:offset+size]))
        offset += size
    n = len(gradients)
    return [result[i*n:(i+1)*n] for i in range(n)]


class SupportSlabs(REFERENCE_SUPPORT):
    def __init__(self, volumes, protocol, processor):
        super().__init__(volumes, protocol, processor)
        if torch.cuda.is_available():
            for case, supports in self.supports.items():
                self.supports[case] = {c: replace(s, foreground=s.foreground.cuda(),
                    inner=s.inner.cuda(), outer=s.outer.cuda()) for c, s in supports.items()}


def forward_plane(model, processor, pixels, protocol, labels=None, criterion=None):
    if criterion is None or protocol['arm'] != 'dice':
        return REFERENCE_PLANE(model, processor, pixels, protocol, labels, criterion)
    # The baseline only needs native losses. The auxiliary semantic reductions
    # have no effect on its objective, optimizer, model calls, or random state.
    from types import SimpleNamespace
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    losses = []
    for class_id in range(1, len(protocol['classes'])):
        view = SimpleNamespace(samples=[(pixels, labels == class_id, '', 0)],
            transform=processor.transform, resolution=processor.resolution)
        point = original.TrainingSlices.__getitem__(view, 0)
        point.find_queries[0].query_text = protocol['prompts'][str(class_id)]
        batch = collate_fn_api([point], dict_key='input', with_seg_masks=True)['input']
        for target in batch.find_targets:
            if target.segments.numel() == 0:
                target.segments = torch.empty((0, processor.resolution, processor.resolution), dtype=torch.bool)
        batch = original.move_to_device(batch, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch)
            losses.append(criterion(outputs, [model.back_convert(t) for t in batch.find_targets])[CORE_LOSS_KEY])
    return None, torch.stack(losses).mean()


def forward_slab(model, criterion, slabs, case, anchor):
    indices = slabs.indices[case]
    planes, natives = [], []
    for z in (anchor, (anchor+1) % len(indices)):
        pixels, labels = slabs.samples[indices[z]]
        lp, native = forward_plane(model, slabs.processor, pixels, slabs.protocol, labels, criterion)
        planes.append(lp)
        natives.append(native)
    result = dict(native=torch.stack(natives).mean())
    if slabs.protocol['arm'] == 'dice':
        return result
    band, edge = [], []
    for c, support in slabs.supports[case].items():
        others = [i for i in range(len(planes[0])) if i != c]
        b, e = original.sampled_losses(planes[0][c].exp(), planes[1][c].exp(), support, anchor,
            log_probability=planes[0][c], log_complement=torch.logsumexp(planes[0][others], 0))
        band.append(b)
        edge.append(e)
    result['bands'] = torch.stack(band).mean()
    if 'edge' in slabs.protocol['arm']:
        result['edge'] = torch.stack(edge).mean()
    return result


@contextmanager
def validation_sampling():
    """Use repeatable sampling backward only while checking equivalence.

    CUDA grid_sample backward uses nondeterministic accumulation, even for
    repeated original-model calls with the same seed. CPU FP32 sampling keeps
    this strict comparison meaningful; all training retains the original CUDA
    implementation, autocast, loss graph and random state.
    """
    functional = torch.nn.functional
    reference = functional.grid_sample
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def sample(input, grid, *args, **kwargs):
        if input.device.type != 'cuda':
            return reference(input, grid, *args, **kwargs)
        dtype = torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
        return reference(input.to(device='cpu', dtype=dtype),
            grid.to(device='cpu', dtype=dtype), *args, **kwargs).to(input.device)

    functional.grid_sample = sample
    try:
        # This check runs in a separate process with its own cuBLAS workspace.
        # Flash Attention requires strict mode to select deterministic backward.
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        functional.grid_sample = reference
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def runtime_validation(model, criterion, parameters, slabs, protocol, output):
    sampler = torch.nn.functional.grid_sample
    settings = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    with validation_sampling():
        result = _runtime_validation(model, criterion, parameters, slabs, protocol, output)
    if (torch.nn.functional.grid_sample is not sampler or settings !=
            (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())):
        raise ValueError('Validation failed to restore the training runtime')
    if output is not None:
        receipt = read(output/'runtime_validation.json')
        write(output/'runtime_validation.json', {**receipt, 'training_runtime_restored': True,
            'training_deterministic_algorithms': settings[0], 'training_warn_only': settings[1]})
    return result


def _runtime_validation(model, criterion, parameters, slabs, protocol, output):
    """Compare real-model losses/gradients before any scientific updates."""
    cache = slabs.processor.transform
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
    arm_before = slabs.protocol['arm']
    case, anchor = slabs.calibration_schedule(4)[0]
    checks = []
    started = time.monotonic()
    try:
        for arm in ('dice', 'dice_bands', 'dice_bands_edge'):
            slabs.protocol['arm'] = arm
            names = ['native'] + [x for x in ('bands', 'edge') if x in arm]
            cache.enabled = False
            original.reset_seed(7319)
            losses = REFERENCE_SLAB(model, criterion, slabs, case, anchor)
            reference = {n: float(losses[n].detach()) for n in names}
            grads = []
            for i, name in enumerate(names):
                gradients = torch.autograd.grad(losses[name], parameters, retain_graph=i < len(names)-1, allow_unused=True)
                grads.append([None if g is None else g.detach().cpu() for g in gradients])
            del losses, gradients
            cache.enabled = True
            original.reset_seed(7319)
            losses = forward_slab(model, criterion, slabs, case, anchor)
            exact = True
            for i, name in enumerate(names):
                actual = float(losses[name].detach())
                if not np.isclose(actual, reference[name], rtol=1e-6, atol=1e-7):
                    raise ValueError(f'Optimized {arm}/{name} loss differs')
                gradients = torch.autograd.grad(losses[name], parameters, retain_graph=i < len(names)-1, allow_unused=True)
                for a, b in zip(grads[i], gradients):
                    if (a is None) != (b is None):
                        raise ValueError('Optimized parameter gradient connectivity differs')
                    if a is not None:
                        b = b.detach().cpu()
                        exact = exact and torch.equal(a, b)
                        if not torch.allclose(a, b, rtol=1e-5, atol=1e-7):
                            raise ValueError(f'Optimized {arm}/{name} gradient differs')
            checks.append(dict(arm=arm, losses=reference, gradients_exact=exact, status='passed'))
            del grads, gradients, losses
        write(output/'runtime_validation.json', dict(status='passed', checks=checks,
            seconds=time.monotonic()-started, source_sha256=digest(__file__),
            protocol_sha256=digest(output/'protocol.json'), torch=torch.__version__,
            validation_sampling='CPU FP32 grid_sample for the equivalence check only; original CUDA sampler restored before calibration/training',
            validation_deterministic_algorithms=True,
            note='Real CUDA model loss/gradient equivalence gate with deterministic sampling reference; not an end-to-end speed benchmark'))
    finally:
        slabs.protocol['arm'] = arm_before
        cache.enabled = True
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
        random.setstate(rng[2]); np.random.set_state(rng[3])


def calibrate(model, criterion, parameters, slabs, protocol, arm, output):
    require_validation(output)
    names = ['native'] + [k for k in ('bands', 'edge') if k in arm]
    if len(names) == 1:
        return dict(weights={}, status='not_required')
    original.reset_seed(protocol['calibration_seed'])
    path = output/'calibration_partial.json'
    rows = read(path)['rows'] if path.exists() else []
    for index, (case, anchor) in enumerate(slabs.calibration_schedule(protocol['calibration_slabs_per_volume'])):
        if index < len(rows):
            continue
        if stopping():
            raise SystemExit(75)
        original.reset_seed(protocol['calibration_seed']+index)
        losses = forward_slab(model, criterion, slabs, case, anchor)
        gradients = [torch.autograd.grad(losses[name], parameters, retain_graph=i < len(names)-1, allow_unused=True)
                     for i, name in enumerate(names)]
        gram = gram_matrix(gradients)
        if not np.isfinite(gram).all() or gram[0][0] <= 0:
            raise ValueError('Invalid calibration gradient')
        rows.append(dict(case=case, anchor=anchor, gram=gram,
            losses={k: float(losses[k].detach()) for k in names}))
        write(path, dict(names=names, rows=rows))
        del gradients, losses
    coefficients, statistics, weights = [1.], {}, {}
    for i, name in enumerate(names[1:], 1):
        ratios = []
        for row in rows:
            gram = np.asarray(row['gram'])
            norm = np.asarray(coefficients) @ gram[:i, :i] @ np.asarray(coefficients)
            if norm > 0 and gram[i, i] > 0:
                ratios.append(np.sqrt(gram[i, i]/norm))
        if len(ratios) < 5:
            raise ValueError('Too few nonzero calibration gradients')
        weight = min(protocol['gradient_ratio_target']/np.median(ratios),
                     protocol['gradient_ratio_p95_cap']/np.quantile(ratios, .95))
        weights[name] = float(weight); coefficients.append(weight)
        statistics[name] = dict(valid=len(ratios), weighted_median=float(np.median(ratios)*weight),
                                weighted_p95=float(np.quantile(ratios, .95)*weight))
    return dict(status='complete', weights=weights, statistics=statistics, rows=rows, names=names,
        scope='Training support only; sequential LoRA gradient norm calibration',
        target=protocol['gradient_ratio_target'], p95_cap=protocol['gradient_ratio_p95_cap'])


def load_foundation(protocol, base_weights):
    model, processor = REFERENCE_LOAD(protocol, base_weights)
    processor.transform = TransformCache(processor.transform)
    return model, processor


def evaluate(model, processor, volumes, protocol, output):
    cache = processor.transform
    write(output/'runtime_cache.json', dict(hits=cache.hits, misses=cache.misses,
        retained_bytes=cache.bytes, maximum_bytes=cache.max_bytes,
        policy='Exact CPU image transforms only; no learned feature cache; released before validation'))
    cache.clear(); cache.enabled = False
    return REFERENCE_EVALUATE(model, processor, volumes, protocol, output)


def require_validation(output):
    receipt = read(output/'runtime_validation.json')
    if (receipt.get('status') != 'passed' or not receipt.get('isolated_process')
            or not receipt.get('training_runtime_restored')
            or receipt.get('source_sha256') != digest(__file__)
            or receipt.get('protocol_sha256') != digest(output/'protocol.json')):
        raise ValueError('No matching isolated numerical validation')
    return receipt


def validation_environment():
    return {**os.environ, 'HIPPO_FOUNDATION_VALIDATE_ONLY': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def validate_only(args):
    protocol = original.read(args.protocol)
    original.check_no_augmentation(protocol)
    actual = {**protocol, 'arm': args.arm}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'protocol.json'
    if path.exists() and read(path) != actual:
        raise ValueError('Validation protocol changed')
    write(path, actual)
    torch.set_num_threads(4)
    original.reset_seed(protocol['seed'])
    model, processor = load_foundation(protocol, args.base_weights)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original.is_adapter(name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 18497664:
        raise ValueError('Unexpected LoRA architecture')
    criterion = original.native_loss(model)
    slabs = SupportSlabs(original.Volumes(actual), actual, processor)
    model.train()
    runtime_validation(model, criterion, parameters, slabs, protocol, args.output)
    receipt = read(args.output/'runtime_validation.json')
    write(args.output/'runtime_validation.json', {**receipt, 'isolated_process': True,
        'validation_python': sys.executable,
        'validation_cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
    print('Isolated strict numerical validation passed for all three arms.', flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ('protocol', 'output', 'base-weights'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', required=True)
    args, _ = parser.parse_known_args()
    if os.environ.get('HIPPO_FOUNDATION_VALIDATE_ONLY') == '1':
        validate_only(args)
        return
    try:
        require_validation(args.output)
    except (FileNotFoundError, ValueError, KeyError):
        subprocess.run([sys.executable, '-m', 'scripts.cross_dataset_campaign.foundation_fast',
                        *sys.argv[1:]], env=validation_environment(), check=True)
        require_validation(args.output)
    write(args.output/'runtime_training_environment.json', dict(python=sys.executable,
        torch=torch.__version__, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), validation_isolated=True))
    original.load_foundation = load_foundation
    original.SupportSlabs = SupportSlabs
    original.forward_plane = forward_plane
    original.forward_slab = forward_slab
    original.calibrate = calibrate
    original.evaluate = evaluate
    original.main()


if __name__ == '__main__':
    main()
