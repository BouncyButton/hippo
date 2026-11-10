"""Stable CPU inverse-CDF evaluation inside the unchanged Swin hash/RNG gates.

This changes only the scratch values checked before the frozen initializer
copies its canonical position biases. It never changes the accepted weights,
the tolerance, the random draws, or any training/restore operation.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import math
from pathlib import Path

RECEIPT_NAME = 'initialization_compat_v2.json'


def stable_trunc_normal(tensor, mean, std, a, b):
    import numpy as np
    import torch
    from scipy.special import erfinv

    if (tensor.device.type != 'cpu' or tensor.dtype != torch.float32
            or (mean, std, a, b) != (0., .02, -2., 2.)):
        raise ValueError('Unsupported Swin CPU initialization recipe')
    # Keep MONAI's float32 uniform draw and every subsequent rounding point.
    # Only erfinv is evaluated in float64, then rounded back to float32.
    lower = (1. + math.erf((a - mean) / std / math.sqrt(2.))) / 2.
    upper = (1. + math.erf((b - mean) / std / math.sqrt(2.))) / 2.
    with torch.no_grad():
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        inverse = erfinv(tensor.detach().numpy().astype(np.float64)).astype(np.float32)
        tensor.copy_(torch.from_numpy(inverse))
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


@contextmanager
def stable_swin_inverse_cdf():
    from monai.networks.layers import weight_init
    previous = weight_init._no_grad_trunc_normal_
    weight_init._no_grad_trunc_normal_ = stable_trunc_normal
    try:
        yield
    finally:
        weight_init._no_grad_trunc_normal_ = previous


def install(runtime):
    """Wrap only Swin; leave the frozen checker and other architectures intact."""
    previous = runtime.canonical_initialization
    if getattr(previous, '_adni_stable_swin', False):
        return

    @wraps(previous)
    def canonical(builder, protocol, architecture, reference):
        if protocol.get('model_identity') != 'swinunetr':
            return previous(builder, protocol, architecture, reference)
        with stable_swin_inverse_cdf():
            model, audit = previous(builder, protocol, architecture, reference)
        audit = dict(audit, initialization_compat=dict(
            method='float32_uniform_float64_scipy_erfinv_float32_rounding',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            tolerance_unchanged=True, canonical_model_hash_checked=True,
            incoming_and_outgoing_rng_checked=True))
        return model, audit

    canonical._adni_stable_swin = True
    runtime.canonical_initialization = canonical


def verify_receipt(runtime, control, root, durable, binding):
    """Additional CPU compatibility gate; the original GPU gate still runs."""
    import json
    control, root, durable = map(Path, (control, root, durable))
    receipt = runtime.read(durable / RECEIPT_NAME)
    expected_seeds = [4] if runtime.read(root/'plan.json')['seeds'] == [4] else [0, 1]
    if (receipt.get('status') != 'passed_cpu_initialization'
            or receipt.get('binding') != binding
            or receipt.get('control_sha256') != runtime.digest(control/'control_manifest.json')
            or receipt.get('compat_sha256') != runtime.digest(control/'initialization_compat.py')
            or receipt.get('seeds') != expected_seeds
            or receipt.get('optimizer_updates') != 0):
        raise RuntimeError('Run this corrected notebook\'s CPU initialization preflight; no results were reset.')
    reference = runtime.read(root/'initialization_reference.json')['models']['swinunetr']
    checks = receipt.get('checks', [])
    if len(checks) != len(expected_seeds):
        raise ValueError('Incomplete CPU initialization checks')
    for row, seed in zip(checks, expected_seeds):
        expected = reference[str(seed)]
        if (row.get('seed') != seed or row.get('model_sha256') != expected['model_sha256']
                or row.get('rng_after_model') != expected['rng_after_model']
                or row.get('portable_calls') != len(expected['calls'])
                or row.get('folds_checked') != [0, 1, 2]):
            raise ValueError('CPU initialization receipt differs from the frozen reference')
    return receipt
