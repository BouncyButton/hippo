"""Stateless, training-only intensity augmentation for the paired HFH pilot."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json

import numpy as np

SPEC = dict(name='mild_intensity_v1', gamma_range=[.9, 1.1], gamma_probability=.5,
            gain_range=[.9, 1.1], gain_probability=.5, noise_std_range=[0., .015],
            noise_probability=.2, preserve_zero_pixels=True, seed=271828,
            spatial='none', evaluation='unaugmented', calibration='reuse_paired_control')


def augmentation_seed(protocol, update, micro, case, anchor):
    identity = [SPEC['seed'], protocol['seed'], protocol['fold'], update, micro, case, anchor]
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:8], 'little')


def transform_pair(images, seed):
    """Share contrast/gain across adjacent slices; sample independent pixel noise."""
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(*SPEC['gamma_range']) if rng.random() < SPEC['gamma_probability'] else 1.
    gain = rng.uniform(*SPEC['gain_range']) if rng.random() < SPEC['gain_probability'] else 1.
    sigma = rng.uniform(*SPEC['noise_std_range']) if rng.random() < SPEC['noise_probability'] else 0.
    outputs = []
    for image in images:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError('Expected original uint8 image plane')
        value = (image.astype(np.float32)/255.) ** gamma * gain
        if sigma:
            value += rng.normal(0., sigma, image.shape).astype(np.float32)
        value[image == 0] = 0.
        outputs.append(np.round(np.clip(value, 0., 1.)*255).astype(np.uint8))
    return outputs, dict(gamma=float(gamma), gain=float(gain), noise_std=float(sigma))


@contextmanager
def augmented_samples(slabs, case, anchor, seed):
    indices = slabs.indices[case]
    ids = [indices[anchor], indices[(anchor+1) % len(indices)]]
    if len(set(ids)) != 2:
        raise ValueError('Adjacent pair must contain two distinct slices')
    saved = [slabs.samples[i] for i in ids]
    changed, params = transform_pair([sample[0] for sample in saved], seed)
    try:
        for i, pixels, (_, labels) in zip(ids, changed, saved):
            slabs.samples[i] = (pixels, labels)
        yield params
    finally:
        for i, sample in zip(ids, saved):
            slabs.samples[i] = sample


def training_source(source):
    old = 'losses=forward_slab(model,criterion,slabs,case,anchor)'
    if source.count(old) != 1 or source.count("augmentation='none'") != 1:
        raise ValueError('Audited training loop changed')
    return source.replace(old, 'losses=hfh_augmented_forward(model,criterion,slabs,case,anchor,update,micro)').replace(
        "augmentation='none'", "augmentation=protocol['augmentation']")


def install(original, protocol, output):
    """Keep model RNG, labels, geometry, calibration and evaluation unchanged."""
    if protocol.get('augmentation') != SPEC or protocol['updates'] != 125:
        raise ValueError('Only the declared 125-update augmentation pilot is allowed')
    original.check_no_augmentation = lambda p: check_protocol(p)
    reference = original.forward_slab

    def augmented_forward(model, criterion, slabs, case, anchor, update, micro):
        seed = augmentation_seed(protocol, update, micro, case, anchor)
        # A random image per sample defeats the transform cache. Disable retention
        # for this call only, avoiding gigabytes of one-use CPU tensors.
        cache = slabs.processor.transform
        enabled = getattr(cache, 'enabled', None)
        try:
            if enabled is not None:
                cache.enabled = False
            with augmented_samples(slabs, case, anchor, seed):
                return reference(model, criterion, slabs, case, anchor)
        finally:
            if enabled is not None:
                cache.enabled = enabled

    original.__dict__['hfh_augmented_forward'] = augmented_forward
    exec(compile(training_source(inspect.getsource(original.train)), __file__, 'exec'), original.__dict__)
    tape = []
    for index, (case, anchor) in enumerate(protocol['frozen_sample_schedule']):
        update, micro = divmod(index, protocol['accumulation'])
        seed = augmentation_seed(protocol, update, micro, case, anchor)
        _, params = transform_pair([np.zeros((1, 1), np.uint8)]*2, seed)
        tape.append(dict(update=update+1, micro=micro, case=case, anchor=anchor, seed=seed, **params))
    original.write(output/'augmentation_schedule.json', tape)

    # Keep the wrapper outside completed-checkpoint recovery so its summary
    # cannot incorrectly label an augmented completed run as unaugmented.
    train = original.train
    def recorded_train(*args):
        result = train(*args)
        path = output/'training_summary.json'
        summary = original.read(path)
        summary['augmentation'] = deepcopy(SPEC)
        original.write(path, summary)
        return result
    original.train = recorded_train


def check_protocol(protocol):
    if (protocol.get('augmentation') != SPEC or protocol['updates'] != 125
            or (protocol['fold'], protocol['seed']) != (0, 0)):
        raise ValueError('Augmentation pilot scope differs')


def install_calibration(original, protocol, output):
    """Reuse audited per-arm weights; do not mix an augmentation and weight ablation."""
    from pathlib import Path
    from scripts.colab_hfh.medsam_constraints_campaign import zip_read
    archive = Path(protocol['augmentation_control_archive'])
    if original.digest(archive) != protocol['augmentation_control_sha256']:
        raise ValueError('Unaugmented control archive differs')
    calibration = zip_read(archive, 'calibration.json')
    def calibrate(model, criterion, parameters, slabs, current, arm, destination):
        from scripts.cross_dataset_campaign.adni_foundation_full_runtime import native_training_check
        native_training_check(model, criterion, parameters, slabs, current, destination)
        original.write(output/'augmentation_calibration_reuse.json', dict(
            control_sha256=protocol['augmentation_control_sha256'], weights=calibration['weights'],
            optimizer_updates=0, method='Exact saved control coefficients; calibration images unaugmented'))
        return deepcopy(calibration)
    original.calibrate = calibrate
