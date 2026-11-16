"""Archived HFH gain augmentation used for training slabs."""
import numpy as np
import hashlib
import json
SPEC = BASE_SPEC = dict(seed=271828, gamma_range=[.9,1.1], gamma_probability=.5, gain_range=[.9,1.1],gain_probability=.5,noise_std_range=[0.,.015],noise_probability=.2)
COMPONENTS=("gain", "gamma", "noise")
def augmentation_seed(protocol, update, micro, case, anchor):
    identity = [SPEC['seed'], protocol['seed'], protocol['fold'], update, micro, case, anchor]
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:8], 'little')

def transform_pair(images, seed, component):
    """Draw all original parameters; apply only the selected factor.

    Drawing disabled factors is deliberate: gain/noise must retain the exact
    marginal perturbations used by the earlier combined-augmentation run.
    """
    if component not in COMPONENTS:
        raise ValueError('Unknown component')
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(*BASE_SPEC['gamma_range']) if rng.random() < BASE_SPEC['gamma_probability'] else 1.
    gain = rng.uniform(*BASE_SPEC['gain_range']) if rng.random() < BASE_SPEC['gain_probability'] else 1.
    sigma = rng.uniform(*BASE_SPEC['noise_std_range']) if rng.random() < BASE_SPEC['noise_probability'] else 0.
    outputs = []
    for image in images:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError('Expected original uint8 plane')
        noise = rng.normal(0., sigma, image.shape).astype(np.float32) if sigma else None
        value = image.astype(np.float32)/255.
        if component == 'gamma':
            value = value**gamma
        elif component == 'gain':
            value = value*gain
        elif noise is not None:
            value = value+noise
        value[image == 0] = 0.
        outputs.append(np.round(np.clip(value, 0., 1.)*255).astype(np.uint8))
    return outputs, dict(component=component, gamma=gamma if component == 'gamma' else 1.,
        gain=gain if component == 'gain' else 1., noise_std=sigma if component == 'noise' else 0.,
        sampled_gamma=float(gamma), sampled_gain=float(gain), sampled_noise_std=float(sigma))
