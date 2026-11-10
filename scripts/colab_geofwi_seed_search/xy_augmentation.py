"""Stateless, paired XY symmetries; native Z depth and global RNG are unchanged."""
import hashlib
import torch

POLICY = 'xy_d4_v1'


def transform_id(seed, update):
    if seed < 0 or update < 0:
        raise ValueError('Nonnegative seed and update required')
    return hashlib.sha256(f'geofwi-xy-d4-v1:{seed}:{update}'.encode()).digest()[0] % 8


def apply_transform(image, label, index):
    """Inputs are [B,C,X,Y,Z]; index selects one of eight square symmetries."""
    if image.ndim != 5 or label.ndim != 5 or image.shape != label.shape:
        raise ValueError('Aligned single-channel image and label required')
    if image.shape[2] != image.shape[3] or index not in range(8):
        raise ValueError('Square XY plane and D4 index required')
    def apply(value):
        value = torch.rot90(value, index % 4, dims=(2, 3))
        if index >= 4:
            value = torch.flip(value, dims=(2,))
        return value.contiguous()
    return apply(image), apply(label)


def augment(image, label, seed, update):
    return apply_transform(image, label, transform_id(seed, update))
