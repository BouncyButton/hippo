"""Reproducible supervised translation control, independent of constraint RNG."""
import hashlib

import torch

from .equivariance import translate_3d


def augment_translation(images, labels, *, seed: int, epoch: int, batch_index: int):
    """Per case: identity with p=1/2, otherwise uniform axis shift of +/-2.

    Image and class-index label receive exactly the same zero-padded shift.
    Sampling is keyed by run/epoch/batch, so checkpoint replay does not need an
    extra mutable RNG and enabling a constraint cannot change augmentation.
    """
    if images.ndim != 5 or labels.shape != (images.shape[0], 1, *images.shape[2:]):
        raise ValueError('Expected images [B,C,D,H,W] and labels [B,1,D,H,W].')
    if epoch < 1 or batch_index < 0 or min(images.shape[-3:]) <= 2:
        raise ValueError('Invalid epoch, batch index, or spatial shape.')
    key = f'translation-augmentation-v1:{seed}:{epoch}:{batch_index}'.encode()
    generator = torch.Generator(device='cpu').manual_seed(
        int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63)
    )
    shifts = ((2, 0, 0), (-2, 0, 0), (0, 2, 0),
              (0, -2, 0), (0, 0, 2), (0, 0, -2))
    augmented_images, augmented_labels, selected = [], [], []
    for index in range(images.shape[0]):
        shift = ((0, 0, 0) if torch.rand((), generator=generator).item() < 0.5
                 else shifts[torch.randint(6, (), generator=generator).item()])
        augmented_images.append(translate_3d(images[index:index+1], shift))
        augmented_labels.append(translate_3d(labels[index:index+1], shift))
        selected.append(shift)
    return torch.cat(augmented_images), torch.cat(augmented_labels), tuple(selected)
