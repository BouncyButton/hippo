"""Compute-matched supervised views for the translation experiment."""

import torch

from .equivariance import translate_3d


TRANSLATION_SHIFTS = (
    (2, 0, 0),
    (-2, 0, 0),
    (0, 2, 0),
    (0, -2, 0),
    (0, 0, 2),
    (0, 0, -2),
)


def augment_translation(images, labels, *, generator: torch.Generator):
    """Return one uniformly shifted image/label view for the whole batch.

    The shift stream and six +/-2 voxel transforms exactly match the original
    equivariance branch. The caller retains the identity view and supervises
    both views, so the augmentation control also matches its two forward passes.
    """
    if images.ndim != 5 or labels.shape != (images.shape[0], 1, *images.shape[2:]):
        raise ValueError('Expected images [B,C,D,H,W] and labels [B,1,D,H,W].')
    if min(images.shape[-3:]) <= 2:
        raise ValueError('Every spatial dimension must be larger than the shift.')
    shift = TRANSLATION_SHIFTS[
        int(torch.randint(len(TRANSLATION_SHIFTS), (), generator=generator).item())
    ]
    return translate_3d(images, shift), translate_3d(labels, shift), shift
