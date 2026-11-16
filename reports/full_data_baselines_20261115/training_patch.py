"""Choose a seeded patch from the training example without changing its voxel spacing."""
import random


def sample_patch(image, label, shape, seed, update):
    assert tuple(image.shape) == tuple(label.shape) == (1, 1, 64, 64, 64)
    positive = (label[0, 0] > 0).nonzero()
    if len(positive) == 0:
        raise ValueError("An eligible training case has no labelled hippocampus")
    rng = random.Random((seed << 32) + update)
    origin = []
    for axis, width in enumerate(shape):
        low, high = (int(positive[:, axis].min()), int(positive[:, axis].max()))
        first, last = max(0, high + 1 - width), min(low, 64 - width)
        if first <= last:
            origin.append(rng.randint(first, last))
        else:
            origin.append(max(0, min(64 - width, (low + high + 1 - width) // 2)))
    region = tuple(slice(a, a + w) for a,w in zip(origin, shape))
    cut = label[(..., *region)].contiguous()
    stats = dict(origin=origin, patch_shape=list(shape), source_shape=list(image.shape[-3:]),
                 foreground_total=len(positive), foreground_retained=int((cut > 0).sum()))
    return image[(..., *region)].contiguous(), cut, stats
