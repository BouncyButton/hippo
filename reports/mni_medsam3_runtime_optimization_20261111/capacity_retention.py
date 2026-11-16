"""Device-capacity retention candidates; preserve checkpointed architecture flags."""
from contextlib import contextmanager


def selected_indices(model, count):
    blocks = model.backbone.vision_backbone.trunk.blocks
    assert len(blocks) == 32
    original = list(range(0, 32, 4))
    windows = [i for i, block in enumerate(blocks) if block.window_size > 0]
    assert set(original).issubset(windows)
    order = original + [i for i in windows if i not in original]
    if count not in (8, 12, 20, 28) or count > len(order):
        raise ValueError('Unaudited retention count')
    return order[:count]


@contextmanager
def retention(model, count):
    import torch.utils.checkpoint as cp
    import sam3.model.text_encoder_ve as text_encoder
    reference, text_reference = cp.checkpoint, text_encoder.checkpoint
    selected = {id(module) for name, module in model.named_modules()
                if name.startswith('backbone.language_backbone')}
    indices = selected_indices(model, count)
    selected.update(id(model.backbone.vision_backbone.trunk.blocks[i]) for i in indices)

    def checkpoint(function, *args, **kwargs):
        owner = getattr(function, '__self__', None)
        if (id(function) in selected or id(owner) in selected) and set(kwargs) <= {'use_reentrant'}:
            if kwargs.get('use_reentrant', False):
                raise ValueError('Reentrant checkpoint not audited')
            return function(*args)
        return reference(function, *args, **kwargs)

    cp.checkpoint = text_encoder.checkpoint = checkpoint
    try:
        yield dict(vision_blocks=indices, architecture_flags_unchanged=True)
    finally:
        cp.checkpoint, text_encoder.checkpoint = reference, text_reference
