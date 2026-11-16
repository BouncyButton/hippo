"""Audited activation retention; original architecture flags remain unchanged."""
from contextlib import contextmanager
import torch

@contextmanager
def retention(model, mode):
    import torch.utils.checkpoint as cp
    import sam3.model.text_encoder_ve as text_encoder
    reference=cp.checkpoint
    text_reference=text_encoder.checkpoint
    selected={id(module) for name,module in model.named_modules()
              if (name.startswith('backbone.language_backbone') if mode.startswith('text')
                  else not name.startswith('backbone.vision_backbone'))}
    if mode.startswith('vision8'):
        selected.update(id(model.backbone.vision_backbone.trunk.blocks[i]) for i in range(0,32,4))
    if mode in ('text_vision4','text_vision8'):
        stride=8 if mode=='text_vision4' else 4
        selected.update(id(model.backbone.vision_backbone.trunk.blocks[i]) for i in range(0,32,stride))
    def checkpoint(function,*args,**kwargs):
        owner=getattr(function,'__self__',None)
        if (id(function) in selected or id(owner) in selected) and set(kwargs)<={'use_reentrant'}:
            if kwargs.get('use_reentrant',False):raise ValueError('Reentrant checkpoint not audited')
            return function(*args)
        return reference(function,*args,**kwargs)
    cp.checkpoint=checkpoint
    text_encoder.checkpoint=checkpoint
    try:yield [dict(policy=mode,selected_modules=len(selected),architecture_flags_unchanged=True)]
    finally:
        cp.checkpoint=reference
        text_encoder.checkpoint=text_reference
