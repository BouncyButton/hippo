"""Cache exact BF16 conversions of immutable FP32 linear weights only."""
from contextlib import contextmanager
import torch
import torch.nn.functional as functional


class FrozenLinearCache:
    def __init__(self, model):
        self.allowed = {id(p) for p in model.parameters() if not p.requires_grad}
        self.entries = {}
        self.hits = self.misses = self.bytes = 0

    def cast(self, value, dtype):
        if (value is None or id(value) not in self.allowed or value.requires_grad or not value.is_leaf
                or value.device.type != 'cuda' or value.dtype != torch.float32):
            return value
        try:
            version = value._version
        except RuntimeError:
            return value
        key = id(value)
        existing = self.entries.get(key)
        if existing is not None and existing[0] is value and existing[1:3] == (version, dtype):
            self.hits += 1
            return existing[3]
        # No arithmetic or parameter mutation: exactly the autocast conversion.
        if existing is None and self.bytes + value.numel()*2 > 2*1024**3:
            return value
        converted = value.to(dtype=dtype)
        if existing is not None:
            self.bytes -= existing[3].numel()*existing[3].element_size()
        self.entries[key] = (value, version, dtype, converted)
        self.bytes += converted.numel()*converted.element_size()
        self.misses += 1
        return converted

    def clear(self):
        self.entries.clear()
        self.bytes = 0


@contextmanager
def frozen_linear_cache(model):
    original = functional.linear
    cache = FrozenLinearCache(model)
    def linear(input, weight, bias=None):
        if (input.device.type == 'cuda' and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') == torch.bfloat16):
            weight = cache.cast(weight, torch.bfloat16)
            bias = cache.cast(bias, torch.bfloat16)
        return original(input, weight, bias)
    functional.linear = linear
    try:
        yield cache
    finally:
        functional.linear = original
        cache.clear()
