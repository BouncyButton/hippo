"""Numerical helpers restricted to the isolated, strict preflight audit."""
from contextlib import contextmanager

import torch
import torch.nn.functional as functional


@contextmanager
def audit_empty_point_grid():
    """Keep native grid_sample forward; supply exact zeros for an empty grid.

    SAM3 encodes an empty set of point prompts in this text-only experiment.
    CUDA grid_sample backward rejects strict determinism even for that empty
    output. There are no samples to scatter, so both input derivatives are
    exactly zero. Nonempty grids still use native autograd and are never exempt
    from PyTorch's strict determinism checks. This context is audit-only.
    """
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Empty-grid audit requires strict deterministic algorithms')
    native = functional.grid_sample
    report = dict(mode='audit-only empty-grid exact-zero backward',
                  native_forward=True, empty_calls=0, nonempty_calls=0, shapes=[])

    class EmptyGridSample(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, grid, mode, padding_mode, align_corners):
            # Native forward retains validation, autocast, shape, dtype and device.
            result = native(value, grid, mode=mode, padding_mode=padding_mode,
                            align_corners=align_corners)
            if grid.numel() != 0 or result.numel() != 0:
                raise RuntimeError('Empty-grid audit received nonempty samples')
            ctx.input_specs = [(tuple(v.shape), v.dtype, v.device) for v in (value, grid)]
            return result

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, derivative):
            if derivative.numel() != 0:
                raise RuntimeError('Empty-grid audit received a nonempty derivative')
            gradients = [torch.zeros(shape, dtype=dtype, device=device) if needed else None
                         for (shape, dtype, device), needed
                         in zip(ctx.input_specs, ctx.needs_input_grad[:2])]
            return *gradients, None, None, None

    def sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
        if value.ndim == 4 and grid.ndim == 4 and grid.numel() == 0:
            shape = dict(input=list(value.shape), grid=list(grid.shape), dtype=str(value.dtype))
            if shape not in report['shapes']:
                report['shapes'].append(shape)
                print('AUDIT empty point grid:', shape, flush=True)
            report['empty_calls'] += 1
            return EmptyGridSample.apply(value, grid, mode, padding_mode, align_corners)
        report['nonempty_calls'] += 1
        return native(value, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    functional.grid_sample = sample
    try:
        yield report
    finally:
        functional.grid_sample = native
