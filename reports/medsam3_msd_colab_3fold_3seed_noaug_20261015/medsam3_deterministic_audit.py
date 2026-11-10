"""Isolated candidate GPU audit. Does not modify or approve the campaign."""
import argparse, json, os, sys
from pathlib import Path
from copy import deepcopy
import numpy as np
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


def equivalence(original, model, processor, criterion, parameters, slabs, p):
    """Real GPU loss/gradient and optimizer checkpoint-resume checks."""
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Strict equivalence requires the isolated deterministic audit process')
    case, anchor = slabs.calibration_schedule(4)[1]
    by_id = {id(value): name for name, value in model.named_parameters()}
    parameter_names = [by_id[id(value)] for value in parameters]
    processor.transform.enabled = False
    original.reset_seed(7319)
    losses = original.forward_slab(model, criterion, slabs, case, anchor)
    expected = [float(v.detach()) for v in losses]
    gradients = []
    for i, loss in enumerate(losses):
        g = torch.autograd.grad(loss, parameters, retain_graph=i<2, allow_unused=True)
        gradients.append([None if x is None else x.detach().cpu().clone() for x in g])
    del losses, g, loss
    for test in ('original_repeat', 'extension_cached'):
        processor.transform.enabled = test == 'extension_cached'
        original.reset_seed(7319)
        losses = (forward(original, model, criterion, slabs, case, anchor) if test == 'extension_cached'
                  else dict(zip(('native', 'bands', 'edge'),
                                original.forward_slab(model, criterion, slabs, case, anchor))))
        for i, name in enumerate(('native', 'bands', 'edge')):
            np.testing.assert_allclose(float(losses[name].detach()), expected[i], rtol=1e-6, atol=1e-7,
                                       err_msg=f'{test}: {name} loss differs')
            actual = torch.autograd.grad(losses[name], parameters, retain_graph=i<2, allow_unused=True)
            for parameter_name, a, b in zip(parameter_names, gradients[i], actual):
                label = f'{test}: {name} gradient for {parameter_name}'
                if (a is None) != (b is None): raise ValueError(label+' connectivity changed')
                if a is not None:
                    torch.testing.assert_close(a, b.detach().cpu(), rtol=1e-5, atol=1e-7,
                                               msg=lambda message: label+'\n'+message)
        del actual, losses
        print('PASSED strict equivalence:', test, flush=True)
    del gradients
    initial = [x.detach().cpu().clone() for x in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=.01)
    def update():
        optimizer.zero_grad(set_to_none=True)
        values = forward(original, model, criterion, slabs, case, anchor)
        total = values['native'] + .01*sum(values[k] for k in ('bands', 'edge', 'sdf', 'depth'))
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True); optimizer.step()
    original.reset_seed(7319); update()
    state = dict(parameters=[x.detach().cpu().clone() for x in parameters],
                 optimizer=deepcopy(optimizer.state_dict()), rng=rng_state())
    update(); expected_params = [x.detach().cpu().clone() for x in parameters]
    expected_optimizer = deepcopy(optimizer.state_dict()); expected_rng = rng_state()
    with torch.no_grad():
        for x, y in zip(parameters, state['parameters']): x.copy_(y)
    optimizer.load_state_dict(state['optimizer']); set_rng(state['rng']); update()
    for x, y in zip(parameters, expected_params): torch.testing.assert_close(x.detach().cpu(), y, rtol=0, atol=0)
    for key, record in optimizer.state_dict()['state'].items():
        for field, value in record.items():
            torch.testing.assert_close(value, expected_optimizer['state'][key][field], rtol=0, atol=0)
    actual_rng = rng_state()
    if not torch.equal(actual_rng['torch'], expected_rng['torch']) or any(not torch.equal(a,b) for a,b in zip(actual_rng['cuda'], expected_rng['cuda'])):
        raise ValueError('Checkpoint resume RNG differs')
    with torch.no_grad():
        for x, y in zip(parameters, initial): x.copy_(y)
    optimizer.zero_grad(set_to_none=True); del optimizer, state, expected_params, expected_optimizer, initial
    torch.cuda.empty_cache()
    # Inference has no annotation inputs and is unchanged by deterministic transform caching.
    model.eval(); pixels = slabs.samples[0][0]
    with torch.inference_mode():
        processor.transform.enabled=False; a,_=original.forward_plane(model, processor, pixels)
        processor.transform.enabled=True; b,_=original.forward_plane(model, processor, pixels)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    model.train()
    return dict(status='passed', original_repeatability=True, original_loss_gradient_equivalence=True,
                optimizer_checkpoint_resume_exact=True, label_free_inference_exact=True,
                numerical_mode='isolated deterministic audit; original calibration/training modes unchanged',
                peak_gpu_bytes=torch.cuda.max_memory_allocated())

CANDIDATE_SOURCES = {'equivalence': '6596a32aca8e3bca0d5365137879ffaef7245c6a856c676d43d20dfc89746c4c', 'audit_numerics.py': '2e4ab46a51bb9a144f8a9fb31c55241ff2453ff4a40bf6aee7d4956c26d408a4'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'source'))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Run the supplied cell with its isolated audit environment.')
    import torch._functorch.config as aot_config
    aot_config.donated_buffer = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    print('AUDIT SETTINGS:', torch.are_deterministic_algorithms_enabled(),
          aot_config.donated_buffer, flush=True)
    import engine
    import runtime
    from scripts.train_medsam3_few_shot import parameter_digest
    runtime.verify_bundle(root)
    globals().update(forward=engine.forward, rng_state=engine.rng_state, set_rng=engine.set_rng)
    report = dict(status='diagnostic_only', preflight_approved=False, scientific_updates=0,
        environment=runtime.environment(), bundle_sha256=runtime.digest(root/'bundle_manifest.json'),
        probe_sha256=runtime.digest(Path(__file__)), candidate_sources=CANDIDATE_SOURCES,
        gradient_rtol=1e-5, gradient_atol=1e-7,
        training_recipe_changed=False, augmentation='none')
    try:
        p = runtime.read(root/'protocols/fold0_seed17.json')
        original, model, processor, criterion, parameters, slabs, old = engine.load(root, p)
        print('AUDIT: strict original repeat, extension, optimizer resume and inference.', flush=True)
        print('Any optimizer steps are temporary diagnostic steps; no training receipts are written.', flush=True)
        with audit_empty_point_grid() as grid_audit:
            report['empty_point_grid'] = grid_audit
            checks = equivalence(original, model, processor, criterion, parameters, slabs, p)
        report['checks'] = checks
        report['adapter_restored'] = parameter_digest(model, True) == old['context']['adapter_before']
        report['frozen_unchanged'] = parameter_digest(model, False) == old['context']['frozen_before']
        if not report['adapter_restored'] or not report['frozen_unchanged']:
            raise RuntimeError('Audit did not restore the original model exactly')
        report['candidate_audit_passed'] = True
        print('CANDIDATE AUDIT PASSED; campaign preflight is still required.', flush=True)
    except Exception as exc:
        report['candidate_audit_passed'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        runtime.write(args.output, report)
        print('AUDIT_JSON:', args.output, flush=True)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
