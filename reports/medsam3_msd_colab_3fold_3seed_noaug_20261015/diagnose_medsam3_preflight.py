"""Read-only model diagnostic; never writes a campaign preflight receipt."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import torch


def gradient_difference(reference, actual, names):
    """Use the existing gate's tolerances and identify affected parameters."""
    if len(reference) != len(actual) or len(reference) != len(names):
        raise ValueError('Gradient parameter lists differ')
    def finite(value):
        return value if math.isfinite(value) else None
    failures = []
    squared_delta = squared_reference = 0.
    count = mismatches = 0
    for name, left, right in zip(names, reference, actual):
        if (left is None) != (right is None):
            failures.append(dict(parameter=name, reason='connectivity differs'))
            continue
        if left is None:
            continue
        right = right.detach().cpu()
        close = torch.isclose(left, right, rtol=1e-5, atol=1e-7)
        different = int((~close).sum())
        count += left.numel(); mismatches += different
        a, b = left.double(), right.double()
        squared_delta += float((a-b).square().sum())
        squared_reference += float(a.square().sum())
        if different:
            failures.append(dict(parameter=name, shape=list(a.shape),
                mismatches=different, elements=a.numel(),
                max_abs=finite(float((a-b).abs().max())), reference_max_abs=finite(float(a.abs().max())),
                reference_finite=bool(a.isfinite().all()), actual_finite=bool(b.isfinite().all())))
    return dict(passed=not failures, failed_parameters=len(failures),
        mismatched_elements=mismatches, compared_elements=count,
        relative_l2=finite((squared_delta/max(squared_reference, 1e-300))**.5),
        first_failures=failures[:5])


def rng_fingerprint(engine):
    state = engine.rng_state()
    return dict(cpu=hashlib.sha256(state['torch'].numpy().tobytes()).hexdigest(),
        cuda=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in state['cuda']],
        python=hashlib.sha256(pickle.dumps(state['python'])).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(state['numpy'])).hexdigest())


def run(root, output):
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'source'))
    import engine
    import runtime
    from scripts.train_medsam3_few_shot import parameter_digest
    runtime.verify_bundle(root)
    environment = runtime.environment()
    torch.set_num_threads(4)
    p = runtime.read(root/'protocols/fold0_seed17.json')
    original, model, processor, criterion, parameters, slabs, old = engine.load(root, p)
    case, anchor = slabs.calibration_schedule(4)[1]
    names_by_id = {id(v): k for k, v in model.named_parameters()}
    names = [names_by_id[id(v)] for v in parameters]
    report = dict(status='diagnostic_only', environment=environment,
        bundle_sha256=runtime.digest(root/'bundle_manifest.json'),
        diagnostic_sha256=runtime.digest(Path(__file__)), case=case, anchor=anchor,
        seed=7319, rtol=1e-5, atol=1e-7, comparisons=[],
        optimizer_updates=0, preflight_approved=False)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        runtime.write(output, report)

    def sample(extended, cached):
        processor.transform.enabled = cached
        original.reset_seed(7319)
        before = rng_fingerprint(engine)
        if extended:
            values = engine.forward(original, model, criterion, slabs, case, anchor)
        else:
            values = dict(zip(('native', 'bands', 'edge'),
                original.forward_slab(model, criterion, slabs, case, anchor)))
        after_forward = rng_fingerprint(engine)
        losses, gradients = {}, {}
        for index, name in enumerate(('native', 'bands', 'edge')):
            losses[name] = float(values[name].detach())
            grad = torch.autograd.grad(values[name], parameters,
                retain_graph=index < 2, allow_unused=True)
            gradients[name] = [None if v is None else v.detach().cpu().clone() for v in grad]
            del grad
        record = dict(losses=losses, rng_before=before, rng_after_forward=after_forward,
            rng_after_backward=rng_fingerprint(engine))
        del values
        gc.collect(); torch.cuda.empty_cache()
        return record, gradients

    try:
        print('DIAGNOSTIC: original code, cache disabled; no optimizer updates', flush=True)
        reference, reference_gradients = sample(False, False)
        report['reference'] = reference; save()
        for label, extended, cached in (
                ('original_repeat', False, False),
                ('original_with_cache', False, True),
                ('extension_with_cache', True, True)):
            print('DIAGNOSTIC:', label, flush=True)
            actual, actual_gradients = sample(extended, cached)
            comparison = dict(test=label, **actual,
                losses_close={name: math.isclose(actual['losses'][name], reference['losses'][name],
                              rel_tol=1e-6, abs_tol=1e-7) for name in ('native', 'bands', 'edge')},
                rng_before_equal=actual['rng_before'] == reference['rng_before'],
                rng_after_forward_equal=actual['rng_after_forward'] == reference['rng_after_forward'],
                rng_after_backward_equal=actual['rng_after_backward'] == reference['rng_after_backward'],
                gradients={name: gradient_difference(reference_gradients[name], actual_gradients[name], names)
                           for name in ('native', 'bands', 'edge')})
            report['comparisons'].append(comparison)
            print(json.dumps(comparison, indent=2), flush=True); save()
            del actual_gradients
            # Original-code variation must be understood before changing the extension.
            if label == 'original_repeat' and any(not v['passed'] for v in comparison['gradients'].values()):
                report['finding'] = 'Original code does not meet the gradient tolerance when repeated with the same seed.'
                break
        else:
            report['finding'] = 'See separate cache and extension comparisons; this diagnostic does not authorize training.'
        report['adapter_unchanged'] = parameter_digest(model, True) == old['context']['adapter_before']
        report['frozen_unchanged'] = parameter_digest(model, False) == old['context']['frozen_before']
        if not report['adapter_unchanged'] or not report['frozen_unchanged']:
            raise RuntimeError('Model state changed during diagnostic forwards/backwards')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
        print('DIAGNOSTIC_JSON:', output, flush=True)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.bundle.resolve(), args.output.resolve())
