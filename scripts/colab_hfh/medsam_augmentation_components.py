"""Matched gamma/gain/noise ablations with observational native-loss logging."""
from copy import deepcopy
import inspect
import math

import numpy as np

from scripts.colab_hfh import medsam_intensity_augmentation as original_aug

BASE_SPEC = deepcopy(original_aug.SPEC)
COMPONENTS = ('gamma', 'gain', 'noise')


def component_spec(component):
    if component not in COMPONENTS:
        raise ValueError('Unknown intensity component')
    spec = deepcopy(BASE_SPEC)
    spec.update(name='mild_intensity_component_v1', component=component,
                draw_policy='Original combined-augmentation parameter and pixel-noise tape')
    for name in COMPONENTS:
        spec[name+'_probability'] = BASE_SPEC[name+'_probability'] if name == component else 0.
    return spec


def transform_pair(images, seed, component):
    """Draw all original parameters; apply only the selected factor.

    Drawing disabled factors is deliberate: gain/noise must retain the exact
    marginal perturbations used by the earlier combined-augmentation run.
    """
    if component not in COMPONENTS:
        raise ValueError('Unknown component')
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(*BASE_SPEC['gamma_range']) if rng.random() < BASE_SPEC['gamma_probability'] else 1.
    gain = rng.uniform(*BASE_SPEC['gain_range']) if rng.random() < BASE_SPEC['gain_probability'] else 1.
    sigma = rng.uniform(*BASE_SPEC['noise_std_range']) if rng.random() < BASE_SPEC['noise_probability'] else 0.
    outputs = []
    for image in images:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError('Expected original uint8 plane')
        noise = rng.normal(0., sigma, image.shape).astype(np.float32) if sigma else None
        value = image.astype(np.float32)/255.
        if component == 'gamma':
            value = value**gamma
        elif component == 'gain':
            value = value*gain
        elif noise is not None:
            value = value+noise
        value[image == 0] = 0.
        outputs.append(np.round(np.clip(value, 0., 1.)*255).astype(np.uint8))
    return outputs, dict(component=component, gamma=gamma if component == 'gamma' else 1.,
        gain=gain if component == 'gain' else 1., noise_std=sigma if component == 'noise' else 0.,
        sampled_gamma=float(gamma), sampled_gain=float(gain), sampled_noise_std=float(sigma))


def scalar_terms(value, prefix=''):
    """Detach scalar diagnostics; never replace the criterion's actual return."""
    import torch
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(scalar_terms(item, prefix+'/'+str(key) if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            result.update(scalar_terms(item, prefix+'/'+str(i)))
    elif isinstance(value, torch.Tensor) and value.numel() == 1:
        result[prefix] = value.detach().reshape(())
    elif isinstance(value, (float, int)):
        result[prefix] = torch.tensor(float(value))
    return result


class LoggedCriterion:
    def __init__(self, criterion):
        self.criterion, self.rows, self.predictions = criterion, [], []

    def __call__(self, *args, **kwargs):
        import torch
        result = self.criterion(*args, **kwargs)
        terms = scalar_terms(result)
        core = terms['core_loss']
        weights = {k:float(v) for fn in self.criterion.loss_fns_find for k,v in fn.weight_dict.items()}
        if self.criterion.normalize_by_stage_num or self.criterion.scale_by_find_batch_size:
            raise ValueError('Loss logging assumes the established image normalization')
        for name, weight in weights.items():
            values = [v.to(core.device) for k,v in terms.items() if k == name or k.startswith(name+'_')]
            terms['weighted/'+name] = torch.stack(values).sum()*weight if values else torch.zeros_like(core)
        reconstructed = sum(terms['weighted/'+name] for name in weights)
        if not torch.isclose(core, reconstructed, rtol=1e-5, atol=1e-3):
            raise ValueError('Logged weighted components do not reconstruct native objective')
        self.rows.append(terms)
        return result


def install_prediction_logging(engine):
    reference = engine.forward_plane

    def forward(model, processor, pixels, protocol, labels=None, criterion=None):
        import torch
        result = reference(model, processor, pixels, protocol, labels, criterion)
        if isinstance(criterion, LoggedCriterion):
            with torch.no_grad():
                pred = result[0].detach().argmax(0) == 1
                truth = torch.as_tensor(labels == 1, device=pred.device)
                criterion.predictions.append(dict(tp=(pred & truth).sum().float(),
                    fp=(pred & ~truth).sum().float(), fn=(~pred & truth).sum().float()))
        return result
    engine.forward_plane = forward


def attach_terms(losses, logged):
    import torch
    if len(logged.rows) != 2 or len(logged.predictions) != 2:
        raise ValueError('Expected two planes and one binary query task per slab')
    keys = set().union(*(set(row) for row in logged.rows))
    zero = torch.zeros_like(losses['native']).detach()
    for key in sorted(keys):
        losses['native_term/'+key] = torch.stack([r.get(key, zero).to(zero.device) for r in logged.rows]).mean().detach()
    for name in ('tp', 'fp', 'fn'):
        losses['augmented_training/'+name] = torch.stack([r[name] for r in logged.predictions]).sum().detach()
    return losses


def install(engine, protocol, output):
    component = protocol['augmentation'].get('component')
    if protocol['augmentation'] != component_spec(component):
        raise ValueError('Unexpected component protocol')
    # This module runs in a dedicated worker process. The existing source and
    # completed experiment protocols are never edited.
    original_aug.SPEC.clear(); original_aug.SPEC.update(component_spec(component))
    original_aug.transform_pair = lambda images, seed: transform_pair(images, seed, component)
    install_prediction_logging(engine)
    original_aug.install(engine, protocol, output)
    reference = engine.hfh_augmented_forward

    def logged_forward(model, criterion, slabs, case, anchor, update, micro):
        logged = LoggedCriterion(criterion)
        losses = reference(model, logged, slabs, case, anchor, update, micro)
        return attach_terms(losses, logged)
    engine.hfh_augmented_forward = logged_forward


def install_calibration(engine, protocol, output):
    """Reuse native control calibration and verify logging before any updates."""
    original_aug.install_calibration(engine, protocol, output)
    reference = engine.calibrate

    def calibrate(model, criterion, parameters, slabs, current, arm, destination):
        import torch
        from scripts.colab_mni_medsam3.optimized import rng_state, set_rng
        from scripts.colab_medsam3.fast_eval import equal_state
        calibration = reference(model, criterion, parameters, slabs, current, arm, destination)
        before = rng_state()
        adapter_before = engine.parameter_digest(model, True)
        case, anchor = protocol['frozen_sample_schedule'][0]
        try:
            losses = engine.forward_slab(model, criterion, slabs, case, anchor)
            # Keep the same native-plus-zero-auxiliary graph as actual training.
            total = losses['native']+sum(calibration['weights'][k]*losses[k] for k in calibration['weights'])
            gradients = torch.autograd.grad(total, parameters, allow_unused=True)
            saved = [g.detach().cpu() if g is not None else None for g in gradients]
            native = float(total); after = rng_state()
            del gradients, losses, total
            set_rng(before)
            logged = LoggedCriterion(criterion)
            losses = attach_terms(engine.forward_slab(model, logged, slabs, case, anchor), logged)
            total = losses['native']+sum(calibration['weights'][k]*losses[k] for k in calibration['weights'])
            gradients = torch.autograd.grad(total, parameters, allow_unused=True)
            exact = native == float(total) and all(
                (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b.detach().cpu()))
                for a,b in zip(saved, gradients))
            if not exact or not equal_state(after, rng_state()):
                raise ValueError('Diagnostic logging altered loss, gradients or RNG')
            terms = {k:float(v) for k,v in losses.items() if k.startswith('native_term/')}
            if not terms or not all(math.isfinite(v) for v in terms.values()):
                raise ValueError('Missing/nonfinite native component diagnostics')
            required = ('loss_ce', 'presence_loss', 'loss_bbox', 'loss_giou', 'loss_mask', 'loss_dice')
            if any(not any(name in k for k in terms) for name in required):
                raise ValueError('Criterion does not expose all required native losses: '+str(terms.keys()))
            engine.write(output/'loss_logging_preflight.json', dict(status='passed', optimizer_updates=0,
                exact_loss=True, exact_gradients=True, exact_rng=True, native_loss=native,
                example_terms=terms, criterion_source=inspect.getsource(type(criterion)),
                weights={k:float(v) for fn in criterion.loss_fns_find for k,v in fn.weight_dict.items()},
                o2m_weight=float(criterion.o2m_weight),
                description='Native return dictionary: raw losses already include upstream o2m multiplier; weighted/name sums all final and auxiliary stages times the native coefficient. Weighted groups reconstruct core_loss. Per-update mean across four training planes. Training TP/FP/FN are per adjacent pair averaged across accumulation, on augmented inputs.'))
            del gradients, saved, total, losses
        finally:
            set_rng(before)
        if engine.parameter_digest(model, True) != adapter_before:
            raise ValueError('Logging preflight changed model')
        return calibration
    engine.calibrate = calibrate
