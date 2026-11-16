"""Bounded HFH experiment: training-error-triggered BAND, alone or with SDF.

No validation values enter the gate, calibration, stopping time or optimizer.
The complete gate state is saved with the optimizer, rather than trusting a
potentially newer diagnostic sidecar on recovery.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import linecache
from pathlib import Path

import numpy as np

CAMPAIGN = 'medsam_adaptive_band_seed0_v1'
ARMS = ('dice_bands', 'dice_bands_sdf')
POLICY = dict(name='full_support_error_gate_v1', probe_every=20,
              activation_fraction=.55, deactivation_fraction=.45,
              subject_fraction=.50, required_subjects=4, consecutive_probes=2,
              ramp_updates=20, calibration_slabs_per_crop=8,
              band_target=.01, band_p95_cap=.05, sdf_target=.1, sdf_p95_cap=.5,
              bootstrap_replicates=1000, bootstrap_seed=20261114,
              gate_inputs='All voxels in all ten unaugmented training crops; no validation',
              geometry='Original two-step inner and outer bands; unchanged semantic gradients',
              no_forced_activation=True)


def check_protocol(p):
    from scripts.colab_hfh.medsam_gain_replication import check_protocol as gain_check
    gain_check(p)
    if p['seed'] != 0 or p['fold'] not in (1, 2) or p['requested_arms'] not in [[a] for a in ARMS]:
        raise ValueError('Only adaptive BAND +/- SDF, folds1/2 seed0 authorized')
    if p.get('adaptive_band') != POLICY or p['constraint_warmup_updates'] != 20:
        raise ValueError('Adaptive policy differs')
    if p.get('learning_rate_horizon_updates') != 200 or p['constraint_recipe'].get('delay', 0):
        raise ValueError('Original LR horizon and separate adaptive ramp required')
    if p['adaptive_source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Adaptive implementation differs from frozen protocol')


def summarize_errors(crops):
    keys = ('fp', 'fn', 'outer_fp', 'inner_fn', 'empty_slice_fp')
    subjects = {}
    for r in crops:
        s = subjects.setdefault(r['case'].rsplit('_h', 1)[0], dict.fromkeys(keys, 0))
        for k in keys:
            s[k] += int(r[k])
    if len(subjects) != 5 or len(crops) != 10 or len({r['case'] for r in crops}) != 10:
        raise ValueError('Gate requires all ten crops from five training subjects')
    def describe(r):
        e = r['fp'] + r['fn']
        return dict(r, errors=e, in_band=r['outer_fp']+r['inner_fn'],
                    fraction=(r['outer_fp']+r['inner_fn'])/e if e else None,
                    fp_fraction=r['outer_fp']/r['fp'] if r['fp'] else None,
                    fn_fraction=r['inner_fn']/r['fn'] if r['fn'] else None)
    total = describe({k:sum(s[k] for s in subjects.values()) for k in keys})
    subjects = {k:describe(v) for k,v in subjects.items()}
    ready = sum(v['fraction'] is not None and v['fraction'] >= POLICY['subject_fraction']
                for v in subjects.values())
    return dict(total=total, subjects=subjects, subjects_ready=ready, crops=crops)


def empty_state():
    return dict(active=False, activation_update=None, streak=0, weight=0.,
                last_probe=None, probes=[], calibrations=[], events=[])


def transition(state, observation, update):
    """Return a new decision state; calibration is performed only on activation."""
    s = deepcopy(state)
    fraction = observation['total']['fraction']
    ready = (fraction is not None and fraction >= POLICY['activation_fraction']
             and observation['subjects_ready'] >= POLICY['required_subjects'])
    s['streak'] = s['streak']+1 if ready else 0
    event = None
    if s['active'] and (fraction is None or fraction < POLICY['deactivation_fraction']
                        or observation['subjects_ready'] < 3):
        s.update(active=False, activation_update=None, weight=0., streak=0)
        event = 'deactivate'
    elif not s['active'] and s['streak'] >= POLICY['consecutive_probes']:
        s.update(active=True, activation_update=update)
        event = 'activate'
    s['last_probe'] = update
    s['probes'].append(dict(update=update, **observation, ready=ready, event=event))
    if event:
        s['events'].append(dict(update=update, action=event))
    return s, event


def band_factor(state, update):
    if not state['active']:
        return 0.
    return min(max((update+1-state['activation_update'])/POLICY['ramp_updates'], 0.), 1.)


def calibration_schedule(p, counts):
    if set(counts) != set(p['train_cases']) or set(counts) & set(p['evaluation_cases']):
        raise ValueError('Calibration must use exactly the training crops')
    rows = []
    for case in sorted(counts):
        identity = [POLICY['name'], p['calibration_seed'], p['fold'], case]
        seed = int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        for j in range(POLICY['calibration_slabs_per_crop']):
            anchor = min(counts[case]-1, int((j+rng.random())*counts[case]/POLICY['calibration_slabs_per_crop']))
            rows.append(dict(case=case, subject=case.rsplit('_h', 1)[0], anchor=anchor,
                             augmentation_seed=int(rng.integers(0, 2**63-1)),
                             model_seed=int(rng.integers(0, 2**31-1))))
    return rows


def coefficient(ratios, term):
    r = np.asarray(ratios, dtype=float)
    if not len(r) or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError('Invalid calibration ratios')
    med, tail = np.quantile(r, [.5, .95])
    if tail <= 0:
        raise ValueError('No nonzero auxiliary gradient support')
    prefix = 'band' if term == 'bands' else term
    return float(min(POLICY[prefix+'_target']/med if med else np.inf, POLICY[prefix+'_p95_cap']/tail))


def calibration_summary(rows, term):
    ratios = np.asarray([np.sqrt(r['auxiliary_norm2']/r['native_norm2']) for r in rows])
    w = coefficient(ratios, term)
    subjects = sorted({r['subject'] for r in rows})
    ids = [np.array([i for i,r in enumerate(rows) if r['subject'] == s]) for s in subjects]
    if len(subjects) != 5 or any(len(i) != 2*POLICY['calibration_slabs_per_crop'] for i in ids):
        raise ValueError('Expected equally represented five training subjects')
    rng = np.random.default_rng(POLICY['bootstrap_seed'])
    boot, undefined = [], 0
    for _ in range(POLICY['bootstrap_replicates']):
        try:
            boot.append(coefficient(ratios[np.concatenate([ids[j] for j in rng.integers(0, 5, 5)])], term))
        except ValueError:
            undefined += 1
    loo = {}
    for subject in subjects:
        try:
            loo[subject] = coefficient(ratios[[r['subject'] != subject for r in rows]], term)
        except ValueError:
            loo[subject] = None
    return dict(term=term, weight=w, rows=rows, weighted_median=float(np.median(ratios)*w),
                weighted_p95=float(np.quantile(ratios, .95)*w), zeros=int((ratios == 0).sum()),
                stability=dict(bootstrap_undefined=undefined,
                    weight_p025_p50_p975=np.quantile(boot, [.025,.5,.975]).tolist() if boot else None,
                    leave_one_subject_out=loo, used_to_select_weight=False))


@contextmanager
def preserve_runtime(model, slabs):
    from scripts.colab_mni_medsam3.optimized import rng_state, set_rng
    state = rng_state()
    modes = [(m, m.training) for m in model.modules()]
    cache = slabs.processor.transform
    enabled = getattr(cache, 'enabled', None)
    try:
        if enabled is not None:
            cache.enabled = False
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        if enabled is not None:
            cache.enabled = enabled
        set_rng(state)


def calibrate_term(engine, model, criterion, parameters, slabs, p, output, term, update):
    import torch
    from scripts.colab_hfh.medsam_intensity_augmentation import augmented_samples
    schedule = calibration_schedule(p, {k:len(v) for k,v in slabs.indices.items()})
    identity = dict(term=term, update=update, adapter_sha256=engine.parameter_digest(model, True),
                    protocol_sha256=engine.digest(output/'protocol.json'))
    path = output/f'adaptive_calibration_{term}_{update:03d}.json'
    saved = engine.read(path) if path.exists() else dict(identity=identity, rows=[])
    if saved['identity'] != identity:
        raise ValueError('Calibration checkpoint identity differs')
    rows = saved['rows']
    if len(rows) > len(schedule) or any(any(r[k] != s[k] for k in s) for r,s in zip(rows,schedule)):
        raise ValueError('Calibration schedule differs')
    with preserve_runtime(model, slabs):
        model.train()
        for entry in schedule[len(rows):]:
            if engine.stopping():
                raise SystemExit(75)
            engine.reset_seed(entry['model_seed'])
            with augmented_samples(slabs, entry['case'], entry['anchor'], entry['augmentation_seed']) as params:
                losses = engine.forward_slab(model, criterion, slabs, entry['case'], entry['anchor'])
                gradients = [torch.autograd.grad(losses[k], parameters, retain_graph=i == 0, allow_unused=True)
                             for i,k in enumerate(('native', term))]
                norms = [sum(float(g.detach().double().square().sum()) for g in gs if g is not None) for gs in gradients]
                if not np.isfinite(norms).all() or norms[0] <= 0:
                    raise ValueError('Nonfinite or empty native calibration gradient')
                rows.append(dict(**entry, augmentation=params, native_norm2=norms[0], auxiliary_norm2=norms[1]))
                engine.write(path, dict(identity=identity, rows=rows))
                if len(rows)%10 == 0:
                    print('ADAPTIVE_CALIBRATION', term, update, len(rows), '/', len(schedule), flush=True)
                del gradients, losses
    if engine.parameter_digest(model, True) != identity['adapter_sha256']:
        raise ValueError('Calibration modified the model')
    result = dict(identity=identity, **calibration_summary(rows, term))
    engine.write(path, result)
    return result


class BandGate:
    def __init__(self, engine, model, criterion, parameters, slabs, protocol, output):
        self.engine, self.model, self.criterion = engine, model, criterion
        self.parameters, self.slabs, self.protocol, self.output = parameters, slabs, protocol, output
        self.state = empty_state()

    def restore(self, state):
        self.state = deepcopy(state)

    def probe(self, update):
        import torch
        engine, model, slabs, p = self.engine, self.model, self.slabs, self.protocol
        crops = []
        with preserve_runtime(model, slabs), torch.inference_mode():
            model.eval()
            for case in p['train_cases']:
                if case in p['evaluation_cases']:
                    raise ValueError('Validation leakage')
                support = slabs.supports[case][1]
                counts = dict(case=case, fp=0, fn=0, outer_fp=0, inner_fn=0, empty_slice_fp=0)
                for z,index in enumerate(slabs.indices[case]):
                    if engine.stopping():
                        raise SystemExit(75)
                    pixels, labels = slabs.samples[index]
                    lp,_ = engine.forward_plane(model, slabs.processor, pixels, slabs.protocol)
                    pred = lp.argmax(0).cpu()
                    if not torch.isfinite(lp).all():
                        raise ValueError('Nonfinite gate prediction')
                    truth = torch.as_tensor(labels == 1)
                    fp,fn = (pred == 1)&~truth, (pred != 1)&truth
                    counts['fp'] += int(fp.sum()); counts['fn'] += int(fn.sum())
                    counts['outer_fp'] += int((fp & support.outer[z]).sum())
                    counts['inner_fn'] += int((fn & support.inner[z]).sum())
                    if not truth.any():
                        counts['empty_slice_fp'] += int(fp.sum())
                crops.append(counts)
                engine.write(self.output/'adaptive_probe_progress.json', dict(update=update, completed_crops=len(crops), total_crops=10))
            return summarize_errors(crops)

    def before_update(self, update):
        if update % POLICY['probe_every'] == 0 and self.state['last_probe'] != update:
            observation = self.probe(update)
            proposed, event = transition(self.state, observation, update)
            if event == 'activate':
                cal = calibrate_term(self.engine, self.model, self.criterion, self.parameters,
                    self.slabs, self.protocol, self.output, 'bands', update)
                proposed['weight'] = cal['weight']
                proposed['calibrations'].append(cal)
            self.state = proposed
            self.engine.write(self.output/'adaptive_band_state.json', self.state)
            print('ADAPTIVE_GATE', json.dumps(dict(update=update, fraction=observation['total']['fraction'],
                subjects_ready=observation['subjects_ready'], event=event, active=proposed['active'], weight=proposed['weight'])), flush=True)
        return dict(active=self.state['active'], band_ramp=band_factor(self.state, update),
                    band_weight=self.state['weight'], last_probe=self.state['last_probe'],
                    activation_update=self.state['activation_update'])


def training_source(source):
    replacements = (
        ('    frozen_before, adapter_before =',
         '    hfh_gate = hfh_make_gate(model, criterion, parameters, slabs, protocol, output)\n    frozen_before, adapter_before ='),
        ("        start,history,elapsed = state['step'],state['history'],state['elapsed_seconds']",
         "        start,history,elapsed = state['step'],state['history'],state['elapsed_seconds']\n        hfh_gate.restore(state['adaptive_band'])"),
        ('save_checkpoint(latest,dict(step=step,history=history,',
         'save_checkpoint(latest,dict(step=step,history=history,adaptive_band=deepcopy(hfh_gate.state),'),
        ("        horizon=protocol.get('learning_rate_horizon_updates',protocol['updates'])",
         "        if update % 20 == 0:\n            commit(update)\n        gate_report=hfh_gate.before_update(update)\n        horizon=protocol.get('learning_rate_horizon_updates',protocol['updates'])"),
        ("total=losses['native']+ramp*sum(calibration['weights'][name]*losses[name] for name in calibration['weights'])",
         "total=losses['native']+ramp*sum(calibration['weights'][name]*losses[name] for name in calibration['weights'])\n            if gate_report['active']:\n                total=total+gate_report['band_ramp']*gate_report['band_weight']*losses['bands']"),
        ('history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm),',
         'history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm),adaptive_band=gate_report,'),
        ("    model.eval().requires_grad_(False)\n    frozen_after", "    write(output/'adaptive_band_state.json',hfh_gate.state)\n    model.eval().requires_grad_(False)\n    frozen_after"),
    )
    for old,new in replacements:
        if source.count(old) != 1:
            raise ValueError('Audited training hook changed: '+old)
        source = source.replace(old,new)
    return source


def install(engine, protocol, output):
    from scripts.colab_hfh import medsam_gain_replication as gain
    check_protocol(protocol)
    engine.hfh_make_gate = lambda *args: BandGate(engine, *args)
    engine.deepcopy = deepcopy
    source = training_source(inspect.getsource(engine.train))
    filename = '<hfh_adaptive_training>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), engine.__dict__)
    gain.install(engine, protocol, output)
    engine.write(output/'adaptive_band_source.json', dict(source=Path(__file__).read_text(), policy=POLICY))


def install_calibration(engine, protocol, output):
    import torch
    from scripts.colab_hfh import medsam_augmentation_components as component
    check_protocol(protocol)
    engine.write(output/'execution_hardware.json', dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        bf16=torch.cuda.is_bf16_supported(), deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32))
    def calibrate(model, criterion, parameters, slabs, current, arm, destination):
        from scripts.cross_dataset_campaign.adni_foundation_full_runtime import native_training_check
        native_training_check(model, criterion, parameters, slabs, current, destination)
        weights = dict(bands=0., edge=0.)
        result = dict(status='adaptive_pending', weights=weights, rows=[], policy=POLICY)
        if 'sdf' in arm:
            cal = calibrate_term(engine, model, criterion, parameters, slabs, protocol, output, 'sdf', 0)
            weights['sdf'] = cal['weight']; result['sdf_calibration'] = cal; result['rows'] = cal['rows']
        return result
    engine.calibrate = calibrate
    source = inspect.getsource(component.install_calibration)
    old = '    original_aug.install_calibration(engine, protocol, output)'
    if source.count(old) != 1:
        raise ValueError('Logging preflight hook changed')
    source = source.replace(old, '    # Preserve adaptive initialization calibration.')
    namespace = dict(component.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    namespace['install_calibration'](engine, protocol, output)
