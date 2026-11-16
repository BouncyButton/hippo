"""Training-distribution BAND calibration, isolated from historical recipes."""
from copy import deepcopy
import hashlib
import inspect
import json

import numpy as np

from scripts.colab_hfh import medsam_augmentation_components as component
from scripts.colab_hfh import medsam_gain_replication as gain
from scripts.colab_hfh import medsam_intensity_augmentation as augmentation

CAMPAIGN = 'medsam_gain_band_recalibration125_seed0_v1'
POLICY = dict(name='uniform_gain_median001_p95005_v1', slabs_per_crop=20,
              median_target=.01, p95_cap=.05, bootstrap_replicates=1000,
              bootstrap_seed=20261113, weight_selection='point_estimate',
              sampling='Equal crop mass; uniformly jittered depth strata over all training anchors',
              augmentation='Original gain-only transform, independent deterministic calibration draws',
              stability='Subject-cluster bootstrap and leave-one-subject-out; diagnostic only')


def calibration_schedule(protocol, counts):
    """Equal crop mass and uniform anchor marginals, independent of training RNG."""
    if set(counts) != set(protocol['train_cases']) or set(counts) & set(protocol['evaluation_cases']):
        raise ValueError('Calibration must use exactly the training crops')
    rows = []
    for case in sorted(counts):
        identity = [POLICY['name'], protocol['calibration_seed'], protocol['fold'], protocol['seed'], case]
        seed = int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        n = counts[case]
        if n < 2:
            raise ValueError('Calibration needs adjacent slices')
        for j in range(POLICY['slabs_per_crop']):
            anchor = min(n - 1, int((j + rng.random()) * n / POLICY['slabs_per_crop']))
            rows.append(dict(case=case, anchor=anchor, subject=case.rsplit('_h', 1)[0],
                             augmentation_seed=int(rng.integers(0, 2**63 - 1)),
                             model_seed=int(rng.integers(0, 2**31 - 1))))
    return rows


def coefficient(ratios):
    ratios = np.asarray(ratios, dtype=float)
    if not len(ratios) or not np.isfinite(ratios).all() or (ratios < 0).any():
        raise ValueError('Invalid gradient ratio distribution')
    median, p95 = np.quantile(ratios, [.5, .95])
    if p95 <= 0:
        raise ValueError('Insufficient nonzero BAND calibration support')
    return float(min(POLICY['median_target'] / median if median > 0 else np.inf,
                     POLICY['p95_cap'] / p95))


def summarize(rows):
    """Keep zero BAND gradients: empty training slabs are part of the sampler."""
    grams = np.asarray([r['gram'] for r in rows], dtype=float)
    if not np.isfinite(grams).all() or (grams[:, 0, 0] <= 0).any() or (grams[:, 1, 1] < 0).any():
        raise ValueError('Invalid calibration gradients')
    ratios = np.sqrt(grams[:, 1, 1] / grams[:, 0, 0])
    weight = coefficient(ratios)
    subjects = sorted({r['subject'] for r in rows})
    by_subject = [np.array([i for i, r in enumerate(rows) if r['subject'] == s]) for s in subjects]
    if len(subjects) != 5 or len({len(x) for x in by_subject}) != 1:
        raise ValueError('Expected five equally represented training subjects')
    rng = np.random.default_rng(POLICY['bootstrap_seed'])
    boot = []
    undefined = 0
    for _ in range(POLICY['bootstrap_replicates']):
        selected = rng.integers(0, len(subjects), len(subjects))
        values = ratios[np.concatenate([by_subject[i] for i in selected])]
        try:
            boot.append(coefficient(values))
        except ValueError:
            undefined += 1
    loo = {}
    for s in subjects:
        values = ratios[[r['subject'] != s for r in rows]]
        try:
            loo[s] = coefficient(values)
        except ValueError:
            loo[s] = None
    stats = dict(valid=len(rows), target=POLICY['median_target'], p95_cap=POLICY['p95_cap'],
                 weighted_median=float(np.median(ratios) * weight),
                 weighted_p95=float(np.quantile(ratios, .95) * weight),
                 weighted_max=float(max(ratios) * weight), zero_band_gradients=int(sum(ratios == 0)))
    groups = {}
    for name in sorted({r['slab_category'] for r in rows}):
        subset = ratios[[r['slab_category'] == name for r in rows]]
        groups[name] = dict(count=len(subset), weighted_median=float(np.median(subset) * weight),
                            weighted_p95=float(np.quantile(subset, .95) * weight))
    stability = dict(subjects=subjects, bootstrap_defined=len(boot), bootstrap_undefined=undefined,
                     bootstrap_weight_p025_p50_p975=np.quantile(boot, [.025, .5, .975]).tolist() if boot else None,
                     leave_one_subject_out=loo, used_to_select_weight=False)
    return dict(status='complete', weights={'bands':weight, 'edge':0.}, statistics={'bands':stats},
                rows=rows, names=['native', 'bands'], scope='Training only, augmented, zero optimizer updates',
                target=POLICY['median_target'], p95_cap=POLICY['p95_cap'], policy=deepcopy(POLICY),
                category_statistics=groups, stability=stability)


def check_protocol(protocol):
    gain.check_protocol(protocol)
    if protocol['seed'] != 0 or protocol['fold'] not in (0, 1, 2):
        raise ValueError('Only three seed-zero BAND runs authorized')
    if protocol.get('recalibration') != POLICY or protocol['requested_arms'] != ['dice_bands']:
        raise ValueError('Recalibration recipe differs')
    if protocol['constraint_warmup_updates'] != 20:
        raise ValueError('Training ramp must remain20')


def install(engine, protocol, output):
    check_protocol(protocol)
    gain.install(engine, protocol, output)


def install_calibration(engine, protocol, output):
    check_protocol(protocol)
    import torch
    engine.write(output / 'execution_hardware.json', dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        bf16=torch.cuda.is_bf16_supported(), deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32))

    def calibrate(model, criterion, parameters, slabs, current, arm, destination):
        from scripts.cross_dataset_campaign.adni_foundation_full_runtime import native_training_check
        from scripts.colab_mni_medsam3.optimized import rng_state, set_rng
        if arm != 'dice_bands':
            raise ValueError('No native or other constraint training permitted')
        native_training_check(model, criterion, parameters, slabs, current, destination)
        schedule = calibration_schedule(protocol, {k:len(v) for k, v in slabs.indices.items()})
        engine.write(output / 'calibration_scope.json', dict(policy=POLICY, anchors=schedule,
            train_cases=protocol['train_cases'], validation_cases=protocol['evaluation_cases'], optimizer_updates=0))
        partial = output / 'calibration_partial.json'
        rows = engine.read(partial)['rows'] if partial.exists() else []
        if len(rows) > len(schedule) or any(any(r[k] != s[k] for k in s) for r, s in zip(rows, schedule)):
            raise ValueError('Partial calibration tape differs')
        before = rng_state()
        adapter_before = engine.parameter_digest(model, True)
        cache = slabs.processor.transform
        enabled = getattr(cache, 'enabled', None)
        try:
            if enabled is not None:
                cache.enabled = False
            for entry in schedule[len(rows):]:
                if engine.stopping():
                    raise SystemExit(75)
                case, anchor = entry['case'], entry['anchor']
                engine.reset_seed(entry['model_seed'])
                indices = slabs.indices[case]
                labels = [slabs.samples[indices[z]][1] for z in (anchor, (anchor + 1) % len(indices))]
                fg = sum(int(np.count_nonzero(x)) for x in labels)
                with augmentation.augmented_samples(slabs, case, anchor, entry['augmentation_seed']) as params:
                    losses = engine.forward_slab(model, criterion, slabs, case, anchor)
                    gradients = [torch.autograd.grad(losses[name], parameters, retain_graph=i == 0,
                        allow_unused=True) for i, name in enumerate(('native', 'bands'))]
                    gram = [[sum(float((a.detach().double() * b.detach().double()).sum())
                        for a, b in zip(ga, gb) if a is not None and b is not None)
                        for gb in gradients] for ga in gradients]
                    row = dict(**entry, gram=gram, augmentation=params, foreground_voxels=fg,
                        slab_category='empty' if fg == 0 else 'foreground',
                        losses={name:float(losses[name].detach()) for name in ('native', 'bands')})
                    if not np.isfinite(gram).all() or not np.isfinite(list(row['losses'].values())).all():
                        raise ValueError('Nonfinite calibration')
                    rows.append(row)
                    engine.write(partial, dict(policy=POLICY, names=['native','bands'], rows=rows))
                    if len(rows) % 10 == 0:
                        print('CALIBRATION', len(rows), '/', len(schedule), flush=True)
                    del gradients, losses
        finally:
            set_rng(before)
            if enabled is not None:
                cache.enabled = enabled
        if engine.parameter_digest(model, True) != adapter_before:
            raise ValueError('Calibration changed adapter parameters')
        result = summarize(rows)
        engine.write(output / 'calibration_stability.json', result['stability'])
        return result

    engine.calibrate = calibrate
    # Retain the established exact loss/gradient/RNG logging preflight around our calibration.
    source = inspect.getsource(component.install_calibration)
    needle = '    original_aug.install_calibration(engine, protocol, output)'
    if source.count(needle) != 1:
        raise ValueError('Logging preflight hook changed')
    source = source.replace(needle, '    # Retain the explicitly installed augmented calibration.')
    namespace = dict(component.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    namespace['install_calibration'](engine, protocol, output)
