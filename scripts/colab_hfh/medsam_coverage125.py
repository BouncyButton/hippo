"""Frozen coverage-first sampling and fresh calibration for the HFH seed-0 pair."""
from copy import deepcopy
import inspect
import random

import numpy as np

CAMPAIGN = 'medsam_coverage125_f0s0_v1'
RECIPE = dict(name='coverage125_edge_depth_r20', target=.1, ramp=20,
              numerics='strict_empty_grid_v1')


def hardware_identity():
    import subprocess
    import torch
    name, uuid = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid',
        '--format=csv,noheader'],text=True).strip().split(',')
    return dict(gpu=name.strip(), uuid=uuid.strip(), torch=torch.__version__, cuda=torch.version.cuda)


def verify_hardware(protocol):
    if hardware_identity() != protocol['paired_execution']:
        raise RuntimeError('Paired GPU/software identity changed; do not mix these runs')


def schedules(foreground_by_case, seed=0):
    """25 unique anchors/crop: all FG slices plus a random subset of empty slices.

    This intentionally changes the finite-budget sampling distribution; no claim
    of preserving the old uniform full-volume objective is made. Both arms use
    the exact same tape. Calibration approximates its FG/empty mixture per crop.
    """
    if len(foreground_by_case) != 10:
        raise ValueError('This pilot requires the frozen ten training crops')
    rng = random.Random(20261113 + seed)
    pools, training, counts = {}, [], {}
    for case, flags in sorted(foreground_by_case.items()):
        fg = [i for i, v in enumerate(flags) if v]
        bg = [i for i, v in enumerate(flags) if not v]
        if len(flags) != 32 or not 0 < len(fg) < 25:
            raise ValueError('Unsupported foreground extent: '+case)
        chosen_bg = rng.sample(bg, 25-len(fg))
        pools[case] = (fg, chosen_bg)
        training.extend([[case, i] for i in fg+chosen_bg])
        counts[case] = dict(foreground=len(fg), background=len(chosen_bg), total=25)
    rng.shuffle(training)
    # Largest-remainder allocation: 40 probes, 4/crop, proportional FG mixture.
    desired = {c:4*len(pools[c][0])/25 for c in pools}
    allocation = {c:int(x) for c,x in desired.items()}
    remaining = round(sum(desired.values()))-sum(allocation.values())
    for c in sorted(pools, key=lambda c: (-(desired[c]-allocation[c]), c))[:remaining]:
        allocation[c] += 1
    calibration = []
    for case, (fg,bg) in pools.items():
        selected = rng.sample(fg, allocation[case])+rng.sample(bg, 4-allocation[case])
        rng.shuffle(selected)
        calibration.extend([[case,z] for z in selected])
    assert len(training) == 250 and len(calibration) == 40
    assert len({tuple(x) for x in training}) == 250
    return training, calibration, dict(policy='All foreground anchors once; remaining quota sampled without replacement from empty anchors; 25 anchors per crop; globally shuffled',
        objective='Coverage-prioritized empirical slice objective; no importance correction; identical native/constrained sampling',
        counts=counts, foreground_anchors=sum(x['foreground'] for x in counts.values()),
        empty_anchors=sum(x['background'] for x in counts.values()), calibration_foreground=sum(allocation.values()),
        calibration_empty=40-sum(allocation.values()), foreground_anchor_coverage=1., seed=20261113+seed)


def make_protocol(reference, foreground_by_case, arm, hardware):
    if arm not in ('dice','dice_edge_depth'):
        raise ValueError('Only the requested seed-0 pair is permitted')
    p = deepcopy(reference)
    p.pop('arm', None)
    if (p['fold'],p['seed'],p['updates'],p['accumulation'],p['augmentation']) != (0,0,125,2,'none'):
        raise ValueError('Expected original unaugmented seed-0 125-update protocol')
    for key in ('augmentation_control_archive','augmentation_control_sha256','matched_native_archive','matched_native_sha256'):
        p.pop(key,None)
    schedule, calibration, audit = schedules(foreground_by_case, p['seed'])
    recipe = {**RECIPE, 'native_control':True} if arm == 'dice' else dict(RECIPE)
    p.update(name=CAMPAIGN+'__'+arm, frozen_sample_schedule=schedule,
        frozen_calibration_schedule=calibration, coverage_sampling=audit,
        sampling=audit['policy'], constraint_recipe=recipe, constraint_warmup_updates=20,
        gradient_ratio_targets={}, gradient_ratio_p95_caps={}, requested_arms=[arm],
        baseline_archive_sha256=None, paired_execution=hardware,
        comparison='Fresh same-GPU native versus EDGE+DEPTH;125 updates;coverage-first tape;fresh training-only calibration;no augmentation',
        budget='125 optimizer updates, accumulation2; unchanged native LR prefix and optimizer; calibration/constraint overhead separate')
    return p


def install_calibration(engine, protocol, output):
    """Fresh original gradient calibration plus the existing exact logging check."""
    from scripts.colab_hfh import medsam_augmentation_components as logging
    verify_hardware(protocol)
    engine.write(output/'execution_hardware.json', hardware_identity())
    source = inspect.getsource(logging.install_calibration)
    old = '    original_aug.install_calibration(engine, protocol, output)'
    if source.count(old) != 1:
        raise ValueError('Logging preflight calibration hook changed')
    source = source.replace(old, '    # Retain the fresh full_runtime.calibrate already installed.')
    namespace = dict(logging.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    namespace['install_calibration'](engine, protocol, output)
    reference = engine.calibrate

    def calibrated(model, criterion, parameters, slabs, current, arm, destination):
        result = reference(model, criterion, parameters, slabs, current, arm, destination)
        if arm != 'dice':
            if [[r['case'],r['anchor']] for r in result['rows']] != protocol['frozen_calibration_schedule']:
                raise ValueError('Fresh calibration did not use coverage distribution')
            names = result['names']
            coef = np.array([0.]+[result['weights'][n] for n in names[1:]])
            ratios, cosines = [], []
            for row in result['rows']:
                gram = np.array(row['gram'])
                a, b = np.sqrt(max(coef @ gram @ coef,0)), np.sqrt(gram[0,0])
                ratios.append(float(a/b))
                cosines.append(float(gram[0] @ coef / max(a*b,1e-30)))
            engine.write(output/'calibration_joint_audit.json', dict(
                scope='Initial training-only gradients; not gradients during optimization',
                median=float(np.median(ratios)), p95=float(np.quantile(ratios,.95)), maximum=max(ratios),
                negative_cosines=sum(c<0 for c in cosines), weights=result['weights'],
                note='Original sequential per-term target .1 and p95 cap .5 retained; no new joint clipping'))
        engine.write(output/'coverage_sampling.json', protocol['coverage_sampling'])
        engine.write(output/'calibration_schedule.json', protocol['frozen_calibration_schedule'])
        return result
    engine.calibrate = calibrated


def coverage_slabs():
    from scripts.colab_hfh.medsam_constraints_support import PairedSlabs

    class CoverageSlabs(PairedSlabs):
        def calibration_schedule(self, count):
            order = self.protocol['frozen_calibration_schedule']
            if count != 4 or len(order) != 40 or any(c not in self.indices or not 0<=z<len(self.indices[c]) for c,z in order):
                raise ValueError('Invalid training-only coverage calibration')
            return order
    return CoverageSlabs
