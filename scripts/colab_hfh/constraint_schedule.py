"""Cost-capped HFH schedules for the declared multi-constraint objectives."""
from copy import deepcopy

from scripts.colab_hfh.constraint_calibration import components, extra_calibration_flops


def training_flops(updates, schedule):
    prefix, late = schedule['prefix_updates'], schedule['sdf_prefix_updates']
    return (min(updates, prefix)*schedule['baseline_flops_per_update']
            + max(0, min(updates, late)-prefix)*schedule['early_flops_per_update']
            + max(0, updates-late)*schedule['flops_per_update'])


def make_schedule(costs, arm, *, budget, prefix, ramp, inner_weight=.5, sdf_ramp=None,
                  online_sdf_ratio=None, adaptive_sides=False):
    names = components(arm)
    sdf_ramp = ramp if sdf_ramp is None else sdf_ramp
    if prefix < 0 or ramp < 1 or sdf_ramp < 1 or budget <= 0:
        raise ValueError('Invalid optimization schedule')
    early_arm = arm.replace('_sdf', '')
    has_sdf, has_depth = 'sdf' in names, 'depth' in names
    native = costs['dice']['update_flops']
    early = costs[early_arm]['update_flops'] + int(has_depth)
    full = costs[arm]['update_flops'] + int(has_depth) + 2*int(has_sdf)
    probe = 0
    if online_sdf_ratio is not None:
        if arm != 'dice_sdf' or not 0 < online_sdf_ratio <= .3:
            raise ValueError('Online SDF pilot requires SDF alone and a ratio in (0,.3]')
        from scripts.colab_hfh.constraint_online_sdf import probe_allowance
        probe = probe_allowance()
        full += probe
    calibration = costs[arm]['calibration_flops'] + extra_calibration_flops(arm)
    if not 0 < inner_weight < 1 or (inner_weight != .5 and 'bands' not in names):
        raise ValueError('Inner/outer weights apply only to a BAND component')
    if inner_weight != .5:
        early += 16;full += 16;calibration += 160
    if adaptive_sides:
        if 'bands' not in names or inner_weight != .5:
            raise ValueError('Adaptive side weighting requires an otherwise balanced BAND arm')
        # B1, one foreground class: detached scalar ratio/clamp and both
        # weighted side derivatives. Conservatively charge 64 operations.
        early += 64;full += 64;calibration += 640
    late = prefix
    if has_sdf:
        target = budget*5//9
        fraction = prefix + max(0, target-prefix*native)//early
        nearest = min((fraction, fraction+1), key=lambda n: (abs(prefix*native+(n-prefix)*early-target), n))
        late = max(prefix+ramp, nearest)
    result = dict(prefix_updates=prefix, sdf_prefix_updates=late,
                  baseline_flops_per_update=native, early_flops_per_update=early,
                  flops_per_update=full, calibration_flops=calibration,
                  constraint_warmup_updates=ramp, sdf_warmup_updates=sdf_ramp,
                  budget_scope='optimization_only',
                  sdf_delay_rule='nearest_5_of_9_compute_but_after_completed_early_ramp')
    if probe:
        result.update(online_sdf_probe_flops_per_active_update=probe,
                      online_sdf_gradient_ratio=online_sdf_ratio)
    steps = 0
    while training_flops(steps+1, result) <= budget:
        steps += 1
    if steps < prefix+ramp or (has_sdf and steps < late+sdf_ramp):
        raise ValueError('Budget cannot fit the prescribed constraint ramp')
    used = training_flops(steps, result)
    result.update(max_steps=steps, consumed_training_flops=used, consumed_flops=used,
                  consumed_total_flops=used+calibration, unused_flops=budget-used,
                  patient_pass_epochs=steps/10)
    return result


def make_payload(base, model, arm, costs, *, prefix, ramp, target, cap, screening=True, inner_weight=.5,
                 component_targets=None, sdf_ramp=None, online_sdf_ratio=None, adaptive_sides=False):
    """One objective, one recipe and the same nine fixed identities."""
    names = components(arm)
    if online_sdf_ratio is not None and (model != 'swinunetr' or arm != 'dice_sdf'):
        raise ValueError('Online strength pilot is restricted to SwinUNETR SDF alone')
    if component_targets is None:
        component_targets = {name: target for name in names}
    if set(component_targets) != set(names) or any(not 0 < value <= 1 for value in component_targets.values()):
        raise ValueError('Every active component needs an explicit finite positive target <=1')
    payload = deepcopy(base)
    payload['rates'] = {model: base['rates'][model]}
    payload['protocols'] = {model: payload['protocols'][model]}
    payload['references'] = {key: value for key, value in payload['references'].items()
                             if key.startswith(model+'_f')}
    payload.update(objective=arm, folds=[0, 2, 1], seeds=[1, 0, 2],
                   constraint_matrix=True, strict_search=False,
                   campaign_phase='screening' if screening else 'frozen_full_matrix')
    for key in ('pairs', 'reject_nonpositive_fold', 'min_pair_delta_pp'):
        payload.pop(key, None)
    if screening:
        payload['min_pair_delta_pp'] = 1.
    protocol = payload['protocols'][model]['protocol.json']
    # No optimizer resets or validation-pair-dependent loss variants carry over.
    for key in list(protocol):
        if key.startswith('band_') or key == 'control_kind':
            protocol.pop(key)
    schedule = make_schedule(costs, arm, budget=protocol['budget_flops'], prefix=prefix, ramp=ramp,
                             inner_weight=inner_weight, sdf_ramp=sdf_ramp, online_sdf_ratio=online_sdf_ratio,
                             adaptive_sides=adaptive_sides)
    protocol['models'][protocol['trainer_alias']]['objectives'] = {arm: schedule}
    protocol.update(active_objective=arm, constraint_warmup_updates=ramp,
                    calibration_target_ratio=target, calibration_p95_cap=cap,
                    constraint_component_targets=component_targets,
                    constraint_joint_target=target, constraint_joint_cap=cap,
                    constraint_calibration='joint_logit_gradient_all_ramp_subsets_v1',
                    constraint_geometry_units='physical_mm_hfh_1x2x1',
                    depth_geometry=dict(band_mm=2., roi_margin_mm=8., sparse_band_voxels=64),
                    sdf_policy={arm: 'delayed_5_of_9' if 'sdf' in names else 'not_applicable'},
                    endpoint='Final-budget checkpoint; unchanged native LR horizon; exploratory validation selection',
                    constraint_definition='Canonical binary losses; joint training-only calibration; physical HFH SDF/depth; unchanged native optimizer')
    if inner_weight != .5:
        protocol.update(band_inner_weight=inner_weight,
                        variant_extra_scalar_flops_allowance=16)
        protocol['constraint_definition'] += f'; BAND inner/outer weights {inner_weight}/{1-inner_weight}; training-only joint calibration, same weights in every pair'
    protocol['name'] = f'HFH_constraints_{model}_{arm}_t{target}_p{prefix}_r{ramp}'
    if online_sdf_ratio is not None:
        protocol.update(online_sdf_gradient_ratio=online_sdf_ratio,online_sdf_max_multiplier=64.,
            online_sdf_definition='Per-training-batch detached FP32 native/SDF logit-gradient RMS ratio; no validation input; preserve native loss path; cap coefficient at64x frozen anchor; charge probe arithmetic within optimization cap')
        protocol['name'] += f'_online{online_sdf_ratio}'
        protocol['constraint_definition'] += '; '+protocol['online_sdf_definition']
    if inner_weight != .5:
        protocol['name'] += f'_inner{inner_weight}'
    if adaptive_sides:
        protocol.update(band_adaptive_sides=True, variant_extra_scalar_flops_allowance=64)
        protocol['name'] += '_adaptive_sides'
        protocol['constraint_definition'] += ('; adaptive BAND ablation: each training case uses '
            'inner weight clamp((detached inner BCE+1e-6)/(detached inner BCE+detached outer BCE+2e-6),.1,.9); '
            'outer weight is its complement; no gradient through weights; same rule in calibration and training; '
            'no validation input; extra scalar arithmetic charged within optimization cap')
    payload['version'] = protocol['name']
    return payload
