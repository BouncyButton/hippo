"""Piecewise optimization accounting and a delayed SDF coefficient."""
from optimization90 import SCOPE


def charge_for_step(charged, schedule):
    if charged < schedule['prefix_updates']:
        return schedule['baseline_flops_per_update']
    if charged < schedule['sdf_prefix_updates']:
        return schedule['early_flops_per_update']
    return schedule['flops_per_update']


def training_flops(attempts, schedule):
    prefix, sdf_prefix = schedule['prefix_updates'], schedule['sdf_prefix_updates']
    return (min(attempts, prefix) * schedule['baseline_flops_per_update']
            + max(0, min(attempts, sdf_prefix)-prefix) * schedule['early_flops_per_update']
            + max(0, attempts-sdf_prefix) * schedule['flops_per_update'])


def schedule_costs(budget, baseline, bands, bands_sdf, calibration):
    """Use measured native loss graphs, plus ramp division and weight multiplication."""
    if min(budget, baseline, bands, bands_sdf, calibration) <= 0:
        raise ValueError('Positive budget and costs required')
    result = dict(prefix_updates=9, sdf_prefix_updates=50,
                  baseline_flops_per_update=baseline, early_flops_per_update=bands,
                  flops_per_update=bands_sdf+2, sdf_weight_selection_flops_per_update=2,
                  calibration_flops=calibration, constraint_warmup_updates=9,
                  sdf_warmup_updates=9, budget_scope=SCOPE)
    steps = 0
    while training_flops(steps+1, result) <= budget:
        steps += 1
    if steps < 59:
        raise ValueError('Insufficient allowance for the delayed SDF ramp')
    consumed = training_flops(steps, result)
    result.update(max_steps=steps, consumed_flops=consumed,
                  consumed_training_flops=consumed,
                  consumed_total_flops=consumed+calibration,
                  unused_flops=budget-consumed, patient_pass_epochs=steps/10,
                  bands_full_strength_updates=steps-18+1,
                  sdf_full_strength_updates=steps-59+1)
    return result


def validation_updates(protocol, arm, cap=None):
    schedule = protocol['models']['nnunet']['objectives'][arm]
    maximum = schedule['max_steps'] if cap is None else min(cap, schedule['max_steps'])
    result = {maximum}
    for target in protocol['validation_target_flops']:
        eligible = [i for i in range(1, maximum+1) if training_flops(i, schedule) <= target]
        if eligible:
            result.add(eligible[-1])
    result.update(range(5, maximum+1, 5))
    return sorted(result)


def sdf_scale_for_step(charged, schedule):
    if charged < schedule['sdf_prefix_updates']:
        return 0.
    return min(1., (charged-schedule['sdf_prefix_updates']+1)/schedule['sdf_warmup_updates'])


def native_training_loss(outputs, labels, arm, weights, scale, criterion,
                         target_sdf=None, sdf_scale=0.):
    """Keep the original tensor loss graph; alter only SDF's scalar coefficient."""
    from nnunet_multi_objective import training_loss

    if 'sdf' not in arm:
        return training_loss(outputs, labels, arm, weights, scale, criterion, target_sdf)
    if scale != 1. or not 0 < sdf_scale <= 1:
        raise ValueError('SDF must activate after the early constraint ramp')
    # The calibrated base weight stays fixed. This one host multiply implements
    # the original nine-step SDF ramp independently of the already-full bands.
    adjusted = {**weights, 'sdf': weights['sdf'] * sdf_scale}
    return training_loss(outputs, labels, arm, adjusted, scale, criterion, target_sdf)
