"""Piecewise optimization accounting and predeclared SDF activation."""
from campaign90 import BUDGET


def charge_for_step(charged, schedule):
    if charged < schedule['prefix_updates']:
        return schedule['baseline_flops_per_update']
    if charged < schedule['sdf_prefix_updates']:
        return schedule['early_flops_per_update']
    return schedule['flops_per_update']


def training_flops(attempts, schedule):
    prefix = schedule['prefix_updates']
    late = schedule['sdf_prefix_updates']
    return (min(attempts, prefix) * schedule['baseline_flops_per_update']
            + max(0, min(attempts, late) - prefix) * schedule['early_flops_per_update']
            + max(0, attempts - late) * schedule['flops_per_update'])


def schedule_costs(baseline, early, full, calibration, prefix, ramp, has_sdf, has_depth):
    # Count independent SDF ramp division and scalar coefficient multiplication;
    # the depth coefficient-times-ramp multiply is a third host FP operation.
    early += int(has_depth)
    full += int(has_depth) + 2 * int(has_sdf)
    result = dict(prefix_updates=prefix, sdf_prefix_updates=prefix,
                  baseline_flops_per_update=baseline, early_flops_per_update=early,
                  flops_per_update=full, calibration_flops=calibration,
                  constraint_warmup_updates=ramp, sdf_warmup_updates=ramp,
                  budget_scope='optimization_only',
                  sdf_delay_fraction_numerator=5, sdf_delay_fraction_denominator=9)
    if has_sdf:
        target = BUDGET * 5 // 9
        fraction = prefix + max(0, target - prefix * baseline) // early
        candidates = (fraction, fraction + 1)
        result['sdf_prefix_updates'] = min(candidates, key=lambda n: (
            abs(prefix * baseline + (n - prefix) * early - target), n))
        if result['sdf_prefix_updates'] < prefix + ramp:
            raise ValueError('SDF onset overlaps the retained early ramp')
    steps = 0
    while training_flops(steps + 1, result) <= BUDGET:
        steps += 1
    if has_sdf and steps < result['sdf_prefix_updates'] + ramp:
        raise ValueError('Allowance cannot fit the retained SDF ramp')
    consumed = training_flops(steps, result)
    result.update(max_steps=steps, consumed_training_flops=consumed,
                  consumed_flops=consumed, consumed_total_flops=consumed + calibration,
                  unused_flops=BUDGET - consumed, patient_pass_epochs=steps / 10)
    return result


def sdf_scale_for_step(charged, schedule):
    if charged < schedule['sdf_prefix_updates']:
        return 0.
    return min(1., (charged - schedule['sdf_prefix_updates'] + 1) / schedule['sdf_warmup_updates'])


def validation_updates(protocol, arm, cap=None):
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
    maximum = min(cap, schedule['max_steps']) if cap is not None else schedule['max_steps']
    stride = max(5, (BUDGET // schedule['baseline_flops_per_update']) // 20)
    checkpoints = set(range(stride, maximum + 1, stride)) | {maximum, schedule['prefix_updates']}
    reference_endpoint = protocol.get('comparison_endpoints', {}).get(arm)
    if reference_endpoint is not None:
        checkpoints.add(reference_endpoint)
    late = schedule['sdf_prefix_updates']
    checkpoints.update((late, late + 1, late + schedule['sdf_warmup_updates']))
    for target in protocol['validation_target_flops']:
        eligible = [n for n in range(1, maximum + 1) if training_flops(n, schedule) <= target]
        if eligible:
            checkpoints.add(eligible[-1])
    return sorted(n for n in checkpoints if 0 < n <= maximum)
