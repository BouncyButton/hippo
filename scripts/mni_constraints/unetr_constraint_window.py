"""Optional finite auxiliary-loss window; original constraint formulas unchanged."""
from copy import deepcopy

from scripts.capped_depth_delay.schedule import training_flops


def envelope(maximum, taper_start, stop):
    """Index by already-completed updates; linear fade, then base-only gradients."""
    assert 0 <= taper_start < stop <= maximum
    return [1. if n <= taper_start else max(0., (stop-n)/(stop-taper_start))
            for n in range(maximum)]


def account(schedule, has_sdf, budget):
    """Charge the extra scalar multiplications; keep computing original terms.

    Terms are still evaluated and backpropagated at zero weight. Their original
    profiled FLOPs remain charged, so this changes auxiliary pressure only.
    """
    result = deepcopy(schedule)
    result['baseline_flops_per_update'] += 1
    result['early_flops_per_update'] += 1
    result['flops_per_update'] += 1 + int(has_sdf)
    maximum = schedule['max_steps']
    assert training_flops(maximum, result) <= budget < training_flops(maximum+1, result)
    consumed = training_flops(maximum, result)
    result.update(consumed_training_flops=consumed, consumed_flops=consumed,
        consumed_total_flops=consumed+result['calibration_flops'], unused_flops=budget-consumed)
    return result


def instrument_source(source):
    old = '        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():\n            logits = model(image)\n'
    new = ('        auxiliary_factor = protocol["auxiliary_envelope"][charged]\n'
           '        scale *= auxiliary_factor\n' + old)
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "sdf_scale_for_step(charged, schedule) if 'sdf' in active_arm else 0.)"
    new = "auxiliary_factor * sdf_scale_for_step(charged, schedule) if 'sdf' in active_arm else 0.)"
    assert source.count(old) == 1
    return source.replace(old, new)
