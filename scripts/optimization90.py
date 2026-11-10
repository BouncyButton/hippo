"""Optimization-only schedule arithmetic for campaign90 models."""

from campaign90 import BUDGET, MODELS

CAMPAIGN = 'four_models_six_optimization_31976_20261006'
SCOPE = 'optimization_only'


def schedule_costs(budget, baseline, update, calibration, prefix=50):
    prefix = prefix if calibration else 0
    if budget < prefix * baseline:
        raise ValueError('Budget cannot fit the fixed calibration anchor')
    steps = prefix + (budget - prefix * baseline) // update
    training = prefix * baseline + (steps - prefix) * update
    return dict(max_steps=steps, prefix_updates=prefix,
                baseline_flops_per_update=baseline, flops_per_update=update,
                calibration_flops=calibration, consumed_flops=training,
                consumed_training_flops=training,
                consumed_total_flops=training + calibration,
                unused_flops=budget - training, patient_pass_epochs=steps / 10,
                budget_scope=SCOPE)


def validation_updates(protocol, arm, cap=None):
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
    maximum = schedule['max_steps'] if cap is None else min(cap, schedule['max_steps'])
    prefix, f0, f1 = (schedule[k] for k in
                     ('prefix_updates', 'baseline_flops_per_update', 'flops_per_update'))
    result = {maximum}
    for target in protocol['validation_target_flops']:
        step = target // f0 if target <= prefix*f0 else prefix + (target-prefix*f0)//f1
        if step > 0:
            result.add(min(maximum, step))
    stride = 5 if protocol['model_identity']=='nnunet' else max(
        5, (BUDGET//MODELS[protocol['model_identity']]['baseline_flops'])//20)
    result.update(range(stride, maximum+1, stride))
    return sorted(result)
