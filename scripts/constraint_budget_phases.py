"""Candidate compute-relative phases; does not alter existing experiments."""
from fractions import Fraction

from optimization90 import schedule_costs


def budget_relative_schedule(budget, baseline_cost, objective_cost, calibration_cost,
                             prefix_fraction='0.1', ramp_fraction='0.1'):
    """Allocate calibration anchor and ramp by common fractions of training FLOPs.

    Prefix uses completed baseline updates at or below its compute target.
    Ramp rounds up to reach its target and follows the calibrated objective.
    Calibration remains separately reported under optimization-only accounting.
    """
    pf, rf = Fraction(prefix_fraction), Fraction(ramp_fraction)
    if min(budget, baseline_cost, objective_cost) <= 0 or calibration_cost < 0:
        raise ValueError('Costs must be positive; calibration cost nonnegative')
    if not 0 < pf < 1 or not 0 < rf < 1 or pf + rf >= 1:
        raise ValueError('Positive phase fractions must leave a full-strength tail')
    if not calibration_cost:
        return {**schedule_costs(budget,baseline_cost,objective_cost,0),
                'constraint_warmup_updates':0,'full_strength_updates':0}
    prefix = max(1, int(Fraction(budget)*pf/baseline_cost))
    ramp_ratio = Fraction(budget)*rf/objective_cost
    ramp = max(1, -(-ramp_ratio.numerator//ramp_ratio.denominator))
    schedule = schedule_costs(budget,baseline_cost,objective_cost,calibration_cost,prefix)
    if schedule['max_steps'] <= prefix+ramp:
        raise ValueError('Cap cannot fit the candidate phases and a full-strength tail')
    schedule.update(constraint_warmup_updates=ramp,
                    full_strength_updates=schedule['max_steps']-prefix-ramp+1,
                    prefix_fraction=str(pf),ramp_fraction=str(rf),
                    candidate_phase_policy='compute_relative_10pct_prefix_10pct_ramp')
    return schedule
