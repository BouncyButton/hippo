"""A shared training-only constraint trigger, independent of validation scores."""
import math

TRAINING_FIELDS = (
    'training_CA1_3_hard_dice',
    'training_subiculum_hard_dice',
    'training_CA4_DG_hard_dice',
)


def regular_steps(cap, baseline_cost, max_steps):
    """The existing 5%-of-budget checks; exclude historical bridge evaluations."""
    return sorted({min(max_steps, (cap*i + 20*baseline_cost-1)//(20*baseline_cost))
                   for i in range(1, 21)})


def onset(rows, threshold, cap, baseline_cost, max_steps, consecutive=2):
    """First regular check with all training subfields above a fixed threshold.

    Only step and the three explicitly named TRAINING columns are inspected.
    Computing this from a matched unconstrained trajectory is equivalent to an
    online trigger while the constrained run is still in its identical prefix.
    A missing observation or threshold never reached is an error, not an excuse
    to pick a favorable validation checkpoint.
    """
    assert 0 < threshold < 1 and consecutive >= 2
    allowed = regular_steps(cap, baseline_cost, max_steps)
    observed = {}
    for row in rows:
        step = int(row['step'])
        if step not in allowed:
            continue
        assert step not in observed, ('duplicate training observation', step)
        values = [float(row[field]) for field in TRAINING_FIELDS]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        observed[step] = min(values)
    assert set(observed) == set(allowed), 'Missing regular training observations'
    streak = 0
    for step in allowed:
        streak = streak + 1 if observed[step] >= threshold else 0
        if streak >= consecutive:
            return step
    raise ValueError('Training readiness threshold never reached')
