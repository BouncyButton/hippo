"""Check that the proposed trigger cannot select on validation or bridge checks."""
from scripts.mni_constraints.unetr_readiness import onset, regular_steps, TRAINING_FIELDS


def main():
    steps = regular_steps(40000, 100, 400)
    rows = [dict(step=s, **{key: .7 if s >= 200 else .4 for key in TRAINING_FIELDS},
                 macro_hard_dice=1. if s < 200 else 0.) for s in steps]
    assert onset(rows, .5, 40000, 100, 400) == 220
    rows.append(dict(step=201, **{key: 1. for key in TRAINING_FIELDS}))
    assert onset(rows, .5, 40000, 100, 400) == 220
    for row in rows:
        row['macro_hard_dice'] = float('nan')
    assert onset(rows, .5, 40000, 100, 400) == 220
    next(r for r in rows if r['step'] == 220)[TRAINING_FIELDS[2]] = .3
    assert onset(rows, .5, 40000, 100, 400) == 260
    try:
        onset(rows, .9, 40000, 100, 400)
    except ValueError:
        pass
    else:
        raise AssertionError('Never-reached threshold silently accepted')
    print('Passed: training-only trigger, bridge exclusion, weakest-class requirement, reset, and no fabricated fallback.')


if __name__ == '__main__':
    main()
