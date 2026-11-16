"""Runtime utilities for the archived HFH protocol."""
READY_POLICY = dict(panel='All ten frozen training crops, observed during each training pass',
                    minimum_mean_dice=.5, minimum_ready_cases=8, case_dice=.5,
                    check_every_updates=10, latch=True, validation_feedback=False)

def readiness(rows):
    """A completed pass over ten distinct cases, never a single easy crop."""
    return (len(rows) == 10 and len({r['case_name'] for r in rows}) == 10
            and sum(r['hard_dice'] for r in rows)/10 >= .5
            and sum(r['hard_dice'] >= .5 for r in rows) >= 8)
