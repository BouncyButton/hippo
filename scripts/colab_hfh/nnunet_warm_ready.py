"""Frozen warm-up/low-LR ablation and training-only foreground readiness."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

NAME = 'nnunet_warm_ready_v1'
FILES = ('nnunet_warm_ready.py', 'nnunet_warm_ready_worker.py', 'nnunet_warm_ready_campaign.py')
# Includes probability monitoring, counters and detached readiness decisions.
MONITOR_FLOPS = 64 * 64**3
READY_POLICY = dict(panel='All ten frozen training crops, observed during each training pass',
                    minimum_mean_dice=.5, minimum_ready_cases=8, case_dice=.5,
                    check_every_updates=10, latch=True, validation_feedback=False)


def readiness(rows):
    """A completed pass over ten distinct cases, never a single easy crop."""
    return (len(rows) == 10 and len({r['case_name'] for r in rows}) == 10
            and sum(r['hard_dice'] for r in rows)/10 >= .5
            and sum(r['hard_dice'] >= .5 for r in rows) >= 8)


def lr(step):
    return .03*(step+1)/5 if step < 5 else .03*(1-(step-5)/85)**.9


def payload(base, plan, arm):
    from scripts.colab_hfh.residual_band_protocol import payload as band_payload
    from scripts.colab_hfh.constraint_schedule import training_flops
    if arm not in ('dice', 'dice_bands'):
        raise ValueError('Only native and conditional BAND were authorized')
    costs = plan['cost_references']['nnunet']['canonical_costs']
    result = band_payload(base, plan)
    p = result['protocols']['nnunet']['protocol.json']
    if arm == 'dice':
        p['models']['nnunet']['objectives']={'dice':p['models']['nnunet']['objectives']['dice_bands']}
        p['sdf_policy']={'dice':'not_applicable'}
        p['constraint_component_targets']={}
    p['active_objective']=arm
    result['objective']=arm
    p.update(name=NAME+'_'+arm, nnunet_warmup_fraction=5/90,
             readiness_policy=READY_POLICY, nnunet_warmup_updates=5,
             constraint_definition='Readiness-gated physical residual BAND with activation-count ramp' if arm != 'dice'
             else 'Native Dice+CE/deep supervision; five-update warm-up and peak LR0.03',
             endpoint='Final original optimization FLOP-cap endpoint; all seeds retained',
             warm_ready_source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in FILES})
    p['initial_lrs']['nnunet'] = .03
    s = p['models']['nnunet']['objectives'][arm]
    # Reserve the full conditional loss cost from the outset. This is an upper
    # bound while readiness is false, never a way to spend beyond the cap.
    cost = (s['flops_per_update'] if arm != 'dice' else costs['dice']['update_flops']) + MONITOR_FLOPS
    s.update(prefix_updates=0, sdf_prefix_updates=0, baseline_flops_per_update=cost,
             early_flops_per_update=cost, flops_per_update=cost,
             readiness_monitor_flops_per_update=MONITOR_FLOPS,
             cost_accounting='Conservative full per-update allowance, including inactive readiness steps')
    steps = p['budget_flops']//cost
    used = steps*cost
    s.update(max_steps=steps, consumed_training_flops=used, consumed_flops=used,
             consumed_total_flops=used+s['calibration_flops'], unused_flops=p['budget_flops']-used,
             patient_pass_epochs=steps/10)
    if arm == 'dice':
        s['calibration_flops']=0; s['consumed_total_flops']=used
    assert steps == (90 if arm == 'dice' else 89)
    assert training_flops(steps,s) <= p['budget_flops'] < training_flops(steps+1,s)
    p['diagnostics'] = dict(parameter_probe_max_flops=16*costs['dice']['update_flops'],
                            logit_probe_max_flops=90*8192*64**3,
                            scope='Read-only gradient probes outside optimization cap; no optimizer updates')
    result.update(version=NAME+'_'+arm, folds=[0], seeds=[0,1,2], rates={'nnunet':.03},
                  interpretation='Paired exploratory schedule ablation; old baseline, new baseline, constrained all reported')
    return result
