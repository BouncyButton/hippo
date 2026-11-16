"""Cross-model shared-cap search; tune constraint exposure instead of cap."""
from copy import deepcopy
from scripts.colab_hfh.nnunet_search13_policy import ARMS,ACTIVITY,probe_allowance,active_coefficients
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.constraint_schedule import training_flops

NAME = 'nnunet_search13_shared_a100_f0s0_v1'
CAP = 31976450667090
SCHEDULE = dict(original_prefix_updates=20,plateau_lr=.07,decay_start=60,horizon=90,power=.9)


def candidates():
    rows = []
    # A broad exposure/strength screen followed by component allocations.
    for label,target,prefix,ramp in [('standard',.10,45,9),('gentle',.05,45,9),
                                   ('strong',.20,45,9),('early',.10,30,18),
                                   ('earliest_gentle',.05,20,20),('late_reference',.10,60,6)]:
        for arm in ARMS:
            rows.append(dict(id=arm+'__'+label,arm=arm,target=target,prefix=prefix,ramp=ramp,emphasis=None))
    for arm in ARMS:
        if len(components(arm)) > 1:
            for name in components(arm):
                rows.append(dict(id=arm+'__emphasize_'+name,arm=arm,target=.1,prefix=45,ramp=9,emphasis=name))
    return rows


def recost(schedule,budget=CAP):
    n = 0
    while training_flops(n+1,schedule) <= budget:n += 1
    used = training_flops(n,schedule)
    schedule.update(max_steps=n,consumed_training_flops=used,consumed_flops=used,
        consumed_total_flops=used+schedule['calibration_flops'],unused_flops=budget-used,patient_pass_epochs=n/10)
    return schedule


def make_protocol(reference,recipe,root):
    p = deepcopy(reference)
    arm = recipe['arm']; names = components(arm)
    if p['active_objective'] != arm:raise ValueError('Wrong canonical reference')
    p.pop('long_schedule',None)
    p.pop('long_source_sha256',None)
    p.update(name=NAME+'__'+recipe['id'],budget_flops=CAP,folds=[0],seeds=[0],
        shared_cap_search=True,native_reference_folder='reference_shared_native',
        lr_horizon_updates=90,seed0_plateau_schedule=SCHEDULE,
        a100_native_anchor=str(root/'runs/native/run'/f"native_update{recipe['prefix']}.json"),
        prefix_anchor_updates=recipe['prefix'],search_recipe=recipe,activity_policy=deepcopy(ACTIVITY),
        minimum_full_strength_updates=15,constraint_warmup_updates=recipe['ramp'],
        calibration_target_ratio=recipe['target'],constraint_joint_target=recipe['target'],
        constraint_component_targets={n:recipe['target']*(2 if n==recipe['emphasis'] else 1) for n in names},
        endpoint='Final shared-cap checkpoint; f0/s0 exploratory search; no intermediate pruning',
        constraint_definition='Same canonical constraints and 31.98T cross-model optimization cap; independently matched native prefixes20/30/45/60; per-update activity floor, all components active together')
    s = p['models']['nnunet']['objectives'][arm]
    s.update(prefix_updates=recipe['prefix'],sdf_prefix_updates=recipe['prefix'],
        constraint_warmup_updates=recipe['ramp'],sdf_warmup_updates=recipe['ramp'],
        activity_probe_flops_per_update=probe_allowance(arm))
    for key in ('early_flops_per_update','flops_per_update'):s[key] += probe_allowance(arm)
    recost(s)
    if s['max_steps']-recipe['prefix']-recipe['ramp']+1 < 15:
        raise ValueError('Insufficient full-strength constraint updates')
    return p
