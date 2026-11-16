"""Increase calibrated coefficients without changing successful timing or cost."""
from copy import deepcopy
from statistics import mean

NAME='nnunet_weight_scale_a100_v1'
CAP=31976450667090
MULTIPLIERS=(1.25,1.5,2.)


def scale_protocol(original,multiplier):
    if multiplier not in MULTIPLIERS:raise ValueError('Undeclared weight multiplier')
    p=deepcopy(original)
    # Joint calibration is homogeneous when targets and calibration ceilings
    # scale together. The independent PER-UPDATE activity envelope stays fixed.
    for key in ('calibration_target_ratio','calibration_p95_cap','constraint_joint_target','constraint_joint_cap'):
        p[key]*=multiplier
    p['constraint_component_targets']={n:v*multiplier for n,v in p['constraint_component_targets'].items()}
    p['search_recipe']=dict(p['search_recipe'],target=p['search_recipe']['target']*multiplier,weight_multiplier=multiplier)
    p['weight_multiplier']=multiplier
    return p


def summarize(results,design):
    summaries=[]
    for key,g in design['groups'].items():
        levels=[]
        for multiplier in (1.,*MULTIPLIERS):
            rows=g['reference_results'] if multiplier==1. else [r for r in results if r['group']==key and r['weight_multiplier']==multiplier]
            complete=len(rows)==3
            levels.append(dict(multiplier=multiplier,completed_folds=len(rows),complete=complete,
                mean_gain_pp=mean(r['native_gain_pp'] for r in rows) if complete else None,
                mean_dice=mean(r['validation_dice'] for r in rows) if complete else None,
                worst_fold_gain_pp=min(r['native_gain_pp'] for r in rows) if complete else None,
                positive_folds=sum(r['native_gain_pp']>0 for r in rows),results=rows))
        # 1x is deliberately first, so an exact tie retains the working recipe.
        best=max((v for v in levels if v['complete']),key=lambda v:v['mean_gain_pp'])
        summaries.append(dict(group=key,arm=g['arm'],seed=g['seed'],fixed_prefix=g['prefix'],
            selected_multiplier=best['multiplier'],mean_gain_pp=best['mean_gain_pp'],
            improvement_over_1x_pp=best['mean_gain_pp']-levels[0]['mean_gain_pp'],
            mean_dice=best['mean_dice'],worst_fold_gain_pp=best['worst_fold_gain_pp'],
            positive_folds=best['positive_folds'],fold_results=best['results'],levels=levels))
    return summaries
