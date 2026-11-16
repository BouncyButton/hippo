"""Seed-specific timing search, with a shared activation across three folds."""
from copy import deepcopy
from statistics import mean
from scripts.colab_hfh.nnunet_search13_shared_policy import CAP,recost

NAME='nnunet_timing_seed_search_a100_v1'


def set_prefix(original,prefix):
    p=deepcopy(original);arm=p['active_objective'];s=p['models']['nnunet']['objectives'][arm]
    p['prefix_anchor_updates']=prefix;p['search_recipe']['prefix']=prefix
    s['prefix_updates']=s['sdf_prefix_updates']=prefix
    recost(s)
    if prefix<1 or s['max_steps']-prefix-p['constraint_warmup_updates']+1<15:
        raise ValueError('Insufficient full-strength updates')
    return p


def candidate_prefixes(original):
    old=original['prefix_anchor_updates']
    feasible=[]
    for prefix in range(1,90):
        try:set_prefix(original,prefix)
        except ValueError:continue
        if prefix!=old:feasible.append(prefix)
    # Prefer later onset, then earlier coarse alternatives. If necessary refine
    # every remaining integer; the finite domain prevents an unbounded search.
    preferred=[30,40,45,50,55,60,65,70,25,20,15,10,5]
    later=[p for p in preferred if p>old and p in feasible]
    earlier=[p for p in preferred if p<old and p in feasible]
    coarse=later+earlier
    return coarse+[p for p in sorted(feasible,key=lambda p:(abs(p-old),-p)) if p not in coarse]


def summarize_groups(groups):
    result=[]
    for key,g in groups.items():
        completed=[a for a in g['attempts'] if len(a['results'])==3]
        best=max(completed,key=lambda a:mean(r['native_gain_pp'] for r in a['results']))
        chosen=next((a for a in completed if a.get('chosen')),best)
        gains=[r['native_gain_pp'] for r in chosen['results']]
        result.append(dict(key=key,arm=g['arm'],seed=g['seed'],status=g['status'],
            selected_prefix=chosen['prefix'],mean_gain_pp=mean(gains),worst_fold_gain_pp=min(gains),
            positive_folds=sum(x>0 for x in gains),fold_results=chosen['results'],
            completed_timing_candidates=len(completed),remaining_candidates=len(g['remaining_prefixes'])))
    return result
