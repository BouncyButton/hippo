"""Schedule fresh matched controls when changing a UNETR base recipe."""
import math
from statistics import mean


def select(tasks, completed, candidates):
    """Completed maps task ID to verified final validation Dice in [0,1].

    Constraint tasks name a baseline_id with the SAME recipe, seed and fold.
    A baseline and its constraint run can execute concurrently; an unpaired
    score never qualifies or rejects a candidate. No old-recipe controls enter
    the paired-gain decision.
    """
    by_id={t['id']:t for t in tasks};assert len(by_id)==len(tasks)
    assert set(completed)<=set(by_id)
    assert all(math.isfinite(v) and 0<=v<=1 for v in completed.values())
    rows=[];groups={}
    for candidate in candidates:
        group=[t for t in tasks if t.get('candidate')==candidate and t['role']=='constraint']
        assert len(group)==9 and len({(t['seed'],t['fold']) for t in group})==9
        groups[candidate]=group
        for task in group:
            baseline=by_id[task['baseline_id']]
            assert baseline['role']=='baseline'
            assert all(task[k]==baseline[k] for k in ('recipe','seed','fold'))
            if task['id'] not in completed or baseline['id'] not in completed:continue
            rows.append(dict(**task,baseline_dice=completed[baseline['id']],
                validation=completed[task['id']],gain_pp=100*(completed[task['id']]-completed[baseline['id']])))
    summary=[]
    for candidate in candidates:
        group=[r for r in rows if r['candidate']==candidate]
        summary.append(dict(candidate=candidate,pairs=len(group),
            rejected=any(r['gain_pp']<=0 for r in group),
            mean_gain_pp=mean(r['gain_pp'] for r in group) if group else None,
            minimum_gain_pp=min((r['gain_pp'] for r in group),default=None),
            all_nine_improved=len(group)==9 and all(r['gain_pp']>0 for r in group)))
    winner=next((s['candidate'] for s in summary if s['all_nine_improved']),None)
    viable=[s for s in summary if not s['rejected'] and s['pairs']<9]
    complete=winner is not None or not viable
    pending=[];seen=set();paired={r['id'] for r in rows}
    if not complete:
        # Spread screening across recipes before allocating later pairs.
        for item in sorted(viable,key=lambda s:(s['pairs'],candidates.index(s['candidate']))):
            task=next(t for t in groups[item['candidate']] if t['id'] not in paired)
            for identifier in (task['baseline_id'],task['id']):
                if identifier not in completed and identifier not in seen:
                    pending.append(by_id[identifier]);seen.add(identifier)
    return dict(rows=rows,summary=summary,pending=pending,complete=complete,
        selected=dict(candidate=winner,winner=winner),
        outcome='nine_pair_success' if winner else 'exhausted' if complete else 'running')
