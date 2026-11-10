"""Fixed missing-task selection; historical folders are never consulted."""
from copy import deepcopy


def restrict(plan, seeds, scope):
    allowed = {t['id']: t for t in scope['missing_tasks'] if t['seed'] in seeds}
    selected = []
    for task in plan['tasks']:
        if task['id'] in allowed:
            if task != allowed[task['id']]:
                raise ValueError('Frozen missing-task identity differs: '+task['id'])
            selected.append(task)
    if {t['id'] for t in selected} != set(allowed):
        raise ValueError('Missing-task manifest does not match the scientific bundle')
    result = deepcopy(plan)
    result.update(tasks=selected, seeds=list(seeds), num_runs=len(selected))
    return result


def historical_baselines(scope, seeds):
    return [t for t in scope['excluded_completed_tasks'] if t['seed'] in seeds and t['arm']=='dice']


def merged_rows(new_rows, reference, scope):
    allowed={t['id'] for t in scope['missing_tasks']}
    old={t['id'] for t in scope['excluded_completed_tasks']}
    if allowed & old or len(allowed)+len(old)!=504:
        raise ValueError('Invalid exclusion manifest')
    if {r['id'] for r in reference} != old or len(reference)!=len(old):
        raise ValueError('Historical summary differs from fixed exclusions')
    seen=set()
    for row in new_rows:
        if row['id'] not in allowed or row['id'] in seen:
            raise ValueError('Duplicate or unauthorized new result')
        seen.add(row['id'])
    return [*deepcopy(reference), *deepcopy(new_rows)]
