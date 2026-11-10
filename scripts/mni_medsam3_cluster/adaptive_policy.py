"""Deterministic milestone admission for the user-approved adaptive experiment."""


def choose_seeds(records, threshold=0.60):
    qualified = {}
    for row in records:
        if row['fold'] == 0 and row['validation_dice'] >= threshold:
            seed = row['seed']
            key = (row['observed_at'], row['updates'], seed)
            if seed not in qualified or key < qualified[seed][0]:
                qualified[seed] = (key, row)
    ordered = sorted(qualified, key=lambda s: qualified[s][0])
    return ordered[:2] if len(ordered) >= 2 else []


def next_trials(plan, records, selected, active_ids):
    """Every seed starts at400; only a failed archived milestone unlocks +100."""
    by_pair = {}
    for row in records:
        by_pair.setdefault((row['fold'], row['seed']), []).append(row)
    pairs = ([(0, s) for s in plan['seeds']] if not selected else
             [(0, s) for s in plan['seeds'] if not by_pair.get((0, s))] +
             [(f, s) for f in (1, 2) for s in selected])
    candidates = []
    for fold, seed in pairs:
        history = by_pair.get((fold, seed), [])
        if any(r['validation_dice'] >= plan['threshold'] for r in history):
            continue
        latest = max(history, key=lambda r: r['updates']) if history else None
        updates = latest['updates'] + 100 if latest else 400
        if updates > plan['maximum_updates']:
            continue
        task = next(t for t in plan['tasks'] if (t['fold'], t['seed'], t['updates']) == (fold, seed, updates))
        if task['id'] not in active_ids:
            candidates.append(task)
    return sorted(candidates, key=lambda t: (t['updates'], t['fold'], t['seed']))


def needs_checkpoint(row, plan, selected, transferred):
    if row['validation_dice'] >= plan['threshold'] or row['updates'] >= plan['maximum_updates']:
        return False
    if row['fold'] == 0 and selected:
        return False
    return row['id'] not in transferred
