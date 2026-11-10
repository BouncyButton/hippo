"""Reject a shared recipe after any nonpositive pair; retain every attempted run."""
import math
from statistics import mean


def select(tasks, rows, candidates, pilots=4):
    """Rows contain archived results only. Candidate order is fixed before launch."""
    assert len({r['id'] for r in rows}) == len(rows)
    expected_by_id = {t['id']:t for t in tasks}
    for row in rows:
        assert math.isfinite(row['gain_pp'])
        assert all(row[k] == expected_by_id[row['id']][k] for k in ('candidate', 'seed', 'fold'))
    summaries = []
    for candidate in candidates:
        group = [r for r in rows if r['candidate'] == candidate]
        expected = [t for t in tasks if t['candidate'] == candidate]
        assert len(expected) == 9
        assert {(r['seed'], r['fold']) for r in group} <= {
            (t['seed'], t['fold']) for t in expected}
        failed = any(r['gain_pp'] <= 0 for r in group)
        summaries.append(dict(candidate=candidate, pairs=len(group), rejected=failed,
            minimum_gain_pp=min((r['gain_pp'] for r in group), default=None),
            mean_gain_pp=mean(r['gain_pp'] for r in group) if group else None,
            all_nine_improved=len(group) == 9 and not failed))
    winner = next((s['candidate'] for s in summaries if s['all_nine_improved']), None)
    viable = [s['candidate'] for s in summaries if not s['rejected'] and s['pairs'] < 9]
    finished = winner is not None or not viable
    completed = {r['id'] for r in rows}
    pending = []
    if not finished:
        for candidate in viable:
            group = [t for t in tasks if t['candidate'] == candidate]
            missing = [t for t in group if t['id'] not in completed]
            # Prefer different recipes. With only one survivor, both reserved
            # GPUs may test its remaining pilots; retain any already-started
            # result if its partner rejects the candidate.
            done = sum(t['id'] in completed for t in group[:pilots])
            missing_pilots = [t for t in group[:pilots] if t['id'] not in completed]
            pending.extend(missing if done == pilots else missing_pilots[:2 if len(viable)==1 else 1])
    return dict(rows=rows, summary=summaries, pending=pending,
        selected=dict(candidate=winner, winner=winner), complete=finished,
        outcome='nine_pair_success' if winner else 'exhausted' if finished else 'running')
