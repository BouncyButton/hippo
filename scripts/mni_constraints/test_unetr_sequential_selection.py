"""Check stopping decisions, retained failures, and nine-pair success criteria."""
from scripts.mni_constraints.unetr_sequential_selection import select


def run():
    candidates = ['first', 'second']
    tasks = [dict(id=f'{c}_{n}', candidate=c, seed=n//3, fold=n%3)
             for c in candidates for n in range(9)]
    result = select(tasks, [], candidates)
    assert [t['id'] for t in result['pending']] == ['first_0', 'second_0']
    rows = [dict(tasks[0], gain_pp=-.1)]
    result = select(tasks, rows, candidates)
    assert result['rows'] == rows and not result['complete']
    assert [t['id'] for t in result['pending']] == ['second_0','second_1']
    almost=select(tasks,rows+[dict(t,gain_pp=.2) for t in tasks[9:12]],candidates)
    assert [t['id'] for t in almost['pending']]==['second_3']  # No confirmation before fourth pilot.
    rows += [dict(t, gain_pp=.2) for t in tasks[9:13]]
    result = select(tasks, rows, candidates)
    assert [t['id'] for t in result['pending']] == [f'second_{n}' for n in range(4,9)]
    failure = select(tasks, rows+[dict(tasks[13], gain_pp=0.)], candidates)
    assert failure['complete'] and failure['outcome'] == 'exhausted'
    assert len(failure['rows']) == 6 and not failure['pending']
    success = select(tasks, rows+[dict(t, gain_pp=.1) for t in tasks[13:]], candidates)
    assert success['complete'] and success['selected']['winner'] == 'second'
    assert len(success['rows']) == 10 and not success['pending']
    print('Sequential rejection and nine-pair completion checks passed.')


if __name__ == '__main__':
    run()
