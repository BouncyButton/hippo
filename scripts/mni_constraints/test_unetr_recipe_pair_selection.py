"""A changed recipe must use a fresh matching control for every paired decision."""
from scripts.mni_constraints.unetr_recipe_pair_selection import select


def run():
    tasks=[];candidates=['a_bands','a_full','b_bands','b_full']
    for recipe in ('a','b'):
        for n in range(9):
            common=dict(recipe=recipe,seed=n//3,fold=n%3)
            baseline=f'{recipe}_base_{n}'
            tasks.append(dict(id=baseline,role='baseline',**common))
            for arm in ('bands','full'):
                tasks.append(dict(id=f'{recipe}_{arm}_{n}',candidate=f'{recipe}_{arm}',
                                  baseline_id=baseline,role='constraint',**common))
    state=select(tasks,{},candidates)
    assert [t['id'] for t in state['pending']]==['a_base_0','a_bands_0','a_full_0','b_base_0','b_bands_0','b_full_0']
    state=select(tasks,{'a_bands_0':.9},candidates)
    assert not state['rows'] and state['pending'][0]['id']=='a_base_0'
    completed={'a_base_0':.7,'a_bands_0':.69,'a_full_0':.71}
    state=select(tasks,completed,candidates)
    assert state['summary'][0]['rejected'] and state['summary'][1]['pairs']==1
    assert state['pending'][0]['id']=='b_base_0'  # Screen untouched recipe first.
    assert not any(t.get('candidate')=='a_bands' for t in state['pending'])
    for n in range(9):completed.update({f'a_base_{n}':.7,f'a_full_{n}':.71})
    state=select(tasks,completed,candidates)
    assert state['complete'] and state['selected']['winner']=='a_full' and not state['pending']
    failed={'a_base_0':.7,'a_bands_0':.7,'a_full_0':.69,'b_base_0':.6,'b_bands_0':.6,'b_full_0':.59}
    assert select(tasks,failed,candidates)['outcome']=='exhausted'
    print('Fresh control matching, unpaired-score exclusion, reuse, rejection and9-pair success checks passed.')


if __name__=='__main__':run()
