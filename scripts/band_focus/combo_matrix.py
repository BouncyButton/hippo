"""Bounded additions to the frozen 81.32397% outer-only objective."""
from copy import deepcopy

ANCHOR = 'w81_strength_qi70_qo10_mi65_mo85_b125_p75'
CONTROL = 'combo_control'
VERSION = 'outer_combo_v1'
TERMS = ('sdf', 'edge', 'depth')
SUBSETS = (('sdf',), ('edge',), ('depth',), ('sdf', 'edge'),
           ('sdf', 'depth'), ('edge', 'depth'), TERMS)
SOURCES = ('combo_matrix.py', 'combo_objectives.py', 'combo_train.py',
           'combo_campaign.py', 'combo_replication.py', 'combo_tests.py', 'run_combo.sh')


def entries(base, coefficients):
    assert base['id'] == ANCHOR and base['objective_version'] == 'winner_v1'
    assert set(coefficients) == {'bands', 'pair'}
    result = []
    for ratio, subset in [(0., ())] + [(r, s) for r in (.025, .05, .1) for s in SUBSETS]:
        entry = deepcopy(base)
        name = CONTROL if not subset else 'combo_' + '_'.join(subset) + f'_g{round(ratio*1000):03}'
        entry.update(id=name, family='control' if not subset else 'outer_combo',
                     objective_version=VERSION, base_objective=deepcopy(base),
                     base_coefficients=dict(coefficients), compare=CONTROL,
                     combo=dict(terms=list(subset), gradient_ratio=ratio),
                     idea='Frozen winning outer loss plus jointly calibrated established auxiliaries.')
        result.append(entry)
    assert len(result) == len({e['id'] for e in result}) == 22
    return result
