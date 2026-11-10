"""Bounded, mirrored A/P study and the explicitly selected outer-loss tail."""
from copy import deepcopy
from itertools import product, combinations
import json

VERSION = 'ap_boundary_v1'
ANCHOR = 'w81_emphasis_qi70_qo10_mi65_mo85'
BACKGROUNDS = ('winner', 'dice')


def retained_ids():
    names = []
    for mi, mo in product((50, 65, 75), (75, 85, 90)):
        if (mi, mo) not in ((65, 85), (75, 75)):
            names.append(f'w81_emphasis_qi70_qo10_mi{mi}_mo{mo}')
    names += [f'w81_emphasis_qi{qi}_qo15_mi65_mo85' for qi in (70, 80)]
    for bb, pb in product((75, 100, 125), (75, 100, 125, 150)):
        if (bb, pb) != (100, 100):
            names.append(f'w81_strength_qi70_qo10_mi65_mo85_b{bb}_p{pb}')
    names += [f'w81_mass_qi70_qo10_mi65_mo85_mass{m}_p{pb}'
              for m, pb in product((45, 55), (100, 150))]
    names += [f'w81_width_qi70_qo10_mi65_mo85_i{i}_o{o}'
              for i, o in ((2, 2), (3, 1), (3, 3), (4, 2))]
    names += [f'w81_shell_qi70_qo10_mi65_mo85_{w}_p100'
              for w in ('503515', '333333', '205030')]
    names += [f'w81_insidece_qi70_qo10_mi65_mo85_ce10_mass50_p{p}' for p in (100, 150)]
    names += [f'w81_fractions_qi80_qo{o}_mi75_mo75' for o in (10, 20)]
    names += [f'w81_endpoint_{s}_p100' for s in ('far75', 'near60_30_10', 'separate')]
    assert len(names) == len(set(names)) == 38
    return names


def default_spec():
    return dict(width=3., anterior_fraction=.5, posterior_fraction=.5,
                anterior_emphasis=.5, posterior_emphasis=.5, anterior_mass=.5,
                endpoint_weights=[.5, .5], band_share=.5, gradient_ratio=.05)


def entry(bases, coefficients, background, phase, tag, spec, changes=None):
    base = deepcopy(bases[background])
    e = deepcopy(base)
    e.update(id=f'ap82_s{phase}_{background}_{tag}', family=f'ap82_stage{phase}',
             objective_version=VERSION, compare=base['id'], background=background,
             base_objective=base, base_coefficients=deepcopy(coefficients[background]),
             ap82=deepcopy(spec), ap82_changes=changes or {}, ap82_phase=phase,
             idea='Training-label A/P bands and independent endpoints; fixed background objective.')
    return e


def stage1(bases, coefficients):
    return [entry(bases, coefficients, bg, 1, f'b{round(share*100)}_g{round(ratio*1000):03}',
                  dict(default_spec(), band_share=share, gradient_ratio=ratio))
            for ratio, share, bg in product((.05, .025, .1), (.5, .25, .75, 0., 1.), BACKGROUNDS)]


def score_key(row):
    return (-float(row['macro_dice']), float(row.get('hd95_mm', float('inf'))), row['objective'])


def select_parent(entries, rows):
    table = {r['objective']: r for r in rows}
    eligible = [e for e in entries if e['background'] == 'winner' and 0 < e['ap82']['band_share'] < 1]
    assert all(e['id'] in table for e in entries), 'Stage one must be complete'
    return min(eligible, key=lambda e: score_key(table[e['id']]))


def stage2(bases, coefficients, parent):
    spec = parent['ap82']
    changes = []
    for a, p in product((.25, .5, .75), repeat=2):
        if (a, p) != (.5, .5):
            changes.append(('selection', f'q{round(a*100)}_{round(p*100)}',
                            dict(anterior_fraction=a, posterior_fraction=p)))
    for a, p in product((.5, .75, .9), repeat=2):
        if (a, p) != (.5, .5):
            changes.append(('emphasis', f'e{round(a*100)}_{round(p*100)}',
                            dict(anterior_emphasis=a, posterior_emphasis=p)))
    for a in (.4, .6):
        changes.append(('mass', f'm{round(a*100)}', dict(anterior_mass=a)))
    for near in (.25, .75):
        changes.append(('distance', f'd{round(near*100)}', dict(endpoint_weights=[near, 1-near])))
    for width in (1., 2.):
        changes.append(('width', f'w{int(width)}', dict(width=width)))
    return [entry(bases, coefficients, bg, 2, tag, dict(spec, **change), {factor: change})
            for factor, tag, change in changes for bg in BACKGROUNDS]


def stage3(bases, coefficients, parent, second, rows):
    """At most 16 mirrored interactions; each block must improve by 0.05 pp."""
    table = {r['objective']: r for r in rows}
    assert all(e['id'] in table for e in second)
    baseline = float(table[parent['id']]['macro_dice'])
    selected = []
    for factor in ('selection', 'emphasis', 'mass', 'distance', 'width'):
        candidates = [e for e in second if e['background'] == 'winner' and factor in e['ap82_changes']]
        best = min(candidates, key=lambda e: score_key(table[e['id']]))
        delta = float(table[best['id']]['macro_dice'])-baseline
        if delta >= .0005:
            selected.append((factor, best['ap82_changes'][factor], delta, best['id']))
    options = []
    for size in range(2, len(selected)+1):
        for group in combinations(selected, size):
            options.append(group)
    # Pairwise interactions first; summed marginal gains only prioritize, not predict.
    options.sort(key=lambda group: (len(group), -sum(x[2] for x in group), tuple(x[0] for x in group)))
    result = []
    for n, group in enumerate(options[:16]):
        changes = {x[0]: x[1] for x in group}
        spec = dict(parent['ap82'])
        for change in changes.values(): spec.update(change)
        for bg in BACKGROUNDS:
            result.append(entry(bases, coefficients, bg, 3, f'combo{n+1:02}', spec, changes))
    return result, [dict(factor=f, changes=c, delta=d, source=n) for f, c, d, n in selected]


def scientific_key(e):
    return json.dumps(dict(background=e['background'], spec=e['ap82']), sort_keys=True)
