"""Finite combinations around the observed 3 mm inside / 2 mm outside winner."""
from copy import deepcopy
from itertools import product
import json

ANCHOR = 'width_i3_o2_q25'


def scientific_key(entry):
    """Normalize side-fraction aliases so equivalent losses are not rerun."""
    value = {k: deepcopy(v) for k, v in entry.items()
             if k not in ('id', 'family', 'compare', 'idea')}
    band = value['band']
    if band.get('kind') == 'regional':
        fraction = band.pop('hard_fraction', .25)
        band.setdefault('inside_fraction', fraction)
        band.setdefault('outside_fraction', fraction)
    # JSON treats 1 and 1.0 differently; they define the same configuration here.
    def normalize(x):
        if isinstance(x, dict):
            return {k: normalize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [normalize(v) for v in x]
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
        return x
    return json.dumps(normalize(value), sort_keys=True)


def added_matrix(original):
    base = next(e for e in original if e['id'] == ANCHOR)
    seen = {scientific_key(e) for e in original}
    runs = []
    def add(name, family, *, band=None, band_budget=1., extra_budget=1., pairs=True, idea):
        entry = deepcopy(base)
        entry.update(id='a32_'+name, family='a32_'+family, compare=ANCHOR, idea=idea,
                     band_budget=band_budget, extra_budget=extra_budget)
        entry['band'].update(band or {})
        if not pairs:
            entry['pair'] = None
        key = scientific_key(entry)
        if key not in seen:
            seen.add(key)
            runs.append(entry)
    def pct(value):
        return str(round(value*100))
    def number(value):
        return str(value).replace('.', 'p')

    for q, mix in product((.1,.15,.2,.25,.35,.4,.5), (.25,.5,.75)):
        add(f'hard_q{pct(q)}_mix{pct(mix)}', 'hardness',
            band=dict(hard_fraction=q, hard_mix=mix),
            idea='Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms.')
    for qi, qo, mix in product((.15,.25,.4,.6), (.15,.25,.4,.6), (.5,.75)):
        add(f'side_qi{pct(qi)}_qo{pct(qo)}_mix{pct(mix)}', 'side_fraction',
            band=dict(inside_fraction=qi, outside_fraction=qo, hard_mix=mix),
            idea='Cross independent inside/outside hard fractions with moderate and stronger focus.')
    profiles = ((.25,.5), (.35,.75), (.4,.5))
    for inside, (q, mix) in product((.4,.45,.55,.6), profiles):
        add(f'mass{pct(inside)}_q{pct(q)}_mix{pct(mix)}', 'side_mass',
            band=dict(inside_weight=inside, hard_fraction=q, hard_mix=mix),
            idea='Test the missed/extra-tissue balance under three supported focus settings.')
    for kernel, q, mix, point in product((3,5,7), (.15,.25,.35,.4,.5), (.5,.75), (0.,.5)):
        add(f'neighbor_k{kernel}_q{pct(q)}_mix{pct(mix)}_point{pct(point)}', 'neighborhood',
            band=dict(neighborhood=[kernel], hard_fraction=q, hard_mix=mix, point_rank_share=point),
            idea='Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking.')
    for kernels, q, mix in product(([3,5],[3,7],[3,5,7]), (.25,.35,.4), (.5,.75)):
        tag = ''.join(map(str, kernels))
        add(f'neighbor_multi{tag}_q{pct(q)}_mix{pct(mix)}', 'multineighborhood',
            band=dict(neighborhood=kernels, hard_fraction=q, hard_mix=mix, point_rank_share=.5),
            idea='Combine local error context at multiple sizes on the asymmetric support.')
    for kernel, (qi,qo), mix in product((3,5), ((.15,.25),(.25,.15),(.25,.4),(.4,.25)), (.5,.75)):
        add(f'neighbor_k{kernel}_qi{pct(qi)}_qo{pct(qo)}_mix{pct(mix)}', 'neighborhood_side',
            band=dict(neighborhood=[kernel], point_rank_share=.5, inside_fraction=qi,
                      outside_fraction=qo, hard_mix=mix),
            idea='Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus.')
    for kernel, inside, (q,mix) in product((3,5), (.45,.55), profiles[:2]):
        add(f'neighbor_k{kernel}_mass{pct(inside)}_q{pct(q)}_mix{pct(mix)}', 'neighborhood_mass',
            band=dict(neighborhood=[kernel], point_rank_share=.5, inside_weight=inside,
                      hard_fraction=q, hard_mix=mix),
            idea='Test whether spatial selection and a modest side-mass adjustment complement each other.')
    for band_budget, pair_budget in product((.75,1.,1.25), (.75,1.,1.25,1.5)):
        add(f'budget_b{number(band_budget)}_p{number(pair_budget)}', 'strength',
            band_budget=band_budget, extra_budget=pair_budget,
            idea='Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm.')
    for budget in (.5,1.,1.5):
        add(f'band_only_strength{number(budget)}', 'pair_ablation', band_budget=budget, pairs=False,
            idea='Matched ablation: test whether endpoint pairs remain necessary with asymmetric focused bands.')

    # All additions precede unfinished originals; front-load the clearest interactions.
    priority = ['a32_hard_q35_mix75', 'a32_neighbor_k3_q25_mix50_point50',
                'a32_side_qi25_qo40_mix50', 'a32_mass45_q25_mix50',
                'a32_hard_q40_mix75', 'a32_neighbor_k3_q35_mix75_point50',
                'a32_neighbor_k5_q25_mix50_point50', 'a32_neighbor_k3_qi25_qo40_mix50',
                'a32_mass55_q25_mix50', 'a32_budget_b1p0_p1p5']
    lookup = {e['id']: e for e in runs}
    assert len(lookup) == len(runs) and set(priority) <= set(lookup)
    return [lookup[n] for n in priority] + [e for e in runs if e['id'] not in priority]


def extend_protocol(standard):
    p = deepcopy(standard)
    added = added_matrix(p['runs'])
    p['runs'] = added + p['runs']
    p['main_runs'] = len(p['runs'])
    assert all(not e.get('training') for e in p['runs'])
    p['loss_definitions']['schedule'] = f'All {p["main_runs"]} arms use max75, StepLR20 gamma0.5 and the original early stopping.'
    p['interpretation'] = ('User-authorized asymmetric-band extension after inspecting fold0/seed0 validation: '
        f'{len(added)} new variants run first, followed by the unfinished original standard75 variants. '
        'New runs use 3 mm inside / 2 mm outside bands. Existing losses, shared epoch-5 initialization, '
        'training-only calibration procedure and all original run identities remain unchanged. '
        'This is adaptive exploration on reused validation, requiring independent seed/fold confirmation.')
    return p
