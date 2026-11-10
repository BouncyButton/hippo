"""Finite, deduplicated interactions around the observed 80.725% configuration."""
from copy import deepcopy
from itertools import product
import json
from scripts.band_evolution.matrix import pair

ANCHOR = 'a32_side_qi60_qo15_mix75'


def scientific_key(entry):
    value = {k: deepcopy(v) for k, v in entry.items()
             if k not in ('id', 'family', 'compare', 'idea', 'objective_version')}
    band = value['band']
    if band.get('kind') == 'regional':
        for side in ('inside', 'outside'):
            for key, default in (('fraction', band.get('hard_fraction', .25)),
                                 ('hard_mix', band.get('hard_mix', .5)),
                                 ('penalty', band.get('penalty', 'brier')),
                                 ('bce_share', band.get('bce_share', .5)),
                                 ('neighborhood', band.get('neighborhood')),
                                 ('point_rank_share', band.get('point_rank_share', 0.))):
                band.setdefault(side+'_'+key, default)
        for key in ('hard_fraction', 'hard_mix', 'penalty', 'bce_share', 'neighborhood', 'point_rank_share'):
            band.pop(key, None)
        for side in ('inside', 'outside'):
            if band[side+'_penalty'] != 'hybrid':
                band.pop(side+'_bce_share')
            if not band[side+'_neighborhood']:
                band.pop(side+'_point_rank_share')
    def normalize(x):
        if isinstance(x, dict): return {k: normalize(v) for k, v in x.items()}
        if isinstance(x, list): return [normalize(v) for v in x]
        if isinstance(x, (int, float)) and not isinstance(x, bool): return float(x)
        return x
    return json.dumps(normalize(value), sort_keys=True)


def retained_ids(original):
    """Retain interpretable confidence, endpoint and depth-shell comparisons."""
    return [e['id'] for e in original if
            (e['family'] == 'penalty' and e['band']['penalty'] in ('bce', 'hybrid')) or
            e['family'] == 'pair_interaction' or
            (e['family'] == 'distance_focus' and e['band'].get('shells'))]


def added_matrix(previous):
    base = next(e for e in previous if e['id'] == ANCHOR)
    seen = {scientific_key(e) for e in previous}
    entries = []
    def tag(value): return str(round(value*100))
    def profile(qi=.6, qo=.15, mi=.75, mo=.75):
        return dict(inside_fraction=qi, outside_fraction=qo, inside_hard_mix=mi, outside_hard_mix=mo)
    def ptag(p): return '_'.join(k+tag(p[v]) for k,v in
        (('qi','inside_fraction'),('qo','outside_fraction'),('mi','inside_hard_mix'),('mo','outside_hard_mix')))
    def add(name, family, *, band=None, pairs=None, band_budget=1., pair_budget=1., idea):
        e = deepcopy(base)
        e.update(id='w81_'+name, family='w81_'+family, objective_version='winner_v1',
                 compare=ANCHOR, idea=idea, band_budget=band_budget, extra_budget=pair_budget)
        e['band'].update(band or {})
        if pairs is not None: e['pair'] = deepcopy(pairs)
        key = scientific_key(e)
        if key not in seen:
            seen.add(key); entries.append(e)

    for qi,qo in product((.5,.6,.7,.8),(.05,.1,.15,.2)):
        b=profile(qi,qo)
        add('fractions_'+ptag(b),'fractions',band=b,
            idea='Refine and extend the broad-inside/narrow-outside selection trend at fixed emphasis.')
    selection_profiles=((.5,.1),(.6,.1),(.6,.15),(.6,.2),(.7,.1),(.7,.15),(.8,.1),(.8,.15))
    for (qi,qo),mi,mo in product(selection_profiles,(.5,.65,.75,.85),(.65,.75,.85,.9)):
        b=profile(qi,qo,mi,mo)
        add('emphasis_'+ptag(b),'side_emphasis',band=b,
            idea='Separate inside/outside emphasis and cross it with supported nearby selection fractions.')
    profiles=[profile(),profile(.6,.1),profile(.7,.1,.65,.85),profile(.8,.15)]
    for b,bb,pb in product((profiles[0],profiles[2],profiles[3]),(.75,1.,1.25),(.75,1.,1.25,1.5)):
        add('strength_'+ptag(b)+f'_b{tag(bb)}_p{tag(pb)}','strength',band=b,band_budget=bb,pair_budget=pb,
            idea='Cross calibrated band and endpoint strengths on the winner and two explicit nearby hypotheses.')
    for b,beta,mass,pb in product(profiles,(.1,.25),(.45,.5,.55),(1.,1.5)):
        add('insidece_'+ptag(b)+f'_ce{tag(beta)}_mass{tag(mass)}_p{tag(pb)}','inside_hybrid',
            band=dict(b,inside_penalty='hybrid',inside_bce_share=beta,inside_weight=mass),pair_budget=pb,
            idea='Strengthen confidently missed inside voxels while preserving outside Brier feedback; cross modest side weights and endpoint strength.')
    for b,mass,pb in product(profiles,(.4,.45,.5,.55,.6),(1.,1.5)):
        add('mass_'+ptag(b)+f'_mass{tag(mass)}_p{tag(pb)}','side_mass',
            band=dict(b,inside_weight=mass),pair_budget=pb,
            idea='Separate total inside/outside loss mass from selected fractions and emphasis.')
    endpoint_profiles=[('inside60',pair(inside_weight=.6)),('inside40',pair(inside_weight=.4)),
        ('inside_far',pair(inside_weights=[.25,.75],outside_weights=[.5,.5])),
        ('near60_30_10',pair((1,3,5),[.6,.3,.1])),('far75',pair(weights=[.25,.75])),
        ('separate',dict(pair(),support='separate'))]
    for (name,spec),pb in product(endpoint_profiles,(1.,1.5)):
        add('endpoint_'+name+'_p'+tag(pb),'endpoint',pairs=spec,pair_budget=pb,
            idea='Transfer a retained distance/side/eligibility comparison directly onto the winning band configuration.')
    for b,shares,pb in product((profiles[0],profiles[2]),([.5,.35,.15],[1/3]*3,[.2,.5,.3]),(1.,1.5)):
        add('shell_'+ptag(b)+'_'+''.join(tag(x) for x in shares)+'_p'+tag(pb),'shell',
            band=dict(b,shells=[1,2,3],shell_weights=shares),pair_budget=pb,
            idea='Preserve hard-example coverage within each physical depth shell; renormalize over supported shells per side.')
    for b,(inner,outer) in product((profiles[0],profiles[2]),((2,2),(3,1),(3,3),(4,2))):
        add('width_'+ptag(b)+f'_i{inner}_o{outer}','width',band=dict(b,inner=inner,outer=outer),
            idea='Check immediate physical-support neighbors after fixing the discovered selection asymmetry.')
    for side,point in product(('inside','outside','both'),(.5,.75)):
        b={}
        for name in (('inside','outside') if side=='both' else (side,)):
            b.update({name+'_neighborhood':[3],name+'_point_rank_share':point})
        add('local_'+side+'_point'+tag(point),'local_context',band=b,
            idea='Small matched diagnostic: add local ranking on one or both sides of the winner without changing its fractions.')

    # First compare single changes, then the strongest mechanistic interactions.
    priority=[
        'w81_fractions_qi60_qo10_mi75_mo75',
        'w81_fractions_qi70_qo15_mi75_mo75',
        'w81_fractions_qi70_qo10_mi75_mo75',
        'w81_strength_qi60_qo15_mi75_mo75_b100_p150',
        'w81_strength_qi60_qo15_mi75_mo75_b100_p125',
        'w81_emphasis_qi60_qo15_mi65_mo75',
        'w81_emphasis_qi60_qo15_mi75_mo85',
        'w81_emphasis_qi60_qo15_mi65_mo85',
        'w81_insidece_qi60_qo15_mi75_mo75_ce10_mass50_p100',
        'w81_insidece_qi60_qo15_mi75_mo75_ce25_mass50_p100',
        'w81_fractions_qi80_qo15_mi75_mo75',
        'w81_emphasis_qi70_qo10_mi65_mo85',
        'w81_insidece_qi60_qo15_mi75_mo75_ce10_mass50_p150',
        'w81_mass_qi60_qo15_mi75_mo75_mass45_p100',
        'w81_mass_qi60_qo15_mi75_mo75_mass55_p100',
        'w81_fractions_qi60_qo5_mi75_mo75',
    ]
    lookup={e['id']:e for e in entries}
    assert set(priority)<=set(lookup)
    assert len(entries)<400
    # Finish core fractions before full interaction grids; test confidence changes early.
    family_order=('fractions','inside_hybrid','strength','side_mass','side_emphasis','endpoint','shell','width','local_context')
    return [lookup[n] for n in priority]+[e for f in family_order for e in entries
        if e['family']=='w81_'+f and e['id'] not in priority]


def extend_protocol(previous, completed_ids):
    p=deepcopy(previous)
    old={e['id']:e for e in p['runs']}
    retained=retained_ids(p['runs'])
    new=added_matrix(p['runs'])
    unfinished_asym=[e for e in p['runs'] if e['id'].startswith('a32_') and e['id'] not in completed_ids]
    # Preserve every completed result and finish the existing 3/2 block first.
    preserved=[e for e in p['runs'] if e['id'] in completed_ids]
    kept=[old[n] for n in retained if n not in completed_ids]
    p['runs']=preserved+unfinished_asym+new[:16]+kept+new[16:]
    p['main_runs']=len(p['runs'])
    assert len({e['id'] for e in p['runs']})==len(p['runs'])
    assert all(not e.get('training') for e in p['runs'])
    p['interpretation']='User-authorized winner refinement: preserve completed evidence, finish pending asymmetric runs, run 16 prioritized winner variants, retain confidence/endpoint/shell comparisons, then finish the finite winner interaction grid. All arms remain ordinary SwinUNETR, 5%, fold0 seed0, max75 with unchanged optimization. Adaptive development on reused validation; independent replication remains necessary.'
    return p
