"""Finite preregistered screen; no outcome-dependent expansion or pruning."""
from copy import deepcopy
from itertools import product


def pair(distances=(1, 3), weights=None, **kw):
    return dict(distances=list(distances), weights=weights or [1/len(distances)]*len(distances),
                support='all', form='independent', **kw)


def experiment_matrix():
    runs, signatures = [], set()
    def add(name, family, *, band=None, pairs=None, surface=None, compare='independent13', idea='', budget=1.):
        import json
        e = dict(id=name, family=family, band=band or dict(kind='canonical'), pair=pairs,
                 surface=surface, ap=None, ap_ce=None, extra_shares=None, compare=compare,
                 idea=idea, fraction='5pct', seed=0, arm='ordinary_swinunetr', extra_budget=budget)
        scientific={k:deepcopy(e[k]) for k in ('band','pair','surface','extra_budget')}
        if pairs and pairs['form']=='independent':
            spec=scientific['pair']
            if spec['support']=='original':spec['support']='all'
            spec.setdefault('inside_weight',.5)
            iw=spec.pop('inside_weights',spec['weights'])
            ow=spec.pop('outside_weights',spec['weights'])
            steps=spec.pop('endpoint_steps',[[(int(d)-1)//2]*2 for d in spec['distances']])
            spec['samples']=[(s,a,b) for s,a,b in zip(steps,iw,ow) if a>0 or b>0]
            spec.pop('weights');spec.pop('distances')
        signature = json.dumps(scientific, sort_keys=True)
        if signature in signatures:
            return
        signatures.add(signature); runs.append(e)
    legacy = dict(support='original', distances=[1,3], weights=[.5,.5], form='coupled')
    winner = dict(legacy, form='independent')
    add('bands', 'control', compare='bands', idea='Exact canonical Dice + bands.')
    add('pair13', 'control', pairs=legacy, compare='bands', idea='Exact previous coupled comparison.')
    add('independent13', 'control', pairs=winner, idea='Exact 78.13% independent-target reference.')
    # Full distance-mixture x side-balance grid. Weight means loss mass, not voxel count.
    profiles = [('d1',(1,),[1.]),('d3',(3,),[1.]),('d5',(5,),[1.]),('d7',(7,),[1.]),
                ('d13_75_25',(1,3),[.75,.25]),('d13_equal',(1,3),[.5,.5]),
                ('d13_25_75',(1,3),[.25,.75]),('d15',(1,5),[.5,.5]),('d35',(3,5),[.5,.5]),
                ('d135_equal',(1,3,5),[1/3]*3),('d135_mid',(1,3,5),[.2,.6,.2]),
                ('d135_near',(1,3,5),[.6,.3,.1]),('d1357',(1,3,5,7),[.4,.3,.2,.1])]
    for (tag, ds, ws), inside in product(profiles, (.4,.5,.6,.7)):
        add(f'{tag}_in{int(100*inside)}', 'distance_side', pairs=pair(ds, ws, inside_weight=inside),
            idea='Factorial distance mixture and inside/outside balance; per-scale eligible support.')
    # Independent scale profiles on either side, crossed with two side balances.
    for a,b,inside in product((.25,.5,.75,1.), (.25,.5,.75,1.), (.5,.6)):
        add(f'sides_i{int(a*100)}_o{int(b*100)}_in{int(inside*100)}','side_scales',
            pairs=pair(inside_weights=[a,1-a], outside_weights=[b,1-b],inside_weight=inside),
            idea='Let missing tissue and extra tissue have different scale preferences.')
    # Strict matched-face controls plus one-sided eligibility for thin structures.
    for support, ds, ws in [('common3',(1,),[1.]),('common3',(3,),[1.]),
                            ('common3',(1,3),[.5,.5]),('common3',(1,3),[.25,.75]),
                            ('common5',(1,3),[.5,.5]),('common5',(1,3,5),[.2,.6,.2]),
                            ('common7',(1,3,5),[.2,.6,.2]),('common7',(1,3,5,7),[.4,.3,.2,.1]),
                            ('separate',(1,3),[.5,.5]),('separate',(1,3,5),[.2,.6,.2]),
                            ('separate',(1,3,5,7),[.4,.3,.2,.1])]:
        spec=pair(ds,ws);spec['support']=support
        tag=''.join(map(str,ds))+'_'+str(int(ws[0]*100))
        add(f'support_{support}_{tag}','support',pairs=spec,
            idea='Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin.')
    # Endpoints indexed as voxel steps from the immediate inside/outside centers.
    for steps in ((0,1),(1,0),(0,2),(2,0),(1,2),(2,1)):
        for inside in (.5,.6):
            spec=pair((1,3),[.5,.5],inside_weight=inside, endpoint_steps=[[0,0],list(steps)])
            add(f'offset_i{steps[0]}_o{steps[1]}_in{int(inside*100)}','offset',pairs=spec,
                idea='Shift far-scale supervision toward one side; retain adjacent-voxel reference.')
    for reduction, inside in product(('unique_voxel','anchor','class_balanced'),(.5,.6)):
        add(f'reduce_{reduction}_in{int(inside*100)}','reduction',pairs=pair(inside_weight=inside,reduction=reduction),
            idea='Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass.')
    # Loss shape x side balance: robust, hard-example and confidence-relaxed targets.
    for tag, settings in [('bce',dict(penalty='bce')),('power15',dict(power=1.5)),
                          ('power3',dict(power=3.)),('huber',dict(penalty='huber',delta=.25)),
                          ('tol05',dict(tolerance=.05)),('tol10',dict(tolerance=.1)),
                          ('top25',dict(hard_fraction=.25)),('top10',dict(hard_fraction=.1))]:
        for inside in (.5,.6):
            add(f'penalty_{tag}_in{int(inside*100)}','penalty',pairs=pair(inside_weight=inside,**settings),
                idea='Alter confidence-error emphasis without changing sampled anatomy.')
    # Regional bands as a replacement AND as context for independent endpoints.
    bands=[('brier1',dict(kind='regional',inner=1,outer=1,penalty='brier')),
           ('brier2',dict(kind='regional',inner=2,outer=2,penalty='brier')),
           ('brier3',dict(kind='regional',inner=3,outer=3,penalty='brier')),
           ('bce1',dict(kind='regional',inner=1,outer=1,penalty='bce')),
           ('bce2',dict(kind='regional',inner=2,outer=2,penalty='bce')),
           ('bce3',dict(kind='regional',inner=3,outer=3,penalty='bce')),
           ('shell123',dict(kind='regional',inner=3,outer=3,penalty='brier',shells=[1,2,3],shell_weights=[.5,.35,.15])),
           ('shell135',dict(kind='regional',inner=5,outer=5,penalty='brier',shells=[1,3,5],shell_weights=[.5,.35,.15])),
           ('inner1outer2',dict(kind='regional',inner=1,outer=2,penalty='brier')),
           ('inner2outer1',dict(kind='regional',inner=2,outer=1,penalty='brier'))]
    for (tag,base),(ptag,spec),inside in product(bands,[('alone',None),('pairs',pair())],(.5,.6)):
        add(f'band_{tag}_{ptag}_in{int(inside*100)}','regional_band',band=dict(base,inside_weight=inside),pairs=spec,
            idea='Test regional Brier/BCE and distance-shell weighting with and without independent endpoints.')
    for tag,settings in [('top25',dict(hard_fraction=.25)),('class',dict(class_balanced=True)),
                         ('decay1',dict(decay_mm=1.)),('decay2',dict(decay_mm=2.))]:
        for inside in (.5,.6):
            add(f'band_focus_{tag}_in{int(inside*100)}','regional_band',
                band=dict(kind='regional',inner=3,outer=3,penalty='brier',inside_weight=inside,**settings),pairs=pair(),
                idea='Focus regional bands on harder errors, balanced class surfaces, or nearby voxels.')
    for tag,surface in [('sdf3',dict(kind='sdf',clip_mm=3)),('sdf8',dict(kind='sdf',clip_mm=8)),
                        ('brier_distance',dict(kind='distance_brier',clip_mm=3,power=1)),
                        ('chamfer',dict(kind='chamfer',penalty='linear',predicted_weight=.5)),
                        ('mass3',dict(kind='local_mass',kernel=3)),('mass5',dict(kind='local_mass',kernel=5))]:
        for inside in (.5,.6):
            add(f'combo_{tag}_in{int(inside*100)}','combination',pairs=pair(inside_weight=inside),surface=surface,
                idea='Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget.')
    for tag,spec in [('equal',pair()),('far',pair(weights=[.25,.75])),('protect_inside',pair(inside_weight=.6))]:
        for budget in (.5,1.5):
            add(f'budget_{tag}_{str(budget).replace(".","p")}','budget',pairs=spec,budget=budget,
                idea='Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5.')
    # Early coverage of the most informative independent hypotheses.
    priority=['bands','pair13','independent13','d1_in50','d3_in50','d13_25_75_in50',
              'd13_equal_in60','sides_i75_o50_in50','d135_mid_in50','support_separate_13_50',
              'combo_sdf8_in50','band_brier2_pairs_in50']
    assert len(runs)==len({r['id'] for r in runs}) and len(runs) < 240
    return [next(r for r in runs if r['id']==n) for n in priority]+[r for r in runs if r['id'] not in priority]
