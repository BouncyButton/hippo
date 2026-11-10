"""Frozen broad follow-up: shape, spatial selection, interactions and schedules."""
from copy import deepcopy
from itertools import product
import json
from scripts.band_evolution.matrix import pair


def focus(**kwargs):
    return dict(kind='regional',inner=3,outer=3,penalty='brier',inside_weight=.5,
                hard_fraction=.25,hard_mix=.5,**kwargs)


def experiment_matrix():
    runs=[];signatures=set()
    def add(name,family,*,band=None,pairs='default',surface=None,ap=None,band_budget=1.,
            extra_budget=1.,shares=None,schedule=None,training=None,idea=''):
        e=dict(id=name,family=family,band=deepcopy(band if band is not None else focus()),
               pair=pair() if pairs=='default' else deepcopy(pairs),surface=surface,ap=ap,ap_ce=None,
               band_budget=band_budget,extra_budget=extra_budget,extra_shares=shares,
               focus_schedule=schedule,training=training,compare='focus25',idea=idea,
               fraction='5pct',seed=0,arm='ordinary_swinunetr')
        scientific={k:v for k,v in e.items() if k not in ('id','family','compare','idea')}
        key=json.dumps(scientific,sort_keys=True)
        if key not in signatures:
            signatures.add(key);runs.append(e)
    def changed(**kw):
        return dict(focus(),**kw)
    add('dice','control',band=dict(kind='none'),pairs=None,idea='Fresh Dice-only control from the common prefix.')
    add('bands','control',band=dict(kind='canonical'),pairs=None,idea='Exact Dice + original bands reference.')
    add('independent13','control',band=dict(kind='canonical'),pairs=dict(pair(),support='original'),idea='Exact previous 78.13% reference.')
    add('focus25','control',idea='Exact winning 78.94% formulation.')
    add('strong_inside','control',band=dict(kind='canonical'),pairs=pair(inside_weight=.6),extra_budget=1.5,
        idea='Exact 78.74% stronger-pair reference.')
    for q,alpha in product((.1,.15,.2,.25,.35,.5),(.25,.5,.75)):
        add(f'hard_q{int(q*100)}_mix{int(alpha*100)}','hardness',band=changed(hard_fraction=q,hard_mix=alpha),
            idea='Separate the selected fraction from its loss mass.')
    for (inner,outer),q in product(((1,1),(2,2),(4,4),(5,5),(1,3),(3,1),(2,3),(3,2)),(.15,.25,.4)):
        add(f'width_i{inner}_o{outer}_q{int(q*100)}','width',band=changed(inner=inner,outer=outer,hard_fraction=q),
            idea='Physical support and concentration interaction, including asymmetric widths.')
    for qi,qo in product((.15,.25,.4,.6),repeat=2):
        if qi==qo==.25:continue
        add(f'side_qi{int(qi*100)}_qo{int(qo*100)}','side_focus',band=changed(inside_fraction=qi,outside_fraction=qo),
            idea='Different error selection fractions for missing and extra tissue, equal side mass.')
    for inside in (.4,.45,.55,.6):
        add(f'side_mass{int(inside*100)}','side_focus',band=changed(inside_weight=inside),
            idea='Side-mass sensitivity around the winning formulation.')
    for band_budget,extra_budget in product((.75,1.,1.25,1.5),(.5,1.,1.25,1.5,2.)):
        tag=lambda x:str(x).replace('.','p')
        add(f'budget_band{tag(band_budget)}_pair{tag(extra_budget)}','strength',band_budget=band_budget,
            extra_budget=extra_budget,idea='Independent calibrated band/pair strength grid; retain p95 safety cap.')
    for strength in (.5,1.,1.5,2.):
        add(f'band_only_strength{str(strength).replace(".","p")}','strength',pairs=None,band_budget=strength,
            idea='Does hard-band supervision need the endpoint pairs?')
    profiles=[('inside60',pair(inside_weight=.6)),('inside40',pair(inside_weight=.4)),
              ('inside_far',pair(inside_weights=[.25,.75],outside_weights=[.5,.5])),
              ('near60_30_10',pair((1,3,5),[.6,.3,.1])),('far75',pair(weights=[.25,.75])),
              ('separate',dict(pair(),support='separate'))]
    for (tag,spec),strength in product(profiles,(1.,1.5)):
        add(f'pair_{tag}_strength{str(strength).replace(".","p")}','pair_interaction',pairs=spec,extra_budget=strength,
            idea='Combine hard bands with supported distance/side/eligibility hypotheses.')
    penalties=[('bce',dict(penalty='bce')),('hybrid25',dict(penalty='hybrid',bce_share=.25)),
               ('hybrid50',dict(penalty='hybrid',bce_share=.5)),('power15',dict(power=1.5)),
               ('power3',dict(power=3.)),('huber',dict(penalty='huber',delta=.25))]
    for (tag,kw),q in product(penalties,(.15,.25)):
        add(f'penalty_{tag}_q{int(q*100)}','penalty',band=changed(hard_fraction=q,**kw),
            idea='Change the confidence-error response within hard regional bands.')
    for decay,q in product((.75,1.,2.),(.15,.25)):
        add(f'decay{str(decay).replace(".","p")}_q{int(q*100)}','distance_focus',
            band=changed(decay_mm=decay,hard_fraction=q),idea='Combine distance emphasis with hard-error selection.')
    for shares in ([.5,.35,.15],[1/3]*3,[.2,.5,.3]):
        tag='_'.join(str(round(x*100)) for x in shares)
        add('shell123_'+tag,'distance_focus',band=changed(shells=[1,2,3],shell_weights=shares),
            idea='Select hard voxels independently within each physical shell.')
    for kernel,q,mix in product((3,5,7),(.15,.25,.5),(0.,.5)):
        add(f'neighbor_k{kernel}_q{int(q*100)}_point{int(mix*100)}','neighborhood',
            band=changed(neighborhood=[kernel],point_rank_share=mix,hard_fraction=q),
            idea='Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors.')
    for kernels in ([3,5],[3,7],[3,5,7]):
        add('neighbor_multi'+''.join(map(str,kernels)),'neighborhood',
            band=changed(neighborhood=kernels,point_rank_share=.5),idea='Combine local error context at multiple scales.')
    for tile,q in product((4,8,16),(.15,.25,.5)):
        add(f'tile{tile}_q{int(q*100)}','spatial_coverage',band=changed(tile=tile,hard_fraction=q),
            idea='Keep a difficult subset in each spatial tile; weight tiles by their valid band mass.')
    for power,alpha in product((.5,1.,2.),(.25,.5,.75)):
        add(f'smooth_power{str(power).replace(".","p")}_mix{int(alpha*100)}','smooth_focus',
            band=changed(smooth_power=power,hard_mix=alpha),idea='Continuous detached difficulty weights, capped at 20, instead of a hard cutoff.')
    for kernel in (3,5):
        add(f'smooth_neighbor{kernel}','smooth_focus',band=changed(smooth_power=1.,neighborhood=[kernel]),
            idea='Smooth neighborhood weights preserve weak supervision everywhere.')
    # Keep nested configuration values JSON-native so the frozen protocol can
    # be compared directly with the reconstructed matrix on the worker.
    for scales in ([[1,.5],[3,.5]],[[2,.5],[5,.5]],[[1,.2],[3,.6],[5,.2]]):
        for strength in (1.,1.5):
            add('multi'+''.join(str(x[0]) for x in scales)+'_strength'+str(strength).replace('.','p'),
                'multiscale',band=changed(multiscale=scales),extra_budget=strength,
                idea='Average nested hard-band losses with one calibrated band budget.')
    for width,penalty,share in product((1,3,5),('bce','brier'),(.1,.25)):
        add(f'ap_{penalty}_w{width}_share{int(share*100)}','ap_focus',
            ap=dict(kind='focused',width_mm=width,penalty=penalty,hard_fraction=.25,hard_mix=.5),
            shares=dict(pair=1-share,ap=share),idea='Actual A/P-label conditional errors near the interface; shares the existing extra budget.')
    for tag,surface in [('sdf8',dict(kind='sdf',clip_mm=8)),
                        ('chamfer',dict(kind='chamfer',penalty='linear',predicted_weight=.5))]:
        for share in (.1,.25):
            add(f'surface_{tag}_share{int(100*share)}','surface_interaction',surface=surface,
                shares=dict(pair=1-share,surface=share),idea='Low-share surface feedback with the new hard-band winner.')
    for tag,start in [('wide',dict(hard_fraction=.75)),('gentle',dict(hard_mix=0.)),
                      ('wide_gentle',dict(hard_fraction=.75,hard_mix=0.))]:
        for end in (20,35):
            add(f'curriculum_{tag}_e{end}','curriculum',schedule=dict(start=start,end_epoch=end),
                idea='Gradually concentrate feedback; calibrate the final objective once at the common prefix.')
    for tag,band,pairs in [('dice',dict(kind='none'),None),('bands',dict(kind='canonical'),None),
                           ('focus25',focus(),pair())]:
        add('long_'+tag,'training_schedule',band=band,pairs=pairs,
            training=dict(max_epochs=150,step_size=40,early_stopping=dict(min_epochs=75,patience=16,min_delta=.0005)),
            idea='Separately labelled longer schedule, with matched Dice and bands controls; no architecture change.')
    priority=['dice','bands','independent13','focus25','strong_inside','budget_band1p0_pair1p5',
              'band_only_strength1p0','neighbor_k3_q25_point50','tile8_q25','smooth_power1p0_mix50',
              'ap_bce_w3_share10','long_dice','long_bands','long_focus25']
    assert len(runs)==len({x['id'] for x in runs}) and len(runs)<250
    return [next(x for x in runs if x['id']==name) for name in priority]+[x for x in runs if x['id'] not in priority]
