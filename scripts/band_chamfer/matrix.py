"""Predeclared, finite hypothesis matrix; no validation-driven additions."""


def experiment_matrix():
    runs = []

    def add(name, family, *, band=None, pair=None, surface=None, ap=None, ap_ce=None,
            shares=None, compare=None, idea=''):
        runs.append(dict(id=name, family=family, band=band or dict(kind='canonical'),
                         pair=pair, surface=surface, ap=ap, ap_ce=ap_ce, extra_shares=shares,
                         compare=compare or 'bands', idea=idea,
                         fraction='5pct', seed=0, arm='ordinary_swinunetr'))

    winner = dict(support='original', distances=[1, 3], weights=[.5, .5], form='coupled')
    chamfer = dict(kind='chamfer', penalty='linear', predicted_weight=.5)
    add('bands', 'control', idea='Fresh exact Dice + canonical two-step bands.')
    add('pair13', 'control', pair=winner, idea='Fresh previous 1 + 3 mm probability-pair winner.')
    for width in (1, 2, 3, 5):
        add(f'band_morph{width}_valid', 'band', band=dict(kind='morph', width=width),
            idea='Change morphological support; exclude padding. Width 2 isolates the FOV policy.')
    for width in (1, 2, 3, 5):
        add(f'band_euclid{width}', 'band', band=dict(kind='euclidean', inner=width, outer=width),
            idea='Physical distance to GT surface rather than six-neighbor graph distance.')
    shells = dict(kind='euclidean', inner=5, outer=5, shells=[1, 3, 5])
    add('band_shell135', 'band', band=shells, idea='Equal mass per nonempty 0-1, 1-3, 3-5 mm shell and side.')
    add('band_outer75', 'band', band=dict(kind='euclidean', inner=3, outer=3, outer_weight=.75),
        compare='band_euclid3', idea='More weight on extra tissue while retaining inner supervision.')
    for inside, outside in ((1, 3), (3, 1)):
        add(f'band_inner{inside}_outer{outside}', 'band', band=dict(kind='euclidean', inner=inside, outer=outside),
            compare='band_euclid3', idea='Separate inside versus outside reach, retaining equal side mass.')
    add('band_soft05', 'band', band=dict(kind='euclidean', inner=3, outer=3, soft_mm=.5),
        compare='band_euclid3', idea='Soft sigmoid targets near the annotation; boundary remains at the same position.')

    for name, distances, weights in (('1', [1], [1.]), ('3', [3], [1.]),
                                     ('13_equal', [1, 3], [.5, .5]), ('13_far75', [1, 3], [.25, .75])):
        add('pair_common_'+name, 'pair', pair=dict(support='common3', distances=distances,
             weights=weights, form='coupled'), compare='pair_common_13_equal',
             idea='Same eligible oriented GT faces for distance and mixture comparisons.')
    add('pair_independent13', 'pair', pair=dict(winner, form='independent'), compare='pair13',
        idea='Remove the cross-product coupling; keep identical endpoint samples.')
    for shift in ('symmetric', 'outward', 'inward'):
        add('pair_shift_'+shift, 'pair', pair=dict(winner, support='common_shift', shift=shift),
            compare='pair_shift_symmetric', idea='Same six-voxel-valid faces; move the 3 mm endpoints only.')
    for margin in (.8, .9):
        add('pair_gap'+str(margin).replace('.', ''), 'pair', pair=dict(winner, form='hinge', margin=margin),
            compare='pair13', idea='Stop pushing when the inside/outside probability gap is sufficient.')

    for clip in (3, 8):
        add(f'sdf{clip}', 'surface', surface=dict(kind='sdf', clip_mm=clip),
            idea='Outer-foreground signed-distance probability loss, normalized separately inside and outside.')
    for power in (1, 2):
        add(f'distance_brier{power}', 'surface', surface=dict(kind='distance_brier', clip_mm=8, power=power),
            idea='Distance-weighted squared probability errors; finite gradients without a predicted surface.')
    variants = [('mean', chamfer), ('squared', dict(chamfer, penalty='squared')),
                ('huber1', dict(chamfer, penalty='huber')),
                ('tolerance05', dict(chamfer, tolerance_mm=.5)),
                ('pred75', dict(chamfer, predicted_weight=.75)),
                ('gt75', dict(chamfer, predicted_weight=.25)),
                ('pred_only', dict(chamfer, predicted_weight=1.)),
                ('gt_only', dict(chamfer, predicted_weight=0.)),
                ('tail10', dict(chamfer, tail_fraction=.1))]
    for name, surface in variants:
        add('chamfer_'+name, 'surface', surface=surface, compare='chamfer_mean',
            idea='Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight.')
    for name, surface in [('mean', chamfer), ('squared', dict(chamfer, penalty='squared')),
                          ('tail10', dict(chamfer, tail_fraction=.1)), ('sdf8', dict(kind='sdf', clip_mm=8))]:
        add('pair13_plus_'+name, 'combination', pair=winner, surface=surface, compare='pair13',
            idea='Pair and surface terms share ONE calibrated extra-gradient budget.')
    add('shell135_plus_chamfer', 'combination', band=shells, surface=chamfer, compare='band_shell135',
        idea='Multiscale regional supervision plus actual predicted-surface displacement.')
    add('shell135_plus_pair13', 'combination', band=shells, pair=winner, compare='band_shell135',
        idea='Multiscale bands plus local cross-boundary ordering.')
    add('inner1_outer3_plus_chamfer', 'combination', band=dict(kind='euclidean', inner=1, outer=3),
        surface=chamfer, compare='band_inner1_outer3', idea='Broader outside reach plus surface alignment.')
    ce = dict(kind='full')
    ap_cd = dict(kind='chamfer', penalty='linear', predicted_weight=.5)
    add('ap_ce_full', 'ap_control', ap_ce=ce, idea='Existing foreground-only original-label conditional A/P CE.')
    add('ap_ce_band3', 'ap_control', ap_ce=dict(kind='band', width_mm=3), compare='ap_ce_full',
        idea='Same conditional CE within 3 mm of the actual annotated interface.')
    add('ap_pair13', 'ap_control', ap=dict(kind='pair'), compare='ap_ce_full',
        idea='A/P crossing pairs only; isolate internal ordering from outer-boundary terms.')
    add('ap_chamfer_shape_only', 'ap_chamfer', ap=ap_cd, compare='ap_chamfer_mean_ce',
        idea='Diagnostic shape-only control: deliberately invariant to swapping A/P names.')
    for name, settings in [('mean', {}), ('squared', dict(penalty='squared')), ('huber1', dict(penalty='huber')),
                           ('tolerance05', dict(tolerance_mm=.5)), ('tail10', dict(tail_fraction=.1)),
                           ('gt75', dict(predicted_weight=.25))]:
        add('ap_chamfer_'+name+'_ce', 'ap_chamfer', ap=dict(ap_cd, **settings), ap_ce=ce,
            compare='ap_ce_full', idea='Actual A/P interface distance plus a semantic guard against label reversal; one shared budget.')
    add('ap_chamfer_column_ce', 'ap_chamfer', ap=dict(ap_cd, kind='column_chamfer'), ap_ce=ce,
        compare='ap_chamfer_mean_ce', idea='Match transition points on the same X/Z column, preventing lateral nearest-point shortcuts.')
    for tag, temperature in [('01', .1), ('1', 1.)]:
        add('ap_cut_w1_t'+tag, 'ap_distribution', ap=dict(kind='cut_w1', temperature=temperature),
            compare='ap_ce_full', idea='Wasserstein-1 between candidate-cut probability and the actual interface Y distribution; retain mixed slices.')
    add('ap_cut_expected_l1', 'ap_distribution', ap=dict(kind='cut_expected_l1', temperature=.1),
        compare='ap_cut_w1_t01', idea='Expected absolute cut displacement control; compare with distribution transport.')
    add('pair13_plus_ap_chamfer', 'ap_combination', pair=winner, ap=ap_cd, ap_ce=ce,
        shares=dict(pair=.5, ap=.25, ap_ce=.25), compare='pair13',
        idea='Split the extra budget equally between outer pairs and guarded internal-surface supervision.')
    add('pair13_plus_ap_cut_w1', 'ap_combination', pair=winner, ap=dict(kind='cut_w1', temperature=.1),
        compare='pair13', idea='Combine outer pairs with global A/P cut-distribution alignment.')
    add('pair13_plus_ap_chamfer_outer75', 'ap_combination', pair=winner, ap=ap_cd, ap_ce=ce,
        shares=dict(pair=.75, ap=.125, ap_ce=.125), compare='pair13_plus_ap_chamfer',
        idea='Retain more outer-pair strength; test whether prior outer+AP degradation was a budget tradeoff.')
    add('outer_chamfer_plus_ap_chamfer', 'ap_combination', surface=chamfer, ap=ap_cd, ap_ce=ce,
        shares=dict(surface=.5, ap=.25, ap_ce=.25), compare='chamfer_mean',
        idea='Geometric supervision of both outer surface and guarded internal transition.')
    add('ap_sdf_ce', 'ap_chamfer', ap=dict(kind='sdf', clip_mm=8), ap_ce=ce, compare='ap_chamfer_mean_ce',
        idea='Distance-weighted conditional A/P errors as a surface-free comparator.')
    assert len(runs) == len({e['id'] for e in runs}) == 64
    priority = ('bands', 'pair13', 'chamfer_mean', 'ap_ce_full', 'ap_chamfer_mean_ce',
                'ap_cut_w1_t01', 'pair13_plus_ap_chamfer', 'pair13_plus_ap_cut_w1',
                'band_euclid3', 'band_shell135')
    runs = [next(e for e in runs if e['id'] == name) for name in priority]+[e for e in runs if e['id'] not in priority]
    return runs
