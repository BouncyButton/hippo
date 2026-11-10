"""A bounded refinement selected for shared-tissue A/P improvement."""
from copy import deepcopy
from . import ap82_matrix as original


RULE = {
    'shared_swap_delta': '< 0',
    'mean_case_shared_error_rate_delta': '< 0',
    'delta_foreground_dice': '>= 0',
    'delta_outer_errors': '<= 0',
    'delta_ap_interface_chamfer_mm': '< 0',
    'delta_ap_interface_hd95_mm': '<= 0',
    'scope': 'Mixed bands/endpoints on the frozen outer winner; same 52 development cases.',
    'ranking': 'Lowest mean paired common-tissue error-rate delta, then interface Chamfer delta, then identifier.',
    'fallback': 'No qualifying parent means no targeted A/P runs; preserve the outer sweep.',
}


def eligible(row):
    return (row['background'] == 'winner' and row['cases_with_shared_tissue'] == 52
        and row['shared_swap_delta'] < 0 and row['mean_case_shared_error_rate_delta'] < 0
        and row['delta_foreground_dice'] >= 0 and row['delta_outer_errors'] <= 0
        and row['delta_ap_interface_chamfer_mm'] < 0 and row['delta_ap_interface_hd95_mm'] <= 0)


def select_parent(stage1, evidence):
    by = {e['id']:e for e in stage1}
    candidates = [r for r in evidence['rows'] if eligible(r)
                  and 0 < by[r['objective']]['ap82']['band_share'] < 1]
    if not candidates: return None
    selected = min(candidates,key=lambda r:(r['mean_case_shared_error_rate_delta'],
                   r['delta_ap_interface_chamfer_mm'],r['objective']))
    return by[selected['objective']]


def targeted_entries(scope, evidence):
    parent = select_parent(scope['stage1'],evidence)
    if parent is None: return [], None
    spec = deepcopy(parent['ap82'])
    ratio, share = spec['gradient_ratio'], spec['band_share']
    # Change one factor block at a time. Prioritize local strength and spatial
    # support, then error emphasis and class-side balance. No automatic crosses.
    changes = [
        ('strength_low', {'gradient_ratio':ratio*.75}),
        ('strength_high', {'gradient_ratio':ratio*1.25}),
        ('band_lower', {'band_share':max(.125,share-.125)}),
        ('band_higher', {'band_share':min(.875,share+.125)}),
        ('width2', {'width':2.}),
        ('width1', {'width':1.}),
        ('hard25', {'anterior_fraction':.25,'posterior_fraction':.25}),
        ('emphasis75', {'anterior_emphasis':.75,'posterior_emphasis':.75}),
        ('anterior40', {'anterior_mass':.4}),
        ('anterior60', {'anterior_mass':.6}),
    ]
    seen = {original.scientific_key(e) for e in scope['stage1']}
    entries = []
    for tag, update in changes:
        for bg in original.BACKGROUNDS:
            e = original.entry(scope['bases'],scope['coefficients'],bg,2,'target_'+tag,
                               dict(spec,**update),{tag:update})
            if original.scientific_key(e) in seen: continue
            e['family']='ap82_targeted'
            e['ap82_parent']=parent['id']
            e['idea']='One-block refinement of shared-tissue A/P improvement with preserved foreground Dice and total outer errors.'
            seen.add(original.scientific_key(e));entries.append(e)
    assert len(entries) <= 20 and len(entries)%2 == 0
    return entries, parent['id']
