"""Conditional matrix recipe helpers; no deployment or training is started here."""
from copy import deepcopy

from scripts.cross_dataset_campaign.common import read, REMOTE
from scripts.mni_constraints import final53_study as legacy
from scripts.mni_constraints import unetr_simple_recipe_study as study
from scripts.mni_constraints import unetr_simple_recipe_runtime as runtime
from scripts.mni_constraints import headroom_bands

COSINE_RECIPE=dict(key='plain_cosine',dropout_rate=0.,augmentation='none')
RECIPES=(*study.RECIPES,COSINE_RECIPE)


def assert_same_training_protocol(original, destination):
    """Allow reuse only when every numerical/data/source field is identical."""
    descriptive={'name','data','primary_metric','timing_policy','selection_caveat','decision_rule'}
    # Relocation is safe only because pkl_sha256 remains among compared fields.
    assert original['pkl_sha256']==destination['pkl_sha256']
    left={k:v for k,v in original.items() if k not in descriptive}
    right={k:v for k,v in destination.items() if k not in descriptive}
    differences=sorted(k for k in left.keys()|right.keys() if left.get(k)!=right.get(k))
    assert left==right, f'Training protocol changed: {differences}'


def selected_recipe(result, *, confirmation_study=study, allowed_recipes=RECIPES):
    """The caller must additionally verify archives and complete cluster cleanup."""
    assert result['status']=='complete' and result['outcome']=='nine_pair_success'
    winner=result['winner']
    assert winner in confirmation_study.CANDIDATES
    summary=next(s for s in result['summary'] if s['candidate']==winner)
    assert summary['all_nine_improved'] and not summary['rejected']
    rows=[r for r in result['rows'] if r['candidate']==winner]
    assert len(rows)==9 and {(r['seed'],r['fold']) for r in rows}==set(confirmation_study.PAIRS)
    assert all(r['gain_pp']>0 for r in rows)
    task=next(t for t in confirmation_study.tasks() if t.get('candidate')==winner)
    assert all(r['recipe']==task['recipe'] and r['arm']==task['arm'] for r in rows)
    recipe=deepcopy(next(r for r in confirmation_study.RECIPES if r['key']==task['recipe']))
    assert recipe in allowed_recipes
    return dict(recipe=recipe,candidate=winner,confirmed_arm=task['arm'],
                absolute_performance=summary['mean_absolute_gain_vs_old_baseline_pp'])


def changed_protocol(old, profile, recipe, *, stage_name, profile_sha256):
    """Apply the verified UNETR base to every arm; keep other model math intact."""
    assert recipe in RECIPES
    p=legacy.changed_protocol(old,profile)
    p.update(name=stage_name,data=str(REMOTE/stage_name/'data/MNI.pkl'),profile_sha256=profile_sha256,
        selection_caveat='One shared UNETR base+constraint recipe verified on all9 development pairs. Apply that base and delayed policy to every original combination as user requested; benefits for other combinations are still experimental. Other model recipes unchanged, including nnUNet LR0.02.')
    if p['model_identity']=='unetr_vit_b16':
        p.update(augmentation=recipe['augmentation'],dropout_rate=recipe['dropout_rate'],
                 unetr_simplification={k:recipe[k] for k in ('augmentation','dropout_rate')})
        if recipe==COSINE_RECIPE:
            p.update(lr_policy='cap_matched',
                learning_rate_policy='24-update warmup then bounded cosine to1%peak; no hold-half phase;409-update horizon.',
                lr_schedule_details=dict(policy='cap_matched',horizon_updates=409,warmup_updates=24,decay_start_index=24))
        runtime.validate(p)
    return p


def train(args):
    """Dispatch once before training; add no operations to other models' updates."""
    protocol=read(args.protocol_dir/'protocol.json')
    if protocol['model_identity']=='unetr_vit_b16':
        runtime.train(args)
    else:
        headroom_bands.train(args)
