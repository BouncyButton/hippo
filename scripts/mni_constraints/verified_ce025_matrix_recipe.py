"""Conditional Dice+0.25CE matrix policy; preserve the confirmed numerical protocol."""
from scripts.cross_dataset_campaign.common import read,REMOTE
from scripts.mni_constraints import verified_unetr_matrix_recipe as shared
from scripts.mni_constraints import unetr_plain_ce025_study as study
from scripts.mni_constraints import unetr_ce025_runtime as runtime,headroom_bands

RECIPES=study.RECIPES
assert_same_training_protocol=shared.assert_same_training_protocol


def selected_recipe(result, *, confirmation_study=study):
    return shared.selected_recipe(result,confirmation_study=confirmation_study,allowed_recipes=RECIPES)


def changed_protocol(old,profile,recipe,*,stage_name,profile_sha256):
    assert recipe in RECIPES
    if old['model_identity']=='unetr_vit_b16':
        p=study.protocol(old,profile,recipe)
        p.update(name=stage_name,data=str(REMOTE/stage_name/'data/MNI.pkl'),profile_sha256=profile_sha256,
            decision_rule='Complete every original constraint combination for all three seeds and folds; no score-dependent stopping.',
            selection_caveat='One shared recipe confirmed on nine development pairs; other combinations remain experimental.')
        runtime.validate(p)
        return p
    return shared.changed_protocol(old,profile,shared.COSINE_RECIPE,stage_name=stage_name,profile_sha256=profile_sha256)


def train(args):
    if read(args.protocol_dir/'protocol.json')['model_identity']=='unetr_vit_b16':
        runtime.train(args)
    else:
        headroom_bands.train(args)
