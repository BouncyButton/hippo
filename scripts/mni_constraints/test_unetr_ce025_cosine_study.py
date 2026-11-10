"""Verify the isolated schedule change, graph reuse, and matched-control plan."""
import ast
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, digest
from scripts.mni_constraints import unetr_ce025_cosine_study as cosine
from scripts.mni_constraints import unetr_plain_ce025_study as prior
from scripts.mni_constraints.campaign import training_flops
from scripts.unetr_subfields.cap_extension import lr_at


def main():
    profile_file = cosine.ROOT/'reports/mni_unetr_constraint_goal_20261108/ce025_recipe_profiles/ce025_original_constraints.json'
    profile = read(profile_file)
    original = cosine.ROOT/'experiments'/cosine.source.ORIGINAL
    allowed = {'name', 'data', 'timing_policy', 'lr_policy',
               'learning_rate_policy', 'lr_schedule_details', 'base_loss_adapter_sha256'}
    checks = []
    for arm in cosine.ARMS:
        parent = read(original/'protocols'/('unetr_vit_b16_'+arm)/'protocol.json')
        old = prior.protocol(parent, profile, prior.RECIPES[0])
        new = cosine.protocol(parent, profile, cosine.RECIPES[0])
        assert {k:v for k,v in new.items() if k not in allowed} == {
            k:v for k,v in old.items() if k not in allowed}
        schedule = new['models']['unet']['objectives'][arm]
        assert schedule['max_steps'] == 409
        assert training_flops(409, schedule) <= cosine.CAP < training_flops(410, schedule)
        assert new['lr_schedule_details'] == dict(
            policy='cap_matched', horizon_updates=409, warmup_updates=24, decay_start_index=24)
        checks.append(arm)
    hold = [lr_at(i,409,.0006,'unetr_vit_b16','hold_half') for i in range(409)]
    curve = [lr_at(i,409,.0006,'unetr_vit_b16','cap_matched') for i in range(409)]
    assert curve[:25] == hold[:25] and curve[-1] == hold[-1]
    assert all(a <= b for a,b in zip(curve,hold))
    assert all(curve[i+1] < curve[i] for i in range(24,408))
    assert len(cosine.tasks()) == 27 and len(cosine.CANDIDATES) == 2
    assert len([t for t in cosine.tasks() if t['role'] == 'baseline']) == 9
    # The measured computation graph is unchanged. Only the runtime's protocol
    # guard differs; every training, loss-binding, and model function is identical.
    old_tree = ast.parse(Path(prior.runtime.__file__).read_text())
    new_tree = ast.parse(Path(cosine.runtime.__file__).read_text())
    functions = lambda tree: {n.name:ast.dump(n) for n in tree.body
                             if isinstance(n, ast.FunctionDef) and n.name != 'validate'}
    assert functions(old_tree) == functions(new_tree)
    assert cosine.runtime.dice is prior.runtime.dice
    assert profile['base_loss_adapter_sha256'] == digest(Path(prior.runtime.__file__))
    assert profile['base_loss_source_sha256'] == digest(Path(cosine.runtime.dice.__file__))
    print(dict(status='passed', arms=checks,
        only_numerical_change='LR hold-half to immediate post-warmup cosine',
        updates=409, graph_functions_identical=True,
        profile_sha256=digest(profile_file), fresh_matched_controls=9))


if __name__ == '__main__':
    main()
