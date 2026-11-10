"""Check that the conditional LR experiment changes only the intended schedule."""
from scripts.cross_dataset_campaign.common import read,digest
from scripts.mni_constraints import unetr_plain_cosine_study as cosine
from scripts.mni_constraints import unetr_simple_recipe_study as prior
from scripts.mni_constraints.campaign import training_flops
from scripts.unetr_subfields.cap_extension import lr_at


def main():
    profile_file=cosine.ROOT/'reports/mni_unetr_constraint_goal_20261108/simplified_recipe_profiles/dropout0_original_constraints.json'
    profile=read(profile_file);original=cosine.ROOT/'experiments'/cosine.source.ORIGINAL
    allowed={'name','data','decision_rule','timing_policy','lr_policy','learning_rate_policy','lr_schedule_details'}
    checks=[]
    for arm in cosine.ARMS:
        parent=read(original/'protocols'/('unetr_vit_b16_'+arm)/'protocol.json')
        old=prior.protocol(parent,profile,dict(key='plain',dropout_rate=0.,augmentation='none'))
        new=cosine.protocol(parent,profile,cosine.RECIPES[0])
        assert {k:v for k,v in new.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
        schedule=new['models']['unet']['objectives'][arm]
        assert schedule['max_steps']==409
        assert training_flops(409,schedule)<=cosine.CAP<training_flops(410,schedule)
        assert new['lr_schedule_details']==dict(policy='cap_matched',horizon_updates=409,warmup_updates=24,decay_start_index=24)
        checks.append(arm)
    hold=[lr_at(i,409,.0006,'unetr_vit_b16','hold_half') for i in range(409)]
    curve=[lr_at(i,409,.0006,'unetr_vit_b16','cap_matched') for i in range(409)]
    assert curve[:25]==hold[:25] and curve[-1]==hold[-1]
    assert all(a<=b for a,b in zip(curve,hold))
    assert curve[25]<hold[25] and all(curve[i+1]<curve[i] for i in range(24,408))
    assert len(cosine.tasks())==36 and len(cosine.CANDIDATES)==3
    print(dict(status='passed',arms=checks,only_numerical_change='LR hold-half to immediate post-warmup cosine',updates=409,profile_sha256=digest(profile_file),stage_frozen=False))


if __name__=='__main__':main()
