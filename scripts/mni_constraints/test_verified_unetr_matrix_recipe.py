"""Guard the nine-pair gate and the exact-cap nnUNet baseline during extension."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, ARMS
from scripts.mni_constraints import verified_unetr_matrix_recipe as helper
from scripts.mni_constraints import unetr_simple_recipe_study as study
from scripts.mni_constraints import final53_study as legacy
from scripts.mni_constraints.campaign import training_flops


def main():
    original=study.ROOT/'experiments'/study.source.ORIGINAL
    drop=study.ROOT/'reports/mni_unetr_constraint_goal_20261108/simplified_recipe_profiles/dropout0_original_constraints.json'
    checked=0
    for recipe in helper.RECIPES:
        for model in legacy.MODELS:
            for arm in ARMS:
                folder=original/'protocols'/(model+'_'+arm)
                old=read(folder/'protocol.json')
                profile_file=drop if model=='unetr_vit_b16' and recipe['dropout_rate']==0. else folder/'profile.json'
                profile=read(profile_file)
                p=helper.changed_protocol(old,profile,recipe,stage_name='conditional_test',profile_sha256=digest(profile_file))
                s=p['models'][p['trainer_alias']]['objectives'][arm]
                assert training_flops(s['max_steps'],s)<=study.CAP<training_flops(s['max_steps']+1,s)
                if model=='unetr_vit_b16':
                    assert s['max_steps']==409
                    assert p['dropout_rate']==recipe['dropout_rate'] and p['augmentation']==recipe['augmentation']
                    assert p['lr_policy']==('cap_matched' if recipe==helper.COSINE_RECIPE else 'hold_half')
                else:
                    expected=legacy.changed_protocol(old,profile)
                    ignored={'name','data','selection_caveat','profile_sha256'}
                    assert {k:v for k,v in p.items() if k not in ignored}=={k:v for k,v in expected.items() if k not in ignored}
                    if model=='nnunet' and arm=='dice':
                        assert s['max_steps']==150 and training_flops(150,s)==study.CAP
                checked+=1
    candidate='plain_full'
    rows=[dict(candidate=candidate,recipe='plain',arm='dice_bands_edge_sdf_depth',seed=s,fold=f,gain_pp=.1) for s,f in study.PAIRS]
    result=dict(status='complete',outcome='nine_pair_success',winner=candidate,rows=rows,
        summary=[dict(candidate=candidate,all_nine_improved=True,rejected=False,mean_absolute_gain_vs_old_baseline_pp=.2)])
    assert helper.selected_recipe(result)['recipe']['key']=='plain'
    for mutation in ('partial','missing','zero'):
        bad=deepcopy(result)
        if mutation=='partial':bad['status']='partial'
        elif mutation=='missing':bad['rows'].pop()
        else:bad['rows'][0]['gain_pp']=0.
        try:helper.selected_recipe(bad)
        except AssertionError:pass
        else:raise AssertionError(mutation)
    from scripts.mni_constraints import unetr_plain_subfields_study as plain
    for candidate in plain.CANDIDATES:
        trial=deepcopy(result)
        arm=next(t['arm'] for t in plain.tasks() if t.get('candidate')==candidate)
        trial['winner']=candidate
        trial['summary'][0]['candidate']=candidate
        for row in trial['rows']:row.update(candidate=candidate,arm=arm)
        assert helper.selected_recipe(trial,confirmation_study=plain)['confirmed_arm']==arm
        trial['rows'][0]['gain_pp']=-.1
        try:helper.selected_recipe(trial,confirmation_study=plain)
        except AssertionError:pass
        else:raise AssertionError('Negative plain-study pair passed the gate')
    from scripts.mni_constraints import unetr_plain_cosine_study as cosine
    assert cosine.RECIPES[0]==helper.COSINE_RECIPE
    for arm in cosine.ARMS:
        folder=cosine.ROOT/'experiments'/cosine.NAME/'protocols'/('plain_cosine_'+arm)
        current=read(folder/'protocol.json')
        destination=helper.changed_protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),
            helper.COSINE_RECIPE,stage_name='conditional_test',profile_sha256=digest(folder/'profile.json'))
        helper.assert_same_training_protocol(current,destination)
        for field,value in (('lr_policy','hold_half'),('dropout_rate',.1),
                            ('calibration_target_ratio',.2),('pkl_sha256','wrong')):
            changed=deepcopy(destination);changed[field]=value
            try:helper.assert_same_training_protocol(current,changed)
            except AssertionError:pass
            else:raise AssertionError(f'Unsafe reuse accepted: {field}')
    for candidate in cosine.CANDIDATES:
        trial=deepcopy(result)
        arm=next(t['arm'] for t in cosine.tasks() if t.get('candidate')==candidate)
        trial['winner']=candidate
        trial['summary'][0]['candidate']=candidate
        for row in trial['rows']:row.update(candidate=candidate,arm=arm,recipe='plain_cosine')
        assert helper.selected_recipe(trial,confirmation_study=cosine)['recipe']==helper.COSINE_RECIPE
        trial['rows'][0]['gain_pp']=0.
        try:helper.selected_recipe(trial,confirmation_study=cosine)
        except AssertionError:pass
        else:raise AssertionError('Tied cosine-study pair passed the gate')
    old_unetr,old_other=helper.runtime.train,helper.headroom_bands.train
    calls=[]
    try:
        helper.runtime.train=lambda args:calls.append('unetr')
        helper.headroom_bands.train=lambda args:calls.append('other')
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);args=SimpleNamespace(protocol_dir=folder)
            for model in legacy.MODELS:
                write(folder/'protocol.json',dict(model_identity=model));helper.train(args)
        assert calls.count('unetr')==1 and calls.count('other')==3
    finally:
        helper.runtime.train,helper.headroom_bands.train=old_unetr,old_other
    print(f'{checked} conditional protocols checked; all UNETR409, nnUNet baseline exactly150; nine-pair gate, exact numerical reuse and model dispatch passed. No matrix built or submitted.')


if __name__=='__main__':
    main()
