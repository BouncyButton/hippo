"""Verify the actual loss substitution, calibration base and profiled schedules."""
from pathlib import Path
import torch
from monai.losses import DiceCELoss

from scripts.cross_dataset_campaign.common import read
from scripts.unetr_subfields import geo_recipe
from scripts.mni_constraints import unetr_plain_diceonly_study as study,objectives
from scripts.mni_constraints.profile import target


def main():
    torch.set_num_threads(4)
    profile=read(study.ROOT/'reports/mni_unetr_constraint_goal_20261108/diceonly_recipe_profiles/diceonly_original_constraints.json')
    original=study.ROOT/'experiments'/study.source.ORIGINAL
    protocols={}
    for arm in study.ARMS:
        old=read(original/'protocols'/('unetr_vit_b16_'+arm)/'protocol.json')
        p=study.protocol(old,profile,study.RECIPES[0]);protocols[arm]=p
        s=p['models']['unet']['objectives'][arm]
        assert s['max_steps']==409 and study.campaign.training_flops(409,s)<=study.CAP<study.campaign.training_flops(410,s)
        assert s['calibration_flops']==profile['costs'][arm]['calibration_flops']
    torch.manual_seed(812);label=target('ellipsoid');z=torch.randn((1,4,64,64,64),requires_grad=True)
    criterion=DiceCELoss(**geo_recipe.LOSS_ARGUMENTS)
    expected=criterion.dice(z.float(),label);gradient=torch.autograd.grad(expected,z,retain_graph=True)[0]
    actual=study.runtime.base_loss(z,label,'unetr_vit_b16')
    assert torch.equal(expected,actual)
    assert torch.equal(gradient,torch.autograd.grad(actual,z,retain_graph=True)[0])
    assert criterion(z,label)>actual
    before=objectives.base_loss
    with study.runtime.use_base_loss():
        sdf=objectives.signed_distance_maps(label)
        depth=objectives.geometry_weights(label,torch.eye(4).numpy(),protocols['dice']['depth_geometry'])
        for arm in study.ARMS[1:]:
            zero=objectives.training_loss(z,label,arm,dict.fromkeys(objectives.TERMS,0.),1.,sdf,None,depth,1.,identity='unetr_vit_b16')
            assert torch.equal(zero,expected)
            assert torch.equal(torch.autograd.grad(zero,z,retain_graph=True)[0],gradient)
            cal=objectives.calibration_case(z,label,arm,sdf,None,depth,identity='unetr_vit_b16')
            assert abs(cal['gradient_pair_means']['base:base']-float((gradient*gradient).mean()))<1e-20
            weights,stats=objectives.select_weights([cal],arm,.05,.75)
            assert all(v['weighted_median_ratio']<=.0500001 and v['weighted_p95_ratio']<=.7500001 for v in stats.values())
    assert objectives.base_loss is before
    with torch.device('meta'):
        model=study.runtime.build_model(protocols['dice'])
        assert list(model(torch.zeros((1,1,64,64,64))).shape)==[1,4,64,64,64]
    assert len(study.tasks())==45 and len(study.CANDIDATES)==4
    print('Passed: exact DiceCE.dice value/gradient, CE absent from actual base, zero-aux exact replay, calibration uses Dice-only gradient, restored binding, actual dropout0 model, all5schedules409updates undercap.45fresh runs maximum; no training launched.')


if __name__=='__main__':main()
