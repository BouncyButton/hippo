"""Check half-CE base gradients, calibration binding and fixed-budget schedules."""
from pathlib import Path
import torch
from monai.losses import DiceCELoss
from scripts.cross_dataset_campaign.common import read,digest,ARMS
from scripts.unetr_subfields import geo_recipe
from scripts.mni_constraints import unetr_delayed_combinations as source
from scripts.mni_constraints import unetr_ce050_cosine_study as study
from scripts.mni_constraints import unetr_ce050_cosine_runtime as runtime,objectives,campaign
from scripts.mni_constraints.profile import target


def main():
    torch.set_num_threads(4)
    profile=read(source.ROOT/'reports/mni_unetr_constraint_goal_20261108/ce050_cosine_recipe_profiles/ce050_original_constraints.json')
    original=source.ROOT/'experiments'/source.ORIGINAL
    quarter=read(source.ROOT/'reports/mni_unetr_constraint_goal_20261108/ce025_recipe_profiles/ce025_original_constraints.json')
    assert profile['costs']==quarter['costs']
    from scripts.mni_constraints import unetr_ce025_cosine_study as previous
    allowed={'name','data','loss_recipe','loss_arguments','loss_weights','training_loss',
             'base_loss_adapter_sha256','base_loss_source_sha256','decision_rule','timing_policy','primary_metric'}
    protocols={}
    for arm in ARMS:
        p=study.protocol(read(original/'protocols'/('unetr_vit_b16_'+arm)/'protocol.json'),profile,study.RECIPES[0])
        runtime.validate(p);protocols[arm]=p
        old=previous.protocol(read(original/'protocols'/('unetr_vit_b16_'+arm)/'protocol.json'),quarter,previous.RECIPES[0])
        assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
        assert p['loss_arguments']==dict(old['loss_arguments'],lambda_ce=.5)
        assert p['loss_weights']==dict(dice=1.,ce=.5)
        schedule=p['models']['unet']['objectives'][arm]
        assert schedule['max_steps']==409
        assert campaign.training_flops(409,schedule)<=source.CAP<campaign.training_flops(410,schedule)
        assert schedule['calibration_flops']==profile['costs'][arm]['calibration_flops']
    torch.manual_seed(812);label=target('ellipsoid');z=torch.randn((1,4,64,64,64),requires_grad=True)
    criterion=DiceCELoss(**geo_recipe.LOSS_ARGUMENTS)
    expected=criterion.dice(z.float(),label)+.5*criterion.ce(z.float(),label)
    gradient=torch.autograd.grad(expected,z,retain_graph=True)[0]
    actual=runtime.base_loss(z,label,'unetr_vit_b16')
    assert torch.equal(expected,actual)
    assert torch.equal(gradient,torch.autograd.grad(actual,z,retain_graph=True)[0])
    assert criterion.dice(z,label)<actual<criterion(z,label)
    before=objectives.base_loss
    with runtime.use_base_loss():
        sdf=objectives.signed_distance_maps(label)
        depth=objectives.geometry_weights(label,torch.eye(4).numpy(),protocols['dice']['depth_geometry'])
        for arm in ARMS[1:]:
            zero=objectives.training_loss(z,label,arm,dict.fromkeys(objectives.TERMS,0.),1.,sdf,None,depth,1.,identity='unetr_vit_b16')
            assert torch.equal(zero,expected)
            assert torch.equal(torch.autograd.grad(zero,z,retain_graph=True)[0],gradient)
            cal=objectives.calibration_case(z,label,arm,sdf,None,depth,identity='unetr_vit_b16')
            assert abs(cal['gradient_pair_means']['base:base']-float((gradient*gradient).mean()))<1e-20
            weights,stats=objectives.select_weights([cal],arm,.05,.75)
            assert all(v['weighted_median_ratio']<=.0500001 and v['weighted_p95_ratio']<=.7500001 for v in stats.values())
        # Other architectures retain their exact original base loss under the context.
        assert torch.equal(objectives.base_loss(z,label,'unet'),before(z,label,'unet'))
    assert objectives.base_loss is before
    with torch.device('meta'):
        model=runtime.build_model(protocols['dice'])
        assert list(model(torch.zeros((1,1,64,64,64))).shape)==[1,4,64,64,64]
    assert len(study.tasks())==27 and len(study.CANDIDATES)==2
    print('Passed exact Dice+0.5CE loss/gradient, all13zero-aux identities and calibration base gradients, restoration/other-model isolation, actual dropout0 model, all14schedules409updates undercap. No training launched.')


if __name__=='__main__':main()
