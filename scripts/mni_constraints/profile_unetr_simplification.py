"""Conditional recipe preparation: reproduce current FLOPs, then remove dropout."""
from copy import deepcopy
from pathlib import Path
import shutil
import torch
import monai
from monai.networks.nets import UNETR

from scripts.unetr_subfields import profile as network, geo_recipe, profile_geo_recipe
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.unetr_subfields.contracts import ROOT
from scripts.mni_constraints.profile import audit_call


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4)
    stage=ROOT/'experiments/mni_unetr_delayed05_confirm53t_20261108'
    reference=stage/'protocols/delayed_t05/profile.json';old=read(reference)
    for name in ('geo_recipe.py','core.py'):
        assert digest(ROOT/'scripts/unetr_subfields'/name)==digest(stage/'source/scripts/unetr_subfields'/name)
    network.build_model,network.model_kwargs=geo_recipe.build_model,geo_recipe.model_kwargs
    network.training_loss,network.intermediate_supplement=geo_recipe.training_loss,profile_geo_recipe.supplement
    current=network.measure(4,(64,64,64))
    assert current['update_flops']==old['costs']['dice']['update_flops']
    assert current['parameters']==old['parameters']
    def kwargs(classes,shape):
        return dict(geo_recipe.model_kwargs(classes,shape),dropout_rate=0.)
    network.model_kwargs=kwargs
    network.build_model=lambda classes,shape:UNETR(**kwargs(classes,shape))
    simplified=network.measure(4,(64,64,64))
    assert simplified['parameters']==current['parameters']
    assert simplified['core_flops_per_stage']==current['core_flops_per_stage']
    assert simplified['kwargs']==dict(current['kwargs'],dropout_rate=0.)
    difference=simplified['update_flops']-current['update_flops'];assert difference<0
    forwards=[]
    for dropout in (.1,0.):
        with torch.device('meta'):
            model=UNETR(**dict(geo_recipe.model_kwargs(4,(64,64,64)),dropout_rate=dropout)).eval()
            image=torch.zeros((1,1,64,64,64))
        def forward():
            with torch.no_grad():return model(image)
        row,_=audit_call(forward);forwards.append(row)
    assert forwards[0]==forwards[1]
    assert forwards[0]['flops']==old['forward_calibration']['flops']
    composed=deepcopy(old)
    for cost in composed['costs'].values():cost['update_flops']+=difference
    for row in composed['profiles']:
        if not row['calibration']:row['composed_flops']+=difference
    for arm,cost in composed['costs'].items():
        assert {r['composed_flops'] for r in composed['profiles'] if r['arm']==arm and not r['calibration']}=={cost['update_flops']}
    sources=[Path(__file__),ROOT/'scripts/unetr_subfields/geo_recipe.py',ROOT/'scripts/unetr_subfields/core.py',
             ROOT/'scripts/unetr_subfields/profile.py',ROOT/'scripts/unetr_subfields/profile_geo_recipe.py',
             ROOT/'scripts/unetr_subfields/profile_dice_ce.py',ROOT/'scripts/mni_constraints/profile.py',
             ROOT/'scripts/profile_intermediate_baselines.py',Path(network.reference.__file__)]
    composed.update(baseline_control=simplified,dropout_rate=0.,
        original_profile_sha256=digest(reference),network_update_flops_difference=difference,
        accounting='Current historical Torch2.2.2/MONAI1.6.0 network update exactly reproduced, then dropout removed. Add measured network difference to every original objective update; original logit-loss arithmetic unchanged. Calibration uses eval-mode forward, so unchanged dropout-disabled arithmetic. No training executed.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources})
    out=ROOT/'reports/mni_unetr_constraint_goal_20261108/simplified_recipe_profiles'
    write(out/'current_dropout01_control.json',current)
    write(out/'dropout0_network.json',simplified)
    write(out/'eval_forward_equivalence.json',dict(status='exact_match',dropout01=forwards[0],dropout0=forwards[1]))
    composed['original_baseline_profile_sha256']=composed.pop('baseline_profile_sha256')
    composed['baseline_profile_sha256']=digest(out/'dropout0_network.json')
    write(out/'dropout0_original_constraints.json',composed)
    for source in sources:
        destination=out/'source'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);assert digest(destination)==digest(source)
    write(out/'summary.json',dict(status='profiled_not_trained',current_update_flops=current['update_flops'],
        dropout0_update_flops=simplified['update_flops'],difference_per_update=difference,
        parameters=simplified['parameters'],current_base_updates=53305109592450//current['update_flops'],
        dropout0_base_updates=53305109592450//simplified['update_flops'],
        calibration_eval_forward_exact=True,
        augmentation_note='Stateless paired XY permutation is excluded by the established optimization-only FLOPs convention; no-augmentation keeps the same network/update profile.',
        runtime=dict(torch=torch.__version__,monai=monai.__version__)))
    print(read(out/'summary.json'))


if __name__=='__main__':main()
