"""Reproduce the plain DiceCE profile, then measure Dice+half-CE and every auxiliary."""
from pathlib import Path
import shutil
import torch
import monai
from monai.networks.nets import UNETR

from scripts.cross_dataset_campaign.common import read,write,digest,ARMS
from scripts.unetr_subfields.contracts import ROOT
from scripts.unetr_subfields import profile as network,geo_recipe,profile_geo_recipe
from scripts.unetr_subfields import foreground_dice_ce050 as dice
from scripts.mni_constraints import objectives,unetr_ce050_cosine_runtime as runtime
from scripts.mni_constraints.profile import audit_call,target

OUT=ROOT/'reports/mni_unetr_constraint_goal_20261108/ce050_cosine_recipe_profiles'


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4)
    oldfile=ROOT/'reports/mni_unetr_constraint_goal_20261108/simplified_recipe_profiles/dropout0_original_constraints.json'
    old=read(oldfile)
    network.model_kwargs=lambda classes,shape:dict(geo_recipe.model_kwargs(classes,shape),dropout_rate=0.)
    network.build_model=lambda classes,shape:UNETR(**network.model_kwargs(classes,shape))
    network.training_loss=geo_recipe.training_loss;network.intermediate_supplement=profile_geo_recipe.supplement
    control=network.measure(4,(64,64,64))
    assert control['update_flops']==old['costs']['dice']['update_flops']==130103216359
    network.training_loss=dice.training_loss
    measured=network.measure(4,(64,64,64))
    assert measured['parameters']==control['parameters'] and measured['kwargs']==control['kwargs']
    assert measured['update_flops']>0
    with torch.device('meta'):
        model=network.build_model(4,(64,64,64)).eval()
    def forward():
        # Match the historical audit boundary, including zero-FLOP input creation.
        with torch.no_grad():return model(torch.zeros((1,1,64,64,64),device='meta'))
    forward_row,out=audit_call(forward)
    write(OUT/'forward_comparison.json',dict(current=forward_row,previous=old['forward_calibration']))
    assert forward_row==old['forward_calibration']
    shapes=[list(out.shape)];del model,out
    settings=read(ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json')['geometry']
    costs,rows,base_logit={},[],{}
    with runtime.use_base_loss():
        for arm in (*ARMS,'dice_edge','dice_edge_sdf'):
            entries=[]
            for geometry in ('ellipsoid','box'):
                label=target(geometry)
                sdf=objectives.signed_distance_maps(label) if 'sdf' in arm else None
                depth=objectives.geometry_weights(label,torch.eye(4).numpy(),settings) if 'depth' in arm else None
                for calibration in ([False] if arm=='dice' else [False,True]):
                    torch.manual_seed(0);outputs=torch.randn(shapes[0],requires_grad=True)
                    def loss_call():
                        if calibration:
                            return objectives.calibration_case(outputs,label,arm,sdf,None,depth,identity='unetr_vit_b16')
                        loss=objectives.training_loss(outputs,label,arm,dict(bands=.04,edge=.1,sdf=.1,depth=.1),1.,sdf,None,depth,.2,identity='unetr_vit_b16')
                        loss.backward();assert torch.isfinite(loss)
                        return float(loss.detach())
                    row,_=audit_call(loss_call)
                    row.update(arm=arm,geometry=geometry,calibration=calibration)
                    if arm=='dice':base_logit[geometry]=row['flops']
                    row['composed_flops']=(forward_row['flops']+row['flops'] if calibration else
                        measured['update_flops']+row['flops']-base_logit[geometry])
                    entries.append(row);rows.append(row)
            training={r['composed_flops'] for r in entries if not r['calibration']}
            calibration={r['composed_flops'] for r in entries if r['calibration']}
            assert len(training)==1 and len(calibration)<=1
            costs[arm]=dict(update_flops=training.pop(),calibration_flops=20*calibration.pop()+8192 if calibration else 0)
            print(arm,costs[arm],flush=True)
            write(OUT/'profile.partial.json',dict(costs=costs,profiles=rows))
    sources=[Path(__file__),Path(runtime.__file__),Path(dice.__file__),Path(objectives.__file__),
        ROOT/'scripts/unetr_subfields/foreground_squared_dice.py',ROOT/'scripts/mni_constraints/profile.py',ROOT/'scripts/unetr_subfields/profile.py',
        ROOT/'scripts/unetr_subfields/profile_geo_recipe.py',ROOT/'scripts/unetr_subfields/profile_dice_ce.py',
        ROOT/'scripts/unetr_subfields/geo_recipe.py',ROOT/'scripts/unetr_subfields/core.py',
        ROOT/'scripts/profile_intermediate_baselines.py',Path(network.reference.__file__)]
    write(OUT/'plain_dicece_control.json',control);write(OUT/'ce050_network.json',measured)
    result=dict(status='profiled',model='unetr_vit_b16',loss_recipe=runtime.LOSS_RECIPE,
        parameters=measured['parameters'],costs=costs,profiles=rows,baseline_control=measured,
        baseline_profile_sha256=digest(OUT/'ce050_network.json'),original_profile_sha256=digest(oldfile),
        forward_calibration=forward_row,output_shapes=shapes,objective_sha256=digest(Path(objectives.__file__)),
        base_loss_adapter_sha256=digest(Path(runtime.__file__)),base_loss_source_sha256=digest(Path(dice.__file__)),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        runtime=dict(torch=torch.__version__,monai=monai.__version__),
        accounting='Exact historical plain DiceCE network control reproduced; measured Dice+0.5CE full update. Each original auxiliary adds measured FP32 objective/backward delta. Calibration20eval forwards plus logit gradients and8192hostFLOPs, separate from optimization cap. Existing dense MAC=2/scalar convention; preprocessing,validation,IO excluded.')
    write(OUT/'ce050_original_constraints.json',result)
    for source in sources:
        destination=OUT/'source'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);assert digest(destination)==digest(source)
    write(OUT/'summary.json',dict(status='profiled_not_trained',dicece_update_flops=control['update_flops'],
        ce050_update_flops=measured['update_flops'],base_updates=53305109592450//measured['update_flops'],
        calibration_crops=20,calibration_host_flops=8192,original_auxiliary_definitions=True,
        profile_sha256=digest(OUT/'ce050_original_constraints.json')))
    print(read(OUT/'summary.json'),flush=True)


if __name__=='__main__':main()
