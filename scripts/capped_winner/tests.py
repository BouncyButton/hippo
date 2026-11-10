"""Loss equivalence, budget coverage, geometry and production GPU gates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.campaign90 import MODELS, build_model, write_json
from scripts.capped_depth_delay.objectives import training_loss as old_loss, augmented_affine
from scripts.band_focus import winner_objectives, combo_objectives
from scripts.band_focus.objectives import select_weights as base_select
from scripts.band_chamfer.geometry import Targets as OldTargets
from scripts.band_focus.geometry import Targets
from scripts.profile_unet_constraint_budget import synthetic_target
from scripts.profile_intermediate_baselines import audit, intermediate_supplement
from .objectives import ARM, ENTRY, SETTINGS, auxiliary, training_loss, calibration_case, select_weights
from .geometry import make_targets, transformed_valid
from .profile import measure

GEOMETRY=dict(band_mm=2.,roi_margin_mm=8.,distance_clip_mm=8.,layer_grid_mm=1.,sparse_band_voxels=64)


def cpu_checks():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    y=synthetic_target('ellipsoid')
    affine=np.eye(4);affine[:3,:3]=np.array([[0,1,0],[1,0,0],[0,0,1]])
    target=make_targets(y,affine,affine,[64]*3,GEOMETRY,'cpu')
    reference=Targets(Path('.'),dict(native_geometry={'synthetic':dict(native_shape=[64]*3)}),ENTRY)
    expected=reference.get(dict(case_name='synthetic',label=y[0],affine=affine),'cpu')
    for key in ('valid','foreground','distance'):
        assert torch.equal(target[key],expected[key]),key
    for actual,wanted in zip(target['pairs'],expected['pairs']):
        for side in ('inside','outside'):
            assert torch.equal(actual[side]['index'],wanted[side]['index'])
            assert torch.equal(actual[side]['weight'],wanted[side]['weight'])
    z=torch.randn(1,3,64,64,64,requires_grad=True)
    entry=dict(ENTRY,objective_version='winner_v1')
    wanted=winner_objectives.components(z,y,entry,target)
    actual=auxiliary(z,y,target)
    for name in ('bands','pair'):
        assert torch.equal(wanted[name],actual[name])
        ga=torch.autograd.grad(wanted[name],z,retain_graph=True)[0]
        gb=torch.autograd.grad(actual[name],z,retain_graph=True)[0]
        assert torch.equal(ga,gb)
    weights=dict(bands=.04,pair=.006,depth=.01)
    warm=training_loss(z,y,'dice',{},1.)
    assert torch.equal(warm,old_loss(z,y,'dice',{},1.))
    new=training_loss(z,y,ARM,weights,1.,depth_weights=target)
    composed=warm
    for name in ('bands','pair','depth'):composed=composed+weights[name]*actual[name]
    assert torch.equal(new,composed)
    torch.testing.assert_close(torch.autograd.grad(new,z,retain_graph=True)[0],
        torch.autograd.grad(composed,z)[0],rtol=1e-6,atol=1e-10)
    row=calibration_case(z,y,ARM,depth_weights=target)
    selected,_=select_weights([row],ARM)
    base=base_select([row],ENTRY,SETTINGS)
    combo=combo_objectives.select_weights([row],dict(base_coefficients=base['coefficients'],
        combo=dict(terms=['depth'],gradient_ratio=.1)),SETTINGS)
    assert selected==combo['coefficients'] and all(np.isfinite(v) and v>0 for v in selected.values())
    # Geometry rebuilt from the same augmented annotation and valid native FOV.
    import train_compute_matched_baselines as trainer
    from scripts.edge_ablation.objectives import valid_field
    native=[50,44,38]
    original=valid_field([64]*3,native)
    accepted=0
    for seed in range(12):
        rng=torch.Generator().manual_seed(seed)
        state=rng.get_state()
        _,aug,stats=trainer.augment_mild(torch.randn_like(y.float()),y,generator=rng)
        converted=augmented_affine(affine,state,stats,y.shape[-3:])
        a=make_targets(aug,affine,converted,native,GEOMETRY,'cpu')
        assert torch.isfinite(a['depth']).all() and abs(float(a['depth'].sum())-1)<1e-5
        assert not (a['foreground'] & ~a['valid']).any()
        accepted+=stats['spatial_applied']
    assert accepted>0
    # Actual graph costs must be below the maximum-cardinality loss bound.
    profile=json.loads((Path(__file__).resolve().parents[2]/'reports/capped_winner_31976_20261013/loss_profile.json').read_text())
    audit.reference.supplementary_flops=intermediate_supplement
    checked=[]
    for kind in ('ellipsoid','box'):
        labels=synthetic_target(kind)
        t=make_targets(labels,affine,affine,[64]*3,GEOMETRY,'cpu')
        variable=torch.randn(1,3,64,64,64,requires_grad=True)
        def backward():
            terms=auxiliary(variable,labels,t)
            sum(.1*v for v in terms.values()).backward()
        count=measure(backward)['flops']+variable.numel()
        assert count<=profile['extra_update_flops_bound']
        checked.append(count)
    return dict(status='passed',unaugmented_targets_exact=True,selected_loss_values_and_gradients_exact=True,
        warmup_unchanged=True,calibration_rule_exact=True,augmented_target_checks=12,
        spatial_augmentations=accepted,measured_extra_flops=checked,
        extra_flops_bound=profile['extra_update_flops_bound'])


def production_preflight(stage,pkl,output):
    """Run on the allocated GPU before any full capped condition starts."""
    import gc
    import train_compute_matched_baselines as trainer
    from .prepare import CAMPAIGN
    from .worker import check_frozen
    torch.set_num_threads(4)
    check_frozen(stage,pkl)
    source=stage/'source'
    p=json.loads((source/'experiments'/CAMPAIGN/'swinunetr/protocol.json').read_text())
    splits=json.loads((source/'experiments'/CAMPAIGN/'swinunetr/splits_10patients.json').read_text())
    dataframe=trainer._load_pkl_dataframe(pkl)
    targets_checked=0
    first=None
    for fold,split in enumerate(splits):
        patients=trainer.PreparedPatients(dataframe,split['train'])
        for index,item in enumerate(patients):
            if first is None:first=item
            geometry=p['native_geometry'][item['case_name']]
            rng=torch.Generator().manual_seed(2000+100*fold+index)
            saved=rng.get_state()
            image,y,stats=trainer.augment_mild(item['image'][None],item['label'][None],generator=rng)
            current=augmented_affine(geometry['model_affine'],saved,stats,y.shape[-3:])
            target=make_targets(y,geometry['model_affine'],current,geometry['native_shape'],p['depth_geometry'],'cpu')
            assert torch.isfinite(target['depth']).all() and abs(float(target['depth'].sum())-1)<1e-5
            assert all(len(group[side]['index'])>0 for group in target['pairs'] for side in ('inside','outside'))
            targets_checked+=1
    assert targets_checked==30
    tested=[]
    for identity,spec in MODELS.items():
        directory=source/'experiments'/CAMPAIGN/identity
        protocol=json.loads((directory/'protocol.json').read_text())
        architecture=json.loads((directory/'nnunet_architecture.json').read_text())
        trainer.seed_everything(0)
        model=build_model(identity,architecture).cuda().train()
        assert sum(q.numel() for q in model.parameters())==spec['parameters']
        criterion,optimizer=trainer.build_loss_and_optimizer(spec['alias'],model,protocol)
        native=criterion if identity=='nnunet' else None
        y=first['label'][None].cuda();image=first['image'][None].cuda()
        geom=protocol['native_geometry'][first['case_name']]
        target=make_targets(y,geom['model_affine'],geom['model_affine'],geom['native_shape'],protocol['depth_geometry'],'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(image)
        record=calibration_case(raw,y,ARM,criterion=native,depth_weights=target)
        weights,_=select_weights([record],ARM)
        # Native coarse-head derivatives must remain exactly their original values.
        if native is not None:
            detached=[r.detach().float().requires_grad_(True) for r in raw]
            a=training_loss(detached,y,ARM,weights,1.,criterion=native,depth_weights=target)
            b=old_loss(detached,y,'dice',{},1.,criterion=native)
            ag=torch.autograd.grad(a,detached[1:3],retain_graph=True)
            bg=torch.autograd.grad(b,detached[1:3])
            assert all(torch.equal(x,z) for x,z in zip(ag,bg))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=training_loss(raw,y,ARM,weights,1.,criterion=native,depth_weights=target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
        if native is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),12,foreach=False,error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(q).all() for q in model.parameters())
        tested.append(dict(model=identity,parameters=spec['parameters'],gpu_forward_backward_update=True))
        del model,optimizer,raw,loss,target
        gc.collect();torch.cuda.empty_cache()
    write_json(output,dict(status='passed',training_target_checks=targets_checked,models=tested,
        native_coarse_gradients_unchanged=True,calibration_training_only=True,
        note='Preflight uses training images only. These disposable model updates do not enter reported training runs.'))
    print(json.dumps(dict(gpu_preflight='passed',target_checks=targets_checked,models=4)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=cpu_checks();write_json(args.output,result);print(json.dumps(result),flush=True)
