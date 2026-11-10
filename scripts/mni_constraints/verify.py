"""Numerical and real-cohort checks for the MNI constraint adaptation."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.mni_constraints import objectives as o
from scripts.cross_dataset_campaign.common import ARMS, read
from scripts.unetr_subfields import baseline_runtime, geo_recipe, core
from scripts.unetr_subfields.contracts import DATASETS
from scripts.depth_supervision.geometry import build_geometry, GeometryConfig


def numeric_checks(settings):
    torch.set_num_threads(4);torch.manual_seed(731)
    label=torch.zeros((1,1,64,64,64),dtype=torch.long)
    label[:,:,24:40,16:29,26:38]=1
    label[:,:,24:40,29:36,26:38]=2
    label[:,:,24:40,36:48,26:38]=3
    sdf=o.signed_distance_maps(label)
    assert sdf.shape==(1,3,64,64,64) and torch.isfinite(sdf).all()
    empty=o.signed_distance_maps(torch.zeros_like(label));assert (empty==1).all()
    for c in (1,2,3):
        assert (sdf[:,c-1][label[:,0]==c]<0).all()
    depth=o.geometry_weights(label,np.eye(4),settings)
    reference=build_geometry((label[0,0].numpy()>0).astype(np.uint8),np.eye(4),GeometryConfig(**settings),fields=False)['weights']
    assert np.array_equal(depth.numpy(),reference)
    assert torch.isfinite(depth).all() and depth.min()>=0 and abs(float(depth.sum())-1)<1e-5
    # Every Geo symmetry must give exactly the same paired permutation to weights.
    found=set()
    for update in range(128):
        index=geo_recipe.transform_id(0,update)
        if index in found:continue
        found.add(index)
        _,transformed=geo_recipe.apply_transform(depth[None,None],depth[None,None],index)
        assert torch.equal(o.transform_depth(depth,0,update),transformed[0,0])
        _,newlabel=geo_recipe.apply_transform(label.float(),label,index)
        mapped_sdf=torch.cat([geo_recipe.apply_transform(sdf[:,c:c+1],label,index)[0] for c in range(3)],1)
        assert torch.equal(o.signed_distance_maps(newlabel),mapped_sdf)
    assert found==set(range(8))
    evidence=[]
    for identity in ('nnunet','swinunetr','unetr_vit_b16','basic_unet16'):
        criterion=None
        if identity=='nnunet':
            p=dict(initial_lrs={'nnunet':.01},sgd_weight_decay=3e-5,nnunet_sgd_momentum=.95)
            criterion,_=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER('nnunet',torch.nn.Linear(1,1),p)
            outputs=[torch.randn((1,4,size,size,size),requires_grad=True) for size in (64,32,16,8)]
        else:outputs=torch.randn((1,4,64,64,64),requires_grad=True)
        z=outputs[0] if isinstance(outputs,list) else outputs
        base=o.base_loss(outputs,label,identity,criterion)
        expected=torch.autograd.grad(base,z,retain_graph=True)[0]
        full=o.training_loss(outputs,label,ARMS[-1],dict.fromkeys(o.TERMS,0.),1.,sdf,criterion,depth,.2,identity=identity)
        actual=torch.autograd.grad(full,z)[0]
        assert torch.equal(base.detach(),full.detach()) and torch.equal(expected,actual),identity
        for arm in ARMS[1:]:
            row=o.calibration_case(outputs,label,arm,sdf,criterion,depth,identity=identity)
            weights,statistics=o.select_weights([row],arm)
            assert set(weights)=={n for n in o.TERMS if n in arm}
            assert all(v>0 and np.isfinite(v) for v in weights.values())
            assert all(s['weighted_median_ratio']<=.1000000001 and s['weighted_p95_ratio']<=.5000000001 for s in statistics.values())
            loss=o.training_loss(outputs,label,arm,weights,1.,sdf,criterion,depth,.2,identity=identity)
            gradient=torch.autograd.grad(loss,z)[0]
            assert torch.isfinite(loss) and torch.isfinite(gradient).all() and gradient.abs().sum()>0
        evidence.append(dict(model=identity,base_value_and_gradient_exact=True,arms_checked=13))
    return dict(numerical=evidence,geo_symmetries_checked=8,sdf_foreground_classes=3)


def check_stage(stage):
    import monai
    assert monai.__version__=='1.6.0'
    folder=stage/'protocols/nnunet_dice_bands_edge_sdf_depth'
    p=read(folder/'protocol.json')
    checks=numeric_checks(p['depth_geometry'])
    splits=read(folder/'splits_10patients.json')
    names=sorted({c for split in splits for c in split['train']})
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    prepared=core.PreparedSubfields(frame,names,DATASETS['MNI'])
    geometries=[]
    for patient in prepared:
        label=patient['label'][None]
        assert label.shape==(1,1,64,64,64) and set(torch.unique(label).tolist())=={0,1,2,3}
        weights=o.geometry_weights(label,p['native_geometry'][patient['case_name']]['model_affine'],p['depth_geometry'])
        assert torch.isfinite(weights).all() and weights.min()>=0 and abs(float(weights.sum())-1)<1e-5
        geometries.append(dict(case=patient['case_name'],weight_sum=float(weights.sum()),foreground=int((label>0).sum())))
    models=[]
    for identity in ('nnunet','swinunetr','unetr_vit_b16','basic_unet16'):
        folder=stage/'protocols'/(identity+'_dice_bands_edge_sdf_depth')
        p=read(folder/'protocol.json');profile=read(folder/'profile.json')
        model=(geo_recipe.build_model(4,(64,64,64)) if identity=='unetr_vit_b16' else baseline_runtime.build_model(identity,read(folder/'nnunet_architecture.json'))).to('meta').train()
        outputs=model(torch.zeros((1,1,64,64,64),device='meta'))
        shapes=[list(v.shape) for v in outputs] if isinstance(outputs,(tuple,list)) else [list(outputs.shape)]
        assert shapes==profile['output_shapes']
        assert sum(v.numel() for v in model.parameters())==profile['parameters']
        models.append(dict(model=identity,output_shapes=shapes,parameters=profile['parameters']))
        del model,outputs
    return dict(**checks,real_training_geometries=geometries,models=models,optimizer_updates=0,
                runtime=dict(torch=torch.__version__,monai=monai.__version__))


if __name__=='__main__':
    from scripts.unetr_subfields.contracts import ROOT
    settings=read(ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json')['geometry']
    print(json.dumps(numeric_checks(settings)),flush=True)
