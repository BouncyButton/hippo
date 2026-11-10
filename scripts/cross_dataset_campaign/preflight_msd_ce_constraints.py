"""Numerical checks of CE, original auxiliaries and sequential calibration."""
from copy import deepcopy
import numpy as np
import torch

from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, digest
from scripts.cross_dataset_campaign import msd_ce_constraints as obj
from scripts.cross_dataset_campaign import probe_unet_constraints as previous_unet
from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
from nnunet_band_objective import native_loss
from thesis.new_constraints.signed_distance import signed_distance_maps


def check(stage, plan, verify_data=True):
    torch.set_num_threads(2); torch.manual_seed(0)
    axis = torch.arange(64)
    x,y,z = torch.meshgrid(axis,axis,axis,indexing='ij')
    fg = ((x-32)/18)**2 + ((y-32)/12)**2 + ((z-32)/10)**2 < 1
    labels = torch.where(fg,torch.where(x<32,1,2),0)[None,None]
    outputs = [torch.randn(1,3,n,n,n,requires_grad=True) for n in (64,32,16,8)]
    affine = np.eye(4); affine[:3,:3] = np.array([[0,1,0],[1,0,0],[0,0,1]])
    sdf = signed_distance_maps(labels)
    datasets, splits = set(), set()
    for identity in MODELS:
        folder = stage/'protocols'/identity
        p = read(folder/'protocol.json')
        for name,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
            if digest(folder/name) != p[key]:
                raise ValueError('Protocol artifact changed: '+name)
        datasets.add((p['data'],p['pkl_sha256'])); splits.add(p['splits_sha256'])
        split = read(folder/'splits_10patients.json')[0]
        if len(split['train']) != 10 or len(split['val']) != 52 or set(split['train']) & set(split['val']):
            raise ValueError('Frozen 10/52 split changed')
        criterion,optimizer = obj.experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],torch.nn.Linear(1,1),p)
        native = criterion if identity == 'nnunet' else None
        logits = outputs if native is not None else outputs[0]
        if native is not None:
            if criterion.loss.dc.do_bg or not np.allclose(criterion.weight_factors,[1/1.75,.5/1.75,.25/1.75,0]):
                raise ValueError('Native Dice/deep supervision changed')
            if type(optimizer).__name__ != 'SGD' or optimizer.param_groups[0]['momentum'] != .95:
                raise ValueError('Native optimizer changed')
            baseline = native_loss(logits,labels,criterion)
        else:
            if not criterion.include_background or type(optimizer).__name__ != 'AdamW':
                raise ValueError('Established MONAI Dice/optimizer changed')
            baseline = paired_loss(logits,labels,'dice',{},0.,weight_ce=1)
        if not torch.equal(baseline,obj.training_loss(logits,labels,'dice',{},0.,criterion=native)):
            raise ValueError('Dice+CE baseline changed')
        depth = obj.original.geometry_weights(labels,affine,p['depth_geometry'])
        terms = dict(bands=obj.original.OuterBoundaryBandLoss()(outputs[0],labels).loss,
            edge=obj.original.BoundaryEdgeConsistencyLoss()(outputs[0],labels),
            sdf=obj.original.signed_distance_loss(outputs[0],labels,sdf),
            depth=obj.original.depth_loss(outputs[0],labels,depth))
        grads = {'dice':torch.autograd.grad(baseline,outputs[0],retain_graph=True)[0]}
        grads.update({name:torch.autograd.grad(loss,outputs[0],retain_graph=True)[0] for name,loss in terms.items()})
        for arm in ARMS[1:]:
            used = [name for name in ('bands','edge','sdf','depth') if name in arm]
            weights = {name:.13 for name in used}
            policy = p['sdf_policy'][arm]
            delayed = 'sdf' in arm and policy == 'delayed_5_of_9'
            scale = 1. if delayed else .4
            actual = obj.training_loss(logits,labels,arm,weights,scale,sdf,native,depth,.2,sdf_policy=policy)
            expected = baseline + sum(weights[name]*scale*(.2 if delayed and name=='sdf' else 1.)*terms[name] for name in used)
            if not torch.allclose(actual,expected,atol=1e-6):
                raise ValueError('Auxiliary loss or SDF ramp differs: '+identity+' '+arm)
            gradients = torch.autograd.grad(actual,outputs[:3] if native is not None else outputs[:1],retain_graph=True)
            if any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Nonfinite training gradients')
            row = obj.calibration_case(logits,labels,arm,sdf,native,depth)
            if not np.isclose(row['dice_rms'],float(grads['dice'].square().mean().sqrt()),rtol=1e-6):
                raise ValueError('Calibration denominator is not joint Dice+CE')
            selected,stats = obj.experiment.select_at_target([deepcopy(row) for _ in range(10)],arm,.6,.75)
            joint = grads['dice'].clone()
            for name in used:
                ratio = float((selected[name]*grads[name]).square().mean().sqrt()/joint.square().mean().sqrt())
                if not np.isclose(ratio,.6,rtol=1e-5) or stats[name]['weighted_p95_ratio'] > .75+1e-8:
                    raise ValueError('Sequential calibration or p95 ceiling differs')
                joint = joint + selected[name]*grads[name]
            if identity == 'basic_unet16' and arm in previous_unet.ARMS[1:]:
                reference = previous_unet.training_loss(outputs[0],labels,arm,weights,scale,depth=depth)
                if not torch.equal(actual,reference):
                    raise ValueError('Previously tested 3D UNet loss changed')
                if row != previous_unet.calibration_case(outputs[0],labels,arm,depth=depth):
                    raise ValueError('Previously tested 3D UNet calibration changed')
        print('Verified',identity,flush=True)
    if len(datasets) != 1 or len(splits) != 1:
        raise ValueError('Models must share data and splits')
    data,sha = next(iter(datasets))
    # Check the actual cluster pickle. A local pickle may have a different
    # serialization and is not a substitute for this provenance check.
    if verify_data and digest(data) != sha:
        raise ValueError('MSD dataset changed')
    return ['56 fresh runs; four models and all 14 screenshot conditions',
        'Same 10 training/52 validation crops, fold0 seed0, no augmentation',
        'Exact per-model Dice+CE baselines, deep supervision and optimizers preserved',
        'All 13 auxiliary combinations and original SDF timing checked numerically',
        'Sequential 60% gradient targets and 75% p95 ceiling checked against direct gradients',
        'Previous 3D UNet bands and bands/edge/depth graphs unchanged',
        'Remeasured FLOP schedules and frozen source/data hashes verified', 'Shell syntax']
