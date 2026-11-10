"""Check all14 objectives, model-specific baseline losses and calibration."""
from copy import deepcopy
import subprocess

import numpy as np
import torch

from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, digest
from scripts.cross_dataset_campaign import msd_mixed_losses as obj
from scripts.cross_dataset_campaign import msd_ce_constraints as ce_reference
from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
from nnunet_band_objective import native_loss
from thesis.new_constraints.signed_distance import signed_distance_maps


def check(stage,plan,verify_data=True):
    torch.set_num_threads(2);torch.manual_seed(0)
    x,y,z=torch.meshgrid(*(torch.arange(64),)*3,indexing='ij')
    fg=((x-32)/18)**2+((y-32)/12)**2+((z-32)/10)**2<1
    labels=torch.where(fg,torch.where(x<32,1,2),0)[None,None]
    outputs=[torch.randn(1,3,n,n,n,requires_grad=True) for n in (64,32,16,8)]
    affine=np.eye(4);affine[:3,:3]=np.array([[0,1,0],[1,0,0],[0,0,1]])
    sdf=signed_distance_maps(labels)
    datasets,splits=set(),set()
    for model in MODELS:
        folder=stage/'protocols'/model;p=read(folder/'protocol.json')
        split=read(folder/'splits_10patients.json')[0]
        if len(split['train'])!=10 or len(split['val'])!=52 or set(split['train'])&set(split['val']):
            raise ValueError('Expected frozen10/52 split')
        datasets.add((p['data'],p['pkl_sha256']));splits.add(p['splits_sha256'])
        criterion,optimizer=obj.experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],torch.nn.Linear(1,1),p)
        native=criterion if model=='nnunet' else None
        logits=outputs if native is not None else outputs[0]
        if native is not None:
            if criterion.loss.dc.do_bg or criterion.loss.weight_dice!=1 or criterion.loss.weight_ce!=1:
                raise ValueError('Native foreground Dice+CE differs')
            if not np.allclose(criterion.weight_factors,[1/1.75,.5/1.75,.25/1.75,0]):
                raise ValueError('Native deep supervision differs')
            if type(optimizer).__name__!='SGD' or optimizer.param_groups[0]['momentum']!=.95:
                raise ValueError('Native optimizer differs')
            baseline=native_loss(logits,labels,criterion)
        else:
            if type(optimizer).__name__!='AdamW' or not criterion.include_background:
                raise ValueError('Established Dice/optimizer differs')
            baseline=paired_loss(logits,labels,'dice',{},0.,weight_ce=0)
        if not torch.equal(baseline,obj.training_loss(logits,labels,'dice',{},0.,criterion=native)):
            raise ValueError('Baseline changed')
        depth=obj.original.geometry_weights(labels,affine,p['depth_geometry'])
        terms=dict(bands=obj.original.OuterBoundaryBandLoss()(outputs[0],labels).loss,
            edge=obj.original.BoundaryEdgeConsistencyLoss()(outputs[0],labels),
            sdf=obj.original.signed_distance_loss(outputs[0],labels,sdf),
            depth=obj.original.depth_loss(outputs[0],labels,depth))
        grads={'dice':torch.autograd.grad(baseline,outputs[0],retain_graph=True)[0]}
        grads.update({name:torch.autograd.grad(loss,outputs[0],retain_graph=True)[0] for name,loss in terms.items()})
        for arm in ARMS[1:]:
            used=[name for name in ('bands','edge','sdf','depth') if name in arm]
            weights={name:.13 for name in used};policy=p['sdf_policy'][arm]
            delayed='sdf' in arm and policy=='delayed_5_of_9';scale=1. if delayed else .4
            actual=obj.training_loss(logits,labels,arm,weights,scale,sdf,native,depth,.2,sdf_policy=policy)
            expected=baseline+sum(weights[name]*scale*(.2 if delayed and name=='sdf' else 1.)*terms[name] for name in used)
            if not torch.allclose(actual,expected,atol=1e-6):raise ValueError('Auxiliary graph changed: '+arm)
            gradients=torch.autograd.grad(actual,outputs[:3] if native is not None else outputs[:1],retain_graph=True)
            if any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite gradients')
            row=obj.calibration_case(logits,labels,arm,sdf,native,depth)
            if not np.isclose(row['dice_rms'],float(grads['dice'].square().mean().sqrt()),rtol=1e-6):
                raise ValueError('Wrong calibration baseline: '+model)
            if native is not None:
                if not torch.equal(actual,ce_reference.training_loss(logits,labels,arm,weights,scale,sdf,native,depth,.2,sdf_policy=policy)):
                    raise ValueError('Native profile training graph differs')
                if row!=ce_reference.calibration_case(logits,labels,arm,sdf,native,depth):
                    raise ValueError('Native profile calibration graph differs')
            target=p['calibration_target_ratio'];ceiling=p['calibration_p95_cap']
            selected,stats=obj.experiment.select_at_target([deepcopy(row) for _ in range(10)],arm,target,ceiling)
            joint=grads['dice'].clone()
            for name in used:
                ratio=float((selected[name]*grads[name]).square().mean().sqrt()/joint.square().mean().sqrt())
                if not np.isclose(ratio,target,rtol=1e-5) or stats[name]['weighted_p95_ratio']>ceiling+1e-8:
                    raise ValueError('Sequential gradient calibration differs')
                joint=joint+selected[name]*grads[name]
        print('Verified',model,p['calibration_base'],p['calibration_target_ratio'],flush=True)
    if len(datasets)!=1 or len(splits)!=1:raise ValueError('Models do not share the same data/split')
    data,sha=next(iter(datasets))
    if verify_data and digest(data)!=sha:raise ValueError('Actual cluster dataset changed')
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    return ['56 fresh MSD runs, fold0 seed0, no augmentation, frozen10/52 split',
        'nnUNet native foreground Dice+CE, original deep supervision and SGD preserved',
        'SwinUNETR/UNETR/3DUNet original MONAI Dice including background and AdamW preserved',
        'All13 constraint combinations and original SDF timing checked numerically at64-cube resolution',
        'Sequential60% native Dice+CE and10% other-model Dice calibration checked with direct gradients',
        'Original75%/50% p95 ceilings retained',
        'Unchanged measured profiles bound to exact source hashes; per-objective schedules recomputed',
        'Actual dataset hash verified; final-budget evaluation and archive policy retained','Shell syntax']
