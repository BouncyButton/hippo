"""Numerically check the exact ADNI mixed-loss recipe and frozen subject split."""
from copy import deepcopy
import subprocess

import numpy as np
import torch

from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, digest
from scripts.cross_dataset_campaign import adni_mixed_losses as obj
from scripts.cross_dataset_campaign import binary_ce_constraints as ce_reference
from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
from nnunet_band_objective import native_loss


def check(stage, plan, verify_data=True):
    torch.set_num_threads(2);torch.manual_seed(0);obj.binary.bind()
    x,y,z=torch.meshgrid(*(torch.arange(64),)*3,indexing='ij')
    labels=((((x-32)/18)**2+((y-32)/12)**2+((z-32)/10)**2)<1).long()[None,None]
    outputs=[torch.randn(1,2,n,n,n,requires_grad=True) for n in (64,32,16,8)]
    affine=np.eye(4);affine[:3,:3]=np.array([[0,1,0],[1,0,0],[0,0,1]])
    sdf=obj.binary.signed_distance_maps(labels)
    datasets=set()
    for model in MODELS:
        p=read(stage/'protocols'/model/'protocol.json')
        datasets.add((p['data'],p['pkl_sha256']))
        criterion,optimizer=obj.experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],torch.nn.Linear(1,1),p)
        native=criterion if model=='nnunet' else None
        logits=outputs if native is not None else outputs[0]
        if native is not None:
            if criterion.loss.dc.do_bg or criterion.loss.weight_dice!=1 or criterion.loss.weight_ce!=1:
                raise ValueError('Native foreground Dice+CE differs')
            if not np.allclose(criterion.weight_factors,[1/1.75,.5/1.75,.25/1.75,0]):
                raise ValueError('Native deep supervision differs')
            if type(optimizer).__name__!='SGD' or optimizer.param_groups[0]['momentum']!=.95:
                raise ValueError('Native SGD differs')
            baseline=native_loss(logits,labels,criterion)
        else:
            if type(optimizer).__name__!='AdamW' or not criterion.include_background:
                raise ValueError('Established Dice/AdamW differs')
            baseline=paired_loss(logits,labels,'dice',{},0.,weight_ce=0)
        if not torch.equal(baseline,obj.training_loss(logits,labels,'dice',{},0.,criterion=native)):
            raise ValueError('ADNI baseline graph differs')
        depth=obj.binary.geometry_weights(labels,affine,p['depth_geometry'])
        terms=dict(bands=obj.binary.bands()(outputs[0],labels).loss,
            edge=obj.binary.BinaryEdge()(outputs[0],labels),
            sdf=obj.binary.sdf_loss(outputs[0],labels,sdf),depth=obj.binary.depth_loss(outputs[0],labels,depth))
        gradients={'dice':torch.autograd.grad(baseline,outputs[0],retain_graph=True)[0]}
        gradients.update({k:torch.autograd.grad(v,outputs[0],retain_graph=True)[0] for k,v in terms.items()})
        for arm in ARMS[1:]:
            names=[n for n in ('bands','edge','sdf','depth') if n in arm]
            weights={n:.13 for n in names};policy=p['sdf_policy'][arm]
            delayed='sdf' in arm and policy=='delayed_5_of_9';scale=1. if delayed else .4
            actual=obj.training_loss(logits,labels,arm,weights,scale,sdf,native,depth,.2,policy=policy)
            expected=baseline+sum(weights[n]*scale*(.2 if delayed and n=='sdf' else 1.)*terms[n] for n in names)
            if not torch.allclose(actual,expected,atol=1e-6):raise ValueError('Auxiliary graph changed: '+arm)
            grads=torch.autograd.grad(actual,outputs[:3] if native is not None else outputs[:1],retain_graph=True)
            if any(not g.isfinite().all() for g in grads):raise ValueError('Nonfinite gradients')
            row=obj.calibration_case(logits,labels,arm,sdf,native,depth)
            if not np.isclose(row['dice_rms'],float(gradients['dice'].square().mean().sqrt()),rtol=1e-6):
                raise ValueError('Calibration uses the wrong baseline')
            if native is not None:
                if not torch.equal(actual,ce_reference.training_loss(logits,labels,arm,weights,scale,sdf,native,depth,.2,sdf_policy=policy)):
                    raise ValueError('Native Dice+CE profile graph differs')
                if row!=ce_reference.calibration_case(logits,labels,arm,sdf,native,depth):
                    raise ValueError('Native profile calibration differs')
            target,ceiling=p['calibration_target_ratio'],p['calibration_p95_cap']
            selected,stats=obj.experiment.select_at_target([deepcopy(row) for _ in range(10)],arm,target,ceiling)
            joint=gradients['dice'].clone()
            for name in names:
                ratio=float((selected[name]*gradients[name]).square().mean().sqrt()/joint.square().mean().sqrt())
                if not np.isclose(ratio,target,rtol=1e-5) or stats[name]['weighted_p95_ratio']>ceiling+1e-8:
                    raise ValueError('Sequential auxiliary calibration differs')
                joint=joint+selected[name]*gradients[name]
        print('Verified ADNI',model,p['calibration_base'],p['calibration_target_ratio'],flush=True)
    if len(datasets)!=1:raise ValueError('Models do not share one frozen dataset')
    data,sha=next(iter(datasets))
    if verify_data and digest(data)!=sha:raise ValueError('Actual ADNI data changed')
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    return ['56 fresh ADNI runs, four models by14 conditions, fold0 seed0, no augmentation',
        'Saved5% subject split: five training subjects/ten crops;27 validation subjects/54 crops; no subject overlap',
        'Native nnUNet foreground Dice+CE, original deep supervision and SGD',
        'Other models use established MONAI Dice including background and AdamW',
        'All13 auxiliary combinations, SDF timing and binary graph checked numerically',
        'Sequential60%/10% auxiliary-gradient targets and75%/50% p95 ceilings checked directly',
        'Existing measured profiles and original ADNI scientific source hashes verified; schedules recomputed',
        'Actual ADNI dataset hash verified' if verify_data else 'Dataset hash check deferred to cluster',
        'MSD completion/archive/cleanup submission gate; original24h resume and per-run archive policy',
        'Shell syntax passed']
