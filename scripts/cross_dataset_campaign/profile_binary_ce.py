"""Audit changed loss graphs, with independent full-model equality checks.

The network/optimizer graphs are unchanged. Replace only the measured loss and
logit-backward costs in existing full profiles. Independently remeasure complete
baseline and four-auxiliary graphs to require exact integer equality.
"""
import argparse
from copy import deepcopy
import fcntl
from functools import partial
from pathlib import Path

import numpy as np

from scripts.cross_dataset_campaign import binary_ce_constraints as new
from scripts.cross_dataset_campaign import conventional
from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT
from scripts.adni_constraints import objectives as old
from scripts.adni_constraints.audit_correction import correct_row
from profile_intermediate_baselines import audit, intermediate_supplement


def target_for(geometry,size):
    target=(audit.synthetic_target(geometry)>0).long().transpose(2,3).contiguous()
    if size!=64:
        target=audit.torch.nn.functional.interpolate(target.float(),size=(size,)*3,mode='nearest').long()
    return target


def component(identity,protocol,arm,geometry,size,calibration,modern):
    torch=audit.torch
    torch.manual_seed(0)
    leaves=[torch.randn(1,2,size//(2**i),size//(2**i),size//(2**i),requires_grad=True) for i in range(4 if identity=='nnunet' else 1)]
    outputs=[value.clone() for value in leaves]
    criterion,_=(conventional.ORIGINAL_LOSS_OPTIMIZER if modern or protocol['dataset']=='ADNI'
        else conventional.dice_only_optimizer)(protocol['trainer_alias'],torch.nn.Linear(1,1),protocol)
    native=criterion if identity=='nnunet' else None
    if native is None:outputs=outputs[0]
    target=target_for(geometry,size)
    sdf=old.signed_distance_maps(target) if 'sdf' in arm else None
    depth=old.geometry_weights(target,np.eye(4),old.GEOMETRY) if 'depth' in arm else None
    operator=audit.reference.OperatorAudit();stages={}
    with operator,audit.FlopCounterMode(display=False,custom_mapping={torch.ops.aten.convolution_backward:audit.reference.conv_backward_flops}) as counter:
        operator.stage='calibration_logit_gradients' if calibration else 'loss'
        before=counter.get_total_flops()
        if calibration:
            (new.calibration_case if modern else old.calibration_case)(outputs,target,arm,sdf,native,depth)
        else:
            fn=new.training_loss if modern else old.training_loss
            loss=fn(outputs,target,arm,dict(bands=.04,edge=.1,sdf=.1,depth=.1),1.,sdf,native,depth,.2)
        stages[operator.stage]=counter.get_total_flops()-before
        if not calibration:
            operator.stage='backward';before=counter.get_total_flops();loss.backward()
            stages['backward']=counter.get_total_flops()-before
    return correct_row(dict(flops=sum(stages.values())+sum(operator.supplement.values()),
        core_flops_per_stage=stages,supplementary_per_stage=operator.supplement,operator_audit=operator.ops))


def full_measure(dataset,identity,architecture,protocol,arm,geometry,calibration):
    # These profilers contain the same established model, optimizer and dense
    # operator audit; replace their loss bindings only in this isolated process.
    if dataset=='ADNI':
        from scripts.cross_dataset_campaign import profile_adni_nnunet as full
        full.dice_only_optimizer=conventional.ORIGINAL_LOSS_OPTIMIZER
    else:
        from scripts.cross_dataset_campaign import profile_seismic as full
        full.conventional.dice_only_optimizer=conventional.ORIGINAL_LOSS_OPTIMIZER
    saved_loss,saved_cal=full.obj.training_loss,full.obj.calibration_case
    full.obj.training_loss=new.training_loss;full.obj.calibration_case=new.calibration_case
    try:
        return full.measure(identity,architecture,protocol,arm,geometry,calibration)
    finally:
        full.obj.training_loss,full.obj.calibration_case=saved_loss,saved_cal
        new.bind()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['ADNI','GeoFWI3D'],required=True)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=args.output.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if args.output.exists():raise ValueError('Profile already exists')
    if audit.monai.__version__!='1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established local profiling runtime')
    audit.torch.set_num_threads(2);audit.reference.supplementary_flops=intermediate_supplement
    folder=ROOT/'experiments/adni_constraints_noaug_5pct_fold0_seed0_20261013/protocols'/args.model
    protocol=read(folder/'protocol.json');protocol['dataset']=args.dataset
    architecture=read(folder/'nnunet_architecture.json')
    reference=(folder/'profile.json' if args.dataset=='ADNI' else
        ROOT/'reports/cross_dataset_full_20261014/profiles'/f'geofwi3d_{args.model}.json')
    old_profile=read(reference);size=64 if args.dataset=='ADNI' else 96;count=10 if size==64 else 19
    rows=[];deltas=[];costs={}
    for arm in (*ARMS,'dice_edge','dice_edge_sdf'):
        for calibration in ([False] if arm=='dice' else [False,True]):
            for geometry in ('ellipsoid','box'):
                before=component(args.model,protocol,arm,geometry,size,calibration,False)
                after=component(args.model,protocol,arm,geometry,size,calibration,True)
                ref=next(r for r in old_profile['profiles'] if (r['arm'],r['geometry'],r['calibration'])==(arm,geometry,calibration))
                value=ref['flops']+after['flops']-before['flops']
                rows.append(dict(model=args.model,arm=arm,geometry=geometry,calibration=calibration,flops=value,parameters=ref['parameters']))
                deltas.append(dict(arm=arm,geometry=geometry,calibration=calibration,reference_flops=ref['flops'],old_component=before,new_component=after))
        train={r['flops'] for r in rows if r['arm']==arm and not r['calibration']}
        cal={r['flops'] for r in rows if r['arm']==arm and r['calibration']}
        if len(train)!=1 or arm!='dice' and len(cal)!=1:raise ValueError('Geometry-dependent arithmetic')
        costs[arm]=dict(update_flops=train.pop(),calibration_flops=count*cal.pop()+(4096 if 'depth' in arm else 1024) if cal else 0)
        write(args.output.with_suffix('.partial.json'),dict(costs=costs,profiles=rows))
        print(args.dataset,args.model,arm,'component audit complete',flush=True)
    full_checks=[]
    for arm in ('dice','dice_bands_edge_sdf_depth'):
        for calibration in ([False] if arm=='dice' else [False,True]):
            for geometry in ('ellipsoid','box'):
                direct=full_measure(args.dataset,args.model,architecture,protocol,arm,geometry,calibration)
                predicted=next(r['flops'] for r in rows if (r['arm'],r['geometry'],r['calibration'])==(arm,geometry,calibration))
                full_checks.append(direct)
                if direct['flops']!=predicted:
                    raise ValueError(f'Full-model/component mismatch: {arm} {calibration}: {direct["flops"]} != {predicted}')
                print(args.dataset,args.model,arm,calibration,geometry,'exact full-model equality',flush=True)
    write(args.output,dict(status='profiled',model=args.model,dataset=args.dataset,loss_recipe='dice_ce',
        costs=costs,profiles=rows,component_audits=deltas,full_model_checks=full_checks,parameters=rows[0]['parameters'],
        input_shape=[1,1,size,size,size],calibration_base='Dice+CE',reference_profile=str(reference.relative_to(ROOT)),
        reference_profile_sha256=digest(reference),objective_sha256=digest(Path(new.__file__)),
        shared_objective_sha256=digest(Path(new.objective.__file__)),
        accounting='Replace measured loss/logit-backward graphs; network and optimizer unchanged. Exact independent full-model equality required for baseline and all-four-constraint train/calibration graphs on two geometries. MAC=2; calibration separate.',
        runtime=dict(torch=audit.torch.__version__,monai=audit.monai.__version__)))


if __name__=='__main__':main()
