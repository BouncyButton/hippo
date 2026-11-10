"""Check frozen arithmetic, LR prefixes, architecture, and native constraint gradients."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path

import monai
import torch
from campaign90 import ARMS, BUDGET, CAMPAIGN, MODELS, ROOT, build_model, digest, validation_updates, write_json
from compute_constraint_objectives import scheduled_costs, select_weights
import train_compute_matched_baselines as base
import nnunet_multi_objective as native
import nnunet_band_objective as previous
from thesis.new_constraints.signed_distance import signed_distance_maps


def semantic_checks(criterion):
    coords = torch.arange(64)
    x,y,z = torch.meshgrid(coords,coords,coords,indexing='ij')
    foreground = (x>=20)&(x<44)&(y>=22)&(y<42)&(z>=25)&(z<39)
    labels = (foreground.long()*(1+(x>=32).long()))[None,None]
    sdf = signed_distance_maps(labels)
    outputs = [torch.randn(1,3,n,n,n,requires_grad=True) for n in [64,32,16,8]]
    weights = {'bands':.04,'edge':.1,'sdf':.1}
    old = previous.training_loss(outputs,labels,'dice_bands',weights,.2,criterion)
    new = native.training_loss(outputs,labels,'dice_bands',weights,.2,criterion)
    if not torch.equal(old,new):
        raise ValueError('Existing bands objective changed')
    old_grad = torch.autograd.grad(old,outputs[:3],retain_graph=True)
    new_grad = torch.autograd.grad(new,outputs[:3],retain_graph=True)
    if any(not torch.equal(a,b) for a,b in zip(old_grad,new_grad)):
        raise ValueError('Existing bands gradients changed')
    old_cal = previous.calibration_case(outputs,labels,'dice_bands',criterion)
    new_cal = native.calibration_case(outputs,labels,'dice_bands',criterion)
    if old_cal != new_cal:
        raise ValueError('Existing bands calibration changed')
    baseline = native.native_loss(outputs,labels,criterion)
    base_grad = torch.autograd.grad(baseline,outputs[:3],retain_graph=True)
    for arm in ARMS[1:]:
        value = native.training_loss(outputs,labels,arm,weights,.2,criterion,sdf)
        gradients = torch.autograd.grad(value,outputs[:3],retain_graph=True)
        if not torch.isfinite(value) or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite auxiliary gradients')
        if any(not torch.equal(a,b) for a,b in zip(base_grad[1:],gradients[1:])):
            raise ValueError('Constraints changed lower-resolution native heads')
        rows = [native.calibration_case(outputs,labels,arm,criterion,sdf) for _ in range(2)]
        selected,stats = select_weights(rows,arm)
        if set(selected) != {s for s in ['bands','edge','sdf'] if s in arm}:
            raise ValueError('Missing calibrated weights')
        for s in stats.values():
            if not 0<s['weighted_median_ratio']<=.1+1e-8 or not 0<s['weighted_p95_ratio']<=.5+1e-8:
                raise ValueError('Gradient ratio caps failed')
    return {'existing_bands_loss_gradient_calibration_identical':True,
            'all_auxiliary_gradients_finite':True,'lower_resolution_native_heads_unchanged':True,
            'all_weight_calibrations_passed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--pkl',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    source=args.stage/'source'
    manifest=json.loads((args.stage/'source_manifest.json').read_text())
    for name,sha in manifest.items():
        if digest(source/name)!=sha:raise ValueError('Frozen source changed: '+name)
    directory=source/'experiments'/CAMPAIGN
    plan=json.loads((directory/'plan.json').read_text())
    if (plan['budget_flops_per_run']!=BUDGET or plan['runs']!=24 or (plan['fold'],plan['seed'])!=(0,0)
            or plan['model_order']!=list(MODELS) or plan['arm_order']!=list(ARMS)
            or monai.__version__!='1.6.0' or digest(args.pkl)!=plan['pkl_sha256']):
        raise ValueError('Campaign identity, data, or runtime changed')
    if torch.__version__!='2.12.1+cu130' or not importlib.metadata.version('dynamic-network-architectures').startswith('0.4.'):
        raise ValueError('Expected previously validated cluster runtime')
    smoke_path=args.stage/'cluster_resume_smoke.json'
    smoke=json.loads(smoke_path.read_text())
    if (smoke['status']!='passed' or smoke['torch']!=torch.__version__
            or not smoke['resumed_model_exactly_matches_uninterrupted']):
        raise ValueError('Missing cluster interruption/resume certificate')
    for name,sha in smoke['source_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Smoke-tested source changed')
    checks=[]
    for identity,spec in MODELS.items():
        model_dir=directory/identity
        protocol=json.loads((model_dir/'protocol.json').read_text())
        if digest(model_dir/'protocol.json')!=plan['protocol_sha256'][identity]:raise ValueError('Protocol changed')
        for key,filename in [('splits_sha256','splits_10patients.json'),('nnunet_architecture_sha256','nnunet_architecture.json')]:
            if digest(model_dir/filename)!=protocol[key]:raise ValueError('Frozen input changed')
        split=json.loads((model_dir/'splits_10patients.json').read_text())[0]
        if len(set(split['train']))!=10 or len(set(split['val']))!=52 or set(split['train'])&set(split['val']):
            raise ValueError('Invalid train/validation split')
        profile_path=source/protocol['profile_evidence']['file']
        if digest(profile_path)!=protocol['profile_evidence']['sha256']:raise ValueError('Profile changed')
        profile=json.loads(profile_path.read_text())
        for name,sha in profile['source_sha256'].items():
            if digest(source/name)!=sha:raise ValueError('Profile source changed')
        for name,sha in protocol['training_source_sha256'].items():
            if digest(source/name)!=sha:raise ValueError('Runtime source changed')
        for arm in ARMS:
            measured=profile['costs'][arm]
            expected=scheduled_costs(BUDGET,spec['baseline_flops'],measured['update_flops'],measured['calibration_flops'],50)
            schedule=protocol['models'][spec['alias']]['objectives'][arm]
            if any(schedule[k]!=v for k,v in expected.items()):raise ValueError('Budget arithmetic changed')
            milestones=validation_updates(protocol,arm)
            if max(milestones)!=schedule['max_steps'] or min(milestones)<1:raise ValueError('Invalid checkpoint schedule')
            from train_campaign90 import configure
            import train_campaign90_core as trainer
            original_lr,original_factory,original_validation=trainer.base.learning_rate,trainer.base.build_model,trainer.base.validation_steps
            configure(protocol,arm)
            for step in range(schedule['max_steps']):
                actual=trainer.base.learning_rate(spec['alias'],step,schedule['max_steps'],spec['initial_lr'])
                expected_lr=original_lr(spec['alias'],step,spec['lr_horizon'],spec['initial_lr'],0.0)
                if actual!=expected_lr:raise ValueError('LR prefix changed')
            trainer.base.learning_rate,trainer.base.build_model,trainer.base.validation_steps=original_lr,original_factory,original_validation
        architecture=json.loads((model_dir/'nnunet_architecture.json').read_text())
        base.seed_everything(0)
        model=build_model(identity,architecture)
        criterion,optimizer=base.build_loss_and_optimizer(spec['alias'],model,protocol)
        if sum(p.numel() for p in model.parameters())!=spec['parameters']:raise ValueError('Architecture changed')
        model.eval()
        with torch.no_grad():outputs=model(torch.randn(1,1,64,64,64))
        logits=outputs[0] if isinstance(outputs,(list,tuple)) else outputs
        if tuple(logits.shape)!=(1,3,64,64,64) or not torch.isfinite(logits).all():raise ValueError('Invalid model output')
        row={'model':identity,'parameters':spec['parameters'],'architecture_forward_passed':True,
             'all_objective_schedules_verified':True,'all_lr_values_match_baseline_horizon':True}
        if identity=='nnunet':
            if optimizer.param_groups[0]['momentum']!=.95:raise ValueError('Momentum changed')
            row.update(semantic_checks(criterion))
        checks.append(row)
        del model,criterion,optimizer,outputs,logits
        gc.collect()
        print(json.dumps(row),flush=True)
    write_json(args.stage/'preflight.json',{'status':'passed','checks':checks,'torch':torch.__version__,
               'monai':monai.__version__,'plan_sha256':digest(directory/'plan.json'),
               'source_manifest_sha256':digest(args.stage/'source_manifest.json'),
               'cluster_resume_smoke_sha256':digest(smoke_path),
               'budget_flops_per_run':BUDGET,'models':4,'arms':6,'total_runs':24})


if __name__=='__main__':
    main()
