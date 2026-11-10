"""Bounded GPU benchmark: exact training/evaluation/resume equivalence."""
import argparse
from copy import deepcopy
from functools import partial
import hashlib
from pathlib import Path
import time

import torch

from scripts.cross_dataset_campaign import msd_mixed_losses as scientific
from scripts.cross_dataset_campaign.common import MODELS,read,write,digest
from scripts.cross_dataset_campaign.msd_runtime_fast import RuntimeAcceleration,evaluate as fast_evaluate


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return deepcopy(value)


def fingerprint(value):
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
        elif isinstance(x,(tuple,list)):
            for item in x:walk(item)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()


def timed(function):
    torch.cuda.synchronize();start=time.perf_counter();value=function();torch.cuda.synchronize()
    return value,time.perf_counter()-start


def check_model(stage,output,identity):
    experiment=scientific.experiment;base=experiment.base;trainer=experiment.trainer
    p=read(stage/'protocols'/identity/'protocol.json');arm='dice_bands_edge_sdf_depth'
    print(dict(phase='start_model',model=identity,time=time.time()),flush=True)
    experiment.configure(p,arm);base.build_loss_and_optimizer=experiment.ORIGINAL_LOSS_OPTIMIZER
    trainer.training_loss=partial(scientific.training_loss,sdf_policy=p['sdf_policy'][arm])
    trainer.calibration_case=scientific.calibration_case
    split=read(stage/'protocols'/identity/'splits_10patients.json')[0]
    architecture=read(stage/'protocols'/identity/'nnunet_architecture.json')
    base.seed_everything(0);model=base.build_model(p['trainer_alias'],architecture).cuda()
    initial=cpu(model.state_dict());data_class=base.PreparedPatients
    dataframe=base._load_pkl_dataframe(Path(p['data']))
    original_train=data_class(dataframe,split['train']);original_val=data_class(dataframe,split['val'])
    weights={k:.13 for k in ('bands','edge','sdf','depth')}
    result=dict(model=identity,torch=torch.__version__,device=torch.cuda.get_device_name(),variants={})
    # Warm deterministic CUDA kernels once; warmup is excluded from comparison.
    criterion,_=base.build_loss_and_optimizer(p['trainer_alias'],model,p)
    model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        raw=model(original_train[0]['image'][None].cuda())
        loss=scientific.training_loss(raw,original_train[0]['label'][None].cuda(),'dice',{},1.,
            criterion=criterion if identity=='nnunet' else None)
    loss.backward();model.zero_grad(set_to_none=True);del raw,loss
    for variant in ('reference','cached'):
        acceleration=RuntimeAcceleration(experiment,p,'cuda') if variant=='cached' else None
        if acceleration:acceleration.install()
        try:
            if acceleration:
                # MONAI normalization can mutate the input arrays in memory.
                # Each production run loads fresh data; the benchmark must too.
                fresh=base._load_pkl_dataframe(Path(p['data']))
                train=base.PreparedPatients(fresh,split['train']);validation=base.PreparedPatients(fresh,split['val'])
                for old,new in zip(list(original_train)+list(original_val),list(train)+list(validation)):
                    for key in ('image','label'):
                        if not torch.equal(old[key],new[key].cpu()):raise ValueError('Benchmark input differs: '+old['case_name'])
            else:train,validation=original_train,original_val
            model.load_state_dict(initial)
            criterion,optimizer=base.build_loss_and_optimizer(p['trainer_alias'],model,p)
            native=criterion if identity=='nnunet' else None
            base.seed_everything(123)
            losses=[];steps=[];target_seconds=[];resume=None
            for index in (0,1,2,0,1,2):
                batch=train[index];label=batch['label'][None]
                def targets():
                    return (trainer.signed_distance_maps(label).cuda(),
                        trainer.geometry_weights(label,p['native_geometry'][batch['case_name']]['model_affine'],p['depth_geometry']).cuda())
                (sdf,depth),seconds=timed(targets);target_seconds.append(seconds)
                def update():
                    model.train();optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        outputs=model(batch['image'][None].cuda())
                        loss=trainer.training_loss(outputs,label.cuda(),arm,weights,1.,sdf,native,depth,.5)
                    loss.backward()
                    if native is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),12,foreach=False,error_if_nonfinite=True)
                    gradient_sha=fingerprint({k:v.grad for k,v in model.named_parameters()})
                    optimizer.step()
                    return float(loss.detach()),gradient_sha
                (loss,gradient_sha),seconds=timed(update)
                losses.append(loss);steps.append(dict(seconds=seconds,gradient_sha256=gradient_sha))
                if len(losses)==3:
                    resume=dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            final_hash=fingerprint(model.state_dict());optimizer_hash=fingerprint(optimizer.state_dict())
            after_training_rng=fingerprint((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
            calibration,calibration_seconds=timed(lambda:trainer.calibrate(model,[train[0],train[1]],arm,
                torch.device('cuda'),True,native,protocol=p))
            evaluation_rng_before=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            evaluation,evaluation_seconds=timed(lambda:base.evaluate(model,validation,torch.device('cuda'),True))
            evaluation_rng=fingerprint((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
            batch_trials=[]
            if acceleration:
                for size in (2,4,8):
                    torch.set_rng_state(evaluation_rng_before[0]);torch.cuda.set_rng_state_all(evaluation_rng_before[1])
                    candidate,seconds=timed(lambda:fast_evaluate(model,validation,torch.device('cuda'),True,p['classes'],size))
                    batch_trials.append(dict(batch_size=size,seconds=seconds,exact_metrics=candidate==evaluation,
                        exact_rng=fingerprint((torch.get_rng_state(),torch.cuda.get_rng_state_all()))==evaluation_rng))
            # Verify exact three-step continuation with the same optimizer/RNG.
            model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
            torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
            resumed_losses=[]
            for index in (0,1,2):
                batch=train[index];label=batch['label'][None];sdf,depth=targets()
                resumed_losses.append(update()[0])
            if (resumed_losses!=losses[3:] or fingerprint(model.state_dict())!=final_hash
                    or fingerprint(optimizer.state_dict())!=optimizer_hash):
                raise ValueError('Resume differs: '+identity+'/'+variant)
            # Best snapshots are immutable; successive saves may skip only the
            # identical snapshot. Latest recovery state is always rewritten.
            scratch=output/identity/variant;scratch.mkdir(parents=True,exist_ok=True)
            best=dict(model=cpu(model.state_dict()),step=6)
            latest=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best_checkpoint=best,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            checkpoint_times=[]
            for _ in range(3):
                def save():
                    base.save_checkpoint(scratch/'checkpoint_latest.pt',latest)
                    base.save_checkpoint(scratch/'checkpoint_best.pt',best)
                _,seconds=timed(save);checkpoint_times.append(seconds)
            if fingerprint(torch.load(scratch/'checkpoint_latest.pt',map_location='cpu',weights_only=False))!=fingerprint(latest):
                raise ValueError('Checkpoint roundtrip differs')
            result['variants'][variant]=dict(losses=losses,steps=steps,target_seconds=target_seconds,
                model_sha256=final_hash,optimizer_sha256=optimizer_hash,training_rng_sha256=after_training_rng,
                calibration=calibration,calibration_seconds=calibration_seconds,
                evaluation=evaluation,evaluation_seconds=evaluation_seconds,evaluation_rng_sha256=evaluation_rng,
                validation_batch_trials=batch_trials,
                checkpoint_seconds=checkpoint_times,resume_exact=True,
                statistics=deepcopy(acceleration.stats) if acceleration else {})
            write(output/(identity+'.json'),result)
        finally:
            if acceleration:acceleration.close()
    a,b=result['variants']['reference'],result['variants']['cached']
    for key in ('losses','model_sha256','optimizer_sha256','training_rng_sha256','evaluation','evaluation_rng_sha256','calibration'):
        if a[key]!=b[key]:raise ValueError('Exact equivalence failed: '+identity+'/'+key)
    if [s['gradient_sha256'] for s in a['steps']]!=[s['gradient_sha256'] for s in b['steps']]:
        raise ValueError('Gradient equivalence failed')
    eligible=[dict(batch_size=1,seconds=b['evaluation_seconds'])]+[v for v in b['validation_batch_trials'] if v['exact_metrics'] and v['exact_rng']]
    result.update(status='passed',exact_loss_gradient_optimizer_model_calibration_metrics_rng_resume=True,
        selected_validation_batch_size=min(eligible,key=lambda v:v['seconds'])['batch_size'],
        validation_batch_trials=b['validation_batch_trials'],
        evaluation_speedup=a['evaluation_seconds']/b['evaluation_seconds'],
        repeated_annotation_speedup=sum(a['target_seconds'][3:])/sum(b['target_seconds'][3:]),
        checkpoint_speedup=sum(a['checkpoint_seconds'][1:])/sum(b['checkpoint_seconds'][1:]))
    write(output/(identity+'.json'),result)
    print({k:v for k,v in result.items() if k!='variants'},flush=True)
    del model,initial,original_train,original_val,train,validation,latest,best,resume,optimizer,criterion
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    p=read(args.stage/'protocols/nnunet/protocol.json')
    if digest(p['data'])!=p['pkl_sha256']:raise ValueError('Dataset changed')
    results=[]
    for identity in MODELS:results.append(check_model(args.stage,args.output,identity))
    write(args.output/'summary.json',dict(status='passed',models=[{k:v for k,v in r.items() if k!='variants'} for r in results],
        scope='Runtime-only benchmark, not scientific results; six matched diagnostic updates per model with original losses and all four auxiliaries, all52 validation crops; no augmentation',
        requires_fresh_source_freeze_before_sweep=True))


if __name__=='__main__':main()
