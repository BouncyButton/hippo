#!/usr/bin/env python3
"""Single fold-0/seed-0 PCGrad run: nested 50 cases and validation-driven LR."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from train_pcgrad_edge import load_reference
from train_separated_edge import (build_model,objectives,audit_model,verify_payload,
    MONITOR_EPOCHS,monitor as frozen_monitor)
from train_edge_lowdata import data_arguments
from probe_edge_consistency import digest,save


def make_scheduler(optimizer, run):
    import torch
    policy=run['lr_policy']
    assert policy['name']=='ReduceLROnPlateau' and policy['monitor']=='val_dice_hard'
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',
        factor=policy['factor'],patience=policy['patience'],threshold=policy['threshold'],
        threshold_mode='abs',min_lr=policy['min_lr'])


def loaders(config):
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    dg,ag,_=trainer.build_experiment_generators(config['run']['seed'])
    train,val,_,nt,nv=trainer.build_data(data_arguments(config),dg)
    assert (nt,nv)==(50,52)
    names={x['case_name'] for x in train.dataset.data}
    val_names={x['case_name'] for x in val.dataset.data}
    cohort=config['cohort']
    assert names==set(cohort['train']) and val_names==set(cohort['validation'])
    assert len(names)==50 and len(val_names)==52 and not names & val_names
    assert set(cohort['original_train_10']) <= names
    return train,val,dg,ag


def monitor(model,dataset,config,epoch,device,dg,ag,output):
    # Keep the same two FP32 training probes as the 10-case seed-0 run.
    ids={x['case_name']:i for i,x in enumerate(dataset.data)}
    selected=[ids[n] for n in sorted(config['cohort']['original_train_10'])[:2]]
    class ProbeCases:
        data=[dataset.data[i] for i in selected]
        def __getitem__(self,i):return dataset[selected[i]]
    return frozen_monitor(model,ProbeCases(),config,epoch,device,dg,ag,output)


def fit(config, output, device):
    import torch
    from torch.utils.data import DataLoader
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.early_stopping import EarlyStopping
    from thesis.new_constraints.separated_edge import rule_losses, edge_loss
    from thesis.new_constraints.pcgrad import backward_tasks, TASKS
    run = config['run']
    trainer.seed_everything(run['seed'])
    train, val, dg, ag = loaders(config)
    model = build_model(config, device)
    optimizer, _ = trainer.build_optimizer_and_scheduler(run['optimizer_mode'], model, run['epochs'],
        adamw_gamma=run['adamw_gamma'], step_size=run['step_size'], learning_rate=run['learning_rate'], weight_decay=run['weight_decay'])
    scheduler = make_scheduler(optimizer, run)
    scaler = torch.cuda.amp.GradScaler(enabled=run['amp'])
    dice, bands = objectives(config)
    edge = edge_loss(run['edge_arm'])
    stopping = EarlyStopping(patience=run['early_stopping_patience'], min_delta=run['early_stopping_min_delta'],
                             min_epochs=run['early_stopping_min_epochs'])
    projection_rng = torch.Generator().manual_seed(20260930 + run['seed'])
    best, best_epoch, start_epoch = -1., 0, 1
    latest = output/'checkpoint_latest.pt'
    if latest.exists():
        cp = torch.load(latest, map_location='cpu', weights_only=True)
        assert cp['run'] == run
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler']); scaler.load_state_dict(cp['scaler'])
        stopping = EarlyStopping(**cp['stopping_state'])
        best, best_epoch, start_epoch = cp['best_hard_dice'], cp['best_epoch'], cp['epoch'] + 1
        trainer.restore_rng_state(cp['rng'], dg, ag)
        projection_rng.set_state(cp['projection_rng'])
        del cp
    else:
        monitor(model, train.dataset, config, 0, device, dg, ag, output)
    cp = None
    for epoch in range(start_epoch, run['epochs'] + 1):
        if stopping.should_stop:
            break
        started = time.perf_counter()
        model.train()
        sums = dict(total=0., dice=0., bands=0., edge=0., inner=0., outer=0., cross=0.)
        scale = trainer.constraint_warmup_scale(epoch, run['constraint_warmup_epochs'])
        step_reports = []
        for batch_index, batch in enumerate(train):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=run['amp']):
                logits = model(images)
                d = dice(logits, labels)
                b = bands(model, images, logits, labels)['loss']
                terms, counts = rule_losses(logits, labels)
                assert all(bool((n > 0).all()) for n in counts.values())
                e = edge(logits, labels) if run['edge_arm'] == 'pooled' else sum(terms.values())/3
                loss = d + scale * (b + run['edge_weight'] * e)
            assert torch.isfinite(loss)
            task_losses = [d, scale*b, *[scale*run['edge_weight']/3*terms[k] for k in ('inner','outer','cross')]]
            step_report = backward_tasks(task_losses, (p for p in model.parameters() if p.requires_grad),
                scaler, projection_rng, project_conflicts=run['gradient_method']=='pcgrad')
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer); scaler.update()
            step_report.update(batch_index=batch_index,case_name=str(batch['case_name'][0]),
                optimizer_step_skipped=float(scaler.get_scale())<scale_before,
                amp_scale_after=float(scaler.get_scale()),tasks=list(TASKS))
            step_reports.append(step_report)
            for k,v in dict(total=loss, dice=d, bands=b, edge=e, **terms).items():
                sums[k] += float(v.detach())
        with torch.inference_mode():
            validation = trainer.evaluate_validation_metrics(model, val, bands, device, num_classes=3,
                amp=run['amp'], evaluate_constraints=False, all_translation_shifts=False, calibration_diagnostics=False)
        row = dict(epoch=epoch, optimizer_updates_seen=epoch*len(train), training_cases=len(train.dataset), **{'train_'+k: v/len(train) for k,v in sums.items()}, **validation,
                   learning_rate=optimizer.param_groups[0]['lr'], constraint_scale=scale,
                   epoch_seconds=time.perf_counter()-started)
        trainer.validate_epoch_metrics(row)
        save(output/f'epochs/{epoch:03d}.json', row)
        save(output/f'updates/{epoch:03d}.json',dict(epoch=epoch,method=run['gradient_method'],steps=step_reports))
        print(json.dumps(dict(fold=run['fold'], seed=run['seed'], arm=run['gradient_method'], **row)), flush=True)
        scheduler.step(validation['val_dice_hard'])
        improved = validation['val_dice_hard'] > best
        if improved:
            best, best_epoch = validation['val_dice_hard'], epoch
        stopping.update(epoch, validation['val_dice_hard'])
        terminal = stopping.should_stop or epoch == run['epochs']
        # Additional loaders/diagnostics must not consume the training RNG stream.
        state = trainer.rng_state(dg, ag)
        if epoch in MONITOR_EPOCHS or terminal:
            monitor(model, train.dataset, config, epoch, device, dg, ag, output)
            audit_model(model, {'train': DataLoader(train.dataset,batch_size=1,shuffle=False), 'validation': val}, config, device, output/f'coherence/epoch_{epoch:03d}.json')
        trainer.restore_rng_state(state, dg, ag)
        verify_payload()
        cp = dict(epoch=epoch, model=model.state_dict(), best_hard_dice=best, best_epoch=best_epoch,
            run=run, rng=state, projection_rng=projection_rng.get_state(), stopping_state=vars(stopping), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        if improved:
            trainer.save_checkpoint_payload(output/'checkpoint_best.pt', dict(epoch=epoch, model=model.state_dict(), run=run))
        trainer.save_checkpoint_payload(latest, cp)
    selected = torch.load(output/'checkpoint_best.pt', map_location='cpu', weights_only=True)
    assert selected['epoch'] == best_epoch
    model.load_state_dict(selected['model'], strict=True)
    audit_model(model, {'train': DataLoader(train.dataset,batch_size=1,shuffle=False), 'validation': val},
                config, device, output/'selected_cases.json')
    save(output/'completion.json', dict(status='complete', selected_epoch=best_epoch, stopped_epoch=stopping.epoch,
        best_val_dice_hard=best, checkpoint_sha256=digest(output/'checkpoint_best.pt'),
        selected_cases_sha256=digest(output/'selected_cases.json'), run=run,
        checkpoint_storage='Selected FP32 weights and final full resumable optimizer/RNG checkpoint retained.'))
    # Retain final optimizer/scheduler/AMP/RNG state for an explicitly authorized continuation.
    del model, optimizer, scheduler, scaler, selected, cp
    torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('/mnt/beegfsstudents/home/3160552'))
    p.add_argument('--mode',choices=['preflight','train'],required=True)
    args=p.parse_args()
    verify_payload()
    old_config,old,old_done=load_reference(args.base,0)
    prior=args.base/'pcgrad_edge_20260929_01/fold0/seed0/pcgrad'
    prior_config=json.loads((prior/'config.json').read_text())
    prior_done=json.loads((prior/'completion.json').read_text())
    assert prior_done['status']=='complete' and prior_done['run']==prior_config['run']
    assert digest(prior/'checkpoint_best.pt')==prior_done['checkpoint_sha256']
    assert digest(prior/'selected_cases.json')==prior_done['selected_cases_sha256']
    prior_source=args.base/'pcgrad_edge_20260929_01/source'
    assert digest(prior_source/'PAYLOAD.json')==prior_config['experiment_payload_sha256']
    for name,h in json.loads((prior_source/'PAYLOAD.json').read_text())['files'].items():
        assert digest(REPO/name)==h,name
    selection=json.loads((args.root/'SPLIT_SELECTION.json').read_text())
    split=args.root/'splits_50cases_seed0.json'
    assert digest(split)==selection['new_split_sha256']
    original=json.loads(Path(old_config['splits_json']).read_text())[0]
    actual=json.loads(split.read_text())[0]
    assert actual['val']==original['val'] and set(original['train'])<=set(actual['train'])
    assert len(actual['train'])==len(set(actual['train']))==50
    full=json.loads((args.root/'full_splits_reference.json').read_text())[0]
    assert digest(args.root/'full_splits_reference.json')==selection['full_split_sha256']
    assert set(actual['train'])<=set(full['train']) and full['val']==actual['val']
    candidate=copy.deepcopy(prior_config)
    candidate['splits_json']=str(split)
    candidate['splits_json_sha256']=digest(split)
    candidate['run'].update(splits_json_sha256=digest(split),training_examples=50,
        constraint_set='bands_edge_separated_pcgrad_50cases',
        lr_policy=dict(name='ReduceLROnPlateau',monitor='val_dice_hard',factor=.5,
            patience=3,threshold=.0005,min_lr=1e-6))
    candidate['cohort']=dict(train=actual['train'],validation=actual['val'],original_train_10=original['train'])
    candidate.update(experiment_payload_sha256=digest(REPO/'PAYLOAD.json'),
        training_script_sha256=digest(__file__),prior_pcgrad_config_sha256=digest(prior/'config.json'))
    assert candidate['run']['seed']==candidate['run']['fold']==0
    assert candidate['run']['initial_checkpoint'] is None
    assert candidate['run']['gradient_method']=='pcgrad' and candidate['run']['epochs']==75
    train,val,_,_=loaders(candidate)
    save(args.root/'PREFLIGHT.json',dict(status='passed',training_cases=len(train.dataset),
        validation_cases=len(val.dataset),original_cases_retained=10,seed=0,fold=0,
        prior_checkpoint_sha256=prior_done['checkpoint_sha256'],split_sha256=digest(split),
        payload_sha256=digest(REPO/'PAYLOAD.json'),training=False))
    if args.mode=='preflight':
        print(json.dumps(dict(preflight='passed',training_cases=50,validation_cases=52)),flush=True)
        return
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from run_degree_bands_ab import check_quota
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    runtime=trainer.collect_runtime_provenance();execution=trainer.collect_execution_provenance(device)
    assert runtime==old_config['runtime_provenance']
    trainer.validate_calibration_execution(old_config['execution_provenance'],execution,allow_mig_profile_change=True)
    candidate.update(runtime_provenance=runtime,execution_provenance=execution)
    output=args.root/'fold0/seed0/pcgrad';output.mkdir(parents=True,exist_ok=True)
    check_quota(output,'start',1.6)
    if (output/'completion.json').exists():
        done=json.loads((output/'completion.json').read_text())
        assert done['status']=='complete' and done['run']==candidate['run']
        assert digest(output/'checkpoint_best.pt')==done['checkpoint_sha256']
        return
    if (output/'config.json').exists():assert json.loads((output/'config.json').read_text())==candidate
    else:save(output/'config.json',candidate)
    fit(candidate,output,device)
    save(args.root/'completion.json',dict(status='complete',fold=0,seed=0,training_cases=50,
        arms=['pcgrad'],new_models=1,job_id=os.getenv('SLURM_JOB_ID')))

if __name__=='__main__':main()
