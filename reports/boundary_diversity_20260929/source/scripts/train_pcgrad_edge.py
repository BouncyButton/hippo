#!/usr/bin/env python3
"""Fold-0 PCGrad versus matched sum-of-task-gradients control; fixed coefficients."""
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
from train_separated_edge import (loaders,build_model,objectives,monitor,audit_model,
                                 verify_payload,reference,MONITOR_EPOCHS)
from probe_edge_consistency import digest,save

ARMS=('sum','pcgrad')


def load_reference(base,seed):
    old=base/'separated_edge_20260929_01'
    root=old/f'fold0/seed{seed}/separated'
    config=json.loads((root/'config.json').read_text())
    done=json.loads((root/'completion.json').read_text())
    assert done['status']=='complete' and done['run']==config['run']
    assert (config['run']['fold'],config['run']['seed'],config['run']['epochs'])==(0,seed,75)
    assert config['run']['edge_arm']=='separated' and config['run']['edge_rule_weights']==[1/3]*3
    assert digest(root/'checkpoint_best.pt')==done['checkpoint_sha256']
    assert digest(root/'selected_cases.json')==done['selected_cases_sha256']
    assert digest(old/'source/PAYLOAD.json')==config['experiment_payload_sha256']
    for name,expected in json.loads((old/'source/PAYLOAD.json').read_text())['files'].items():
        assert digest(REPO/name)==expected,name
    original,_,_=reference(base,0,seed)
    for k,v in original['run'].items():
        if k not in {'constraint_set','edge_weight','edge_calibration','edge_calibration_sha256'}:
            assert config['run'][k]==v,k
    for name in ('pkl','splits_json'):
        assert digest(config[name])==config[name+'_sha256']
    calibration=Path(config['run']['edge_calibration'])
    assert digest(calibration)==config['run']['edge_calibration_sha256']
    assert json.loads(calibration.read_text())['weights']['separated']==config['run']['edge_weight']
    return config,root,done


def all_case_audit(config,checkpoint,device,output):
    """FP32 diagnostic at the retained separated-control selected checkpoint."""
    import numpy as np
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.separated_edge import rule_losses
    from thesis.new_constraints.pcgrad import gram_matrix,TASKS
    trainer.seed_everything(config['run']['seed'])
    train,_,_,_=loaders(config)
    model=build_model(config,device)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(cp['model'],strict=True)
    model.eval()
    params=tuple(p for p in model.parameters() if p.requires_grad)
    dice,bands=objectives(config)
    rows=[]
    for i in sorted(range(len(train.dataset)),key=lambda i:train.dataset.data[i]['case_name']):
        batch=train.dataset[i]
        images,labels=batch['image'][None].to(device),batch['label'][None].to(device)
        z=model(images.float())
        terms,counts=rule_losses(z,labels)
        assert all(bool((v>0).all()) for v in counts.values())
        losses={'dice':dice(z,labels),'bands':bands(model,images,z,labels)['loss'],**terms}
        vectors=[]
        for j,loss in enumerate(losses.values()):
            g=torch.autograd.grad(loss,params,retain_graph=j<len(losses)-1,allow_unused=True)
            vectors.append(torch.cat([(v.detach().float() if v is not None else torch.zeros_like(p)).reshape(-1) for p,v in zip(params,g)]))
        gram=gram_matrix(torch.stack(vectors))
        norm=gram.diag().clamp_min(0).sqrt()
        cosine={}
        for a in range(len(TASKS)):
            for b in range(a+1,len(TASKS)):
                denom=float(norm[a]*norm[b])
                cosine[TASKS[a]+'__'+TASKS[b]]=float(gram[a,b])/denom if denom else None
        rows.append(dict(case_name=str(batch['case_name']),losses={k:float(v.detach()) for k,v in losses.items()},
            parameter_norms=dict(zip(TASKS,norm.tolist())),parameter_cosines=cosine))
        del vectors,g,z,losses,terms
    assert len(rows)==10
    count=sum(r['parameter_cosines']['inner__cross'] is not None and r['parameter_cosines']['inner__cross']<0 for r in rows)
    result=dict(status='complete',seed=config['run']['seed'],checkpoint_sha256=digest(checkpoint),selected_epoch=cp['epoch'],
        cases=rows,inner_cross_conflicting_cases=count,confirmed=count>=6,
        scope='All ten training cases at original selected separated-control checkpoint; FP32 eval-mode parameter gradients; majority conflict confirmation is a descriptive gate, not a significance test.')
    save(output,result)
    del model,cp
    torch.cuda.empty_cache()
    return result


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
    optimizer, scheduler = trainer.build_optimizer_and_scheduler(run['optimizer_mode'], model, run['epochs'],
        adamw_gamma=run['adamw_gamma'], step_size=run['step_size'], learning_rate=run['learning_rate'], weight_decay=run['weight_decay'])
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
        row = dict(epoch=epoch, **{'train_'+k: v/len(train) for k,v in sums.items()}, **validation,
                   learning_rate=optimizer.param_groups[0]['lr'], constraint_scale=scale,
                   epoch_seconds=time.perf_counter()-started)
        trainer.validate_epoch_metrics(row)
        save(output/f'epochs/{epoch:03d}.json', row)
        save(output/f'updates/{epoch:03d}.json',dict(epoch=epoch,method=run['gradient_method'],steps=step_reports))
        print(json.dumps(dict(fold=run['fold'], seed=run['seed'], arm=run['gradient_method'], **row)), flush=True)
        scheduler.step()
        improved = validation['val_dice_hard'] > best
        if improved:
            best, best_epoch = validation['val_dice_hard'], epoch
        stopping.update(epoch, validation['val_dice_hard'])
        terminal = stopping.should_stop or epoch == run['epochs']
        # Additional loaders/diagnostics must not consume the training RNG stream.
        state = trainer.rng_state(dg, ag)
        if epoch in MONITOR_EPOCHS or terminal:
            monitor(model, train.dataset, config, epoch, device, dg, ag, output)
            audit_model(model, {'validation': val}, config, device, output/f'coherence/epoch_{epoch:03d}.json')
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
        checkpoint_storage='Selected FP32 inference checkpoint retained; completed-arm optimizer state retired.'))
    # Only retire this experiment's own intermediate after checkpoint and audit are durable.
    latest.unlink()
    del model, optimizer, scheduler, scaler, selected, cp
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('/mnt/beegfsstudents/home/3160552'))
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--mode',choices=('preflight','audit','train'),required=True)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    verify_payload()
    config,old,done=load_reference(args.base,args.seed)
    save(args.root/'reference_binding.json',dict(directory=str(old),config_sha256=digest(old/'config.json'),
        checkpoint_sha256=done['checkpoint_sha256'],selected_epoch=done['selected_epoch']))
    if args.mode=='preflight':
        print(json.dumps(dict(status='preflight_passed',seed=args.seed)),flush=True)
        return
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from run_degree_bands_ab import check_quota
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    trainer.seed_everything(args.seed)
    runtime,execution=trainer.collect_runtime_provenance(),trainer.collect_execution_provenance(device)
    assert runtime==config['runtime_provenance']
    trainer.validate_calibration_execution(config['execution_provenance'],execution,allow_mig_profile_change=True)
    if args.mode=='audit':
        result=all_case_audit(config,old/'checkpoint_best.pt',device,args.root/'all_case_conflicts.json')
        print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
        return
    gate=json.loads((args.root.parent.parent/'CONFLICT_GATE.json').read_text())
    assert gate['passed'] and gate['seeds']==[0,1,2]
    for arm in ARMS:
        output=args.root/arm
        if (output/'completion.json').exists():
            completed=json.loads((output/'completion.json').read_text())
            assert completed['status']=='complete'
            assert digest(output/'checkpoint_best.pt')==completed['checkpoint_sha256']
            assert digest(output/'selected_cases.json')==completed['selected_cases_sha256']
            continue
        output.mkdir(exist_ok=True)
        check_quota(output,'start',.8)
        candidate=copy.deepcopy(config)
        candidate['run'].update(gradient_method=arm,constraint_set='bands_edge_separated_'+arm,
            projection_seed=20260930+args.seed,pcgrad_tasks=['dice','bands','inner','outer','cross'],
            projection_reduction='sum',pcgrad_order='independent_uniform_per_task_per_update')
        candidate.update(runtime_provenance=runtime,execution_provenance=execution,
            prior_reference_config_sha256=digest(old/'config.json'),experiment_payload_sha256=digest(REPO/'PAYLOAD.json'),
            training_script_sha256=digest(__file__))
        if (output/'config.json').exists():assert json.loads((output/'config.json').read_text())==candidate
        else:save(output/'config.json',candidate)
        fit(candidate,output,device)
    save(args.root/'completion.json',dict(status='complete',fold=0,seed=args.seed,arms=list(ARMS),job_id=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__':main()
