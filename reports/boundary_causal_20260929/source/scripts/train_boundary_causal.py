#!/usr/bin/env python3
"""Matched 50-case loss/augmentation intervention using the frozen runtime."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from probe_edge_consistency import digest, save
from train_separated_edge import build_model, objectives, verify_payload
from train_pcgrad_50cases import loaders, make_scheduler


def spatial_metrics(probabilities, pred, truth):
    import numpy as np
    from scipy import ndimage as ndi
    from audit_edge_coherence import metrics, geometry
    p = probabilities[1:].sum(0)
    result = metrics(p, pred, truth)
    y, inner, outer = geometry(truth)
    hard = pred > 0
    swaps = (truth > 0) & (pred > 0) & (pred != truth)
    union_error = y != hard
    st = ndi.generate_binary_structure(3, 1)
    si = y & ~ndi.binary_erosion(y, st, border_value=0)
    so = ndi.binary_dilation(y, st) & ~y
    result.update(ap_swaps=int(swaps.sum()), all_wrong=int((pred != truth).sum()),
        boundary_union_errors=int((union_error & (inner | outer)).sum()),
        beyond_shell_union_errors=int((union_error & ~(inner | outer)).sum()),
        shell_ap_swaps=int((swaps & (inner | outer)).sum()),
        inner_layer1_fn=int((union_error & si).sum()),
        inner_layer2_fn=int((union_error & inner & ~si).sum()),
        outer_layer1_fp=int((union_error & so).sum()),
        outer_layer2_fp=int((union_error & outer & ~so).sum()),
        balanced_boundary_error=.5*((~hard & inner).sum()/inner.sum() + (hard & outer).sum()/outer.sum()))
    assert result['all_wrong'] == result['all_fp'] + result['all_fn'] + result['ap_swaps']
    assert result['boundary_union_errors'] == result['inner_fn'] + result['outer_fp']
    sp = hard & ~ndi.binary_erosion(hard, st, border_value=0)
    if sp.any():
        distance = np.concatenate([ndi.distance_transform_edt(~si)[sp], ndi.distance_transform_edt(~sp)[si]])
        result.update(assd_voxels=float(distance.mean()), hd95_voxels=float(np.percentile(distance, 95)),
            surface_dice_1voxel=float((distance <= 1).mean()))
    else:
        result.update(assd_voxels=None, hd95_voxels=None, surface_dice_1voxel=0.)
    return result


def audit(model, cohorts, config, device, output):
    import numpy as np
    import torch
    model.eval()
    _, objective = objectives(config)
    rows = []
    for split, loader in cohorts.items():
        for batch in loader:
            with torch.inference_mode(), torch.autocast('cuda', enabled=config['run']['amp']):
                logits = model(batch['image'].to(device))
            with torch.inference_mode():
                band = objective.bands(logits, batch['label'].to(device))
            p = logits[0].float().softmax(0).cpu().numpy()
            truth = batch['label'][0, 0].numpy().astype(np.uint8)
            result = spatial_metrics(p, p.argmax(0).astype(np.uint8), truth)
            result.update(bands_bce=float(band.details['case_loss'][0]),
                bands_inner_bce=float(band.details['inner_loss'][0]),
                bands_outer_bce=float(band.details['outer_loss'][0]), bands_truth=float(band.truth[0]))
            rows.append(dict(split=split, case_name=str(batch['case_name'][0]), metrics=result))
    save(output, rows)
    return rows


def fit(config, output, device):
    import torch
    from torch.utils.data import DataLoader
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.early_stopping import EarlyStopping
    from thesis.new_constraints.separated_edge import rule_losses
    from thesis.new_constraints.pcgrad import backward_tasks
    from thesis.new_constraints.training_augmentation import augment_mild
    run = config['run']
    trainer.seed_everything(run['seed'])
    train, val, dg, ag = loaders(config)
    model = build_model(config, device)
    optimizer, _ = trainer.build_optimizer_and_scheduler(run['optimizer_mode'], model, run['epochs'],
        adamw_gamma=run['adamw_gamma'], step_size=run['step_size'],
        learning_rate=run['learning_rate'], weight_decay=run['weight_decay'])
    scheduler = make_scheduler(optimizer, run)
    scaler = torch.cuda.amp.GradScaler(enabled=run['amp'])
    dice, bands = objectives(config)
    stopping = EarlyStopping(patience=run['early_stopping_patience'],
        min_delta=run['early_stopping_min_delta'], min_epochs=run['early_stopping_min_epochs'])
    projection_rng = torch.Generator().manual_seed(20260930 + run['seed'])
    best, best_epoch, start = -1., 0, 1
    latest = output/'checkpoint_latest.pt'
    if latest.exists():
        cp = torch.load(latest, map_location='cpu', weights_only=True)
        assert cp['run'] == run
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler'])
        scaler.load_state_dict(cp['scaler']); stopping = EarlyStopping(**cp['stopping_state'])
        trainer.restore_rng_state(cp['rng'], dg, ag); projection_rng.set_state(cp['projection_rng'])
        best, best_epoch, start = cp['best_hard_dice'], cp['best_epoch'], cp['epoch'] + 1
        del cp
    cohorts = {'train': DataLoader(train.dataset, batch_size=1, shuffle=False), 'validation': val}
    for epoch in range(start, run['epochs'] + 1):
        if stopping.should_stop:
            break
        started = time.perf_counter()
        model.train()
        sums = dict(total=0., dice=0., bands=0., edge=0.)
        aug_stats, skipped, order = {}, 0, []
        scale = trainer.constraint_warmup_scale(epoch, run['constraint_warmup_epochs'])
        for batch in train:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            order.append(str(batch['case_name'][0]))
            if run['training_augmentation'] == 'mild_v1':
                images, labels, stats = augment_mild(images, labels, generator=ag)
                for k, v in stats.items(): aug_stats[k] = aug_stats.get(k, 0) + v
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=run['amp']):
                logits = model(images)
                d = dice(logits, labels)
                if run['causal_objective'] == 'sum':
                    b = bands(model, images, logits, labels)['loss']
                    terms, counts = rule_losses(logits, labels)
                    assert all(bool((n > 0).all()) for n in counts.values())
                    e = sum(terms.values())/3
                    loss = d + scale*(b + run['edge_weight']*e)
                else:
                    b, e, loss = d.detach()*0, d.detach()*0, d
            assert torch.isfinite(loss)
            if run['causal_objective'] == 'sum':
                task_losses = [d, scale*b, *[scale*run['edge_weight']/3*terms[k] for k in ('inner', 'outer', 'cross')]]
                backward_tasks(task_losses, (p for p in model.parameters() if p.requires_grad),
                    scaler, projection_rng, project_conflicts=False)
            else:
                scaler.scale(loss).backward()
            before = float(scaler.get_scale())
            scaler.step(optimizer); scaler.update()
            skipped += int(float(scaler.get_scale()) < before)
            for k, v in dict(total=loss, dice=d, bands=b, edge=e).items(): sums[k] += float(v.detach())
        with torch.inference_mode():
            validation = trainer.evaluate_validation_metrics(model, val, bands, device, num_classes=3,
                amp=run['amp'], evaluate_constraints=False, all_translation_shifts=False, calibration_diagnostics=False)
        row = dict(epoch=epoch, optimizer_updates_seen=epoch*len(train), skipped_updates=skipped,
            **{'train_'+k:v/len(train) for k,v in sums.items()}, **validation,
            learning_rate=optimizer.param_groups[0]['lr'], constraint_scale=scale,
            augmentation=aug_stats, case_order=order, epoch_seconds=time.perf_counter()-started)
        trainer.validate_epoch_metrics(row)
        save(output/f'epochs/{epoch:03d}.json', row)
        print(json.dumps(dict(arm=run['causal_arm'], seed=run['seed'], **{k:v for k,v in row.items() if k!='case_order'})), flush=True)
        scheduler.step(validation['val_dice_hard'])
        improved = validation['val_dice_hard'] > best
        if improved: best, best_epoch = validation['val_dice_hard'], epoch
        stopping.update(epoch, validation['val_dice_hard'])
        terminal = stopping.should_stop or epoch == run['epochs']
        state = trainer.rng_state(dg, ag)
        if epoch in (15,30,45,60,75) or terminal:
            audit(model, cohorts, config, device, output/f'audits/epoch_{epoch:03d}.json')
        trainer.restore_rng_state(state, dg, ag)
        cp = dict(epoch=epoch, model=model.state_dict(), best_hard_dice=best, best_epoch=best_epoch,
            run=run, rng=state, projection_rng=projection_rng.get_state(), stopping_state=vars(stopping),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        if improved:
            trainer.save_checkpoint_payload(output/'checkpoint_best.pt', dict(epoch=epoch, model=model.state_dict(), run=run))
        trainer.save_checkpoint_payload(latest, cp)
    final_epoch = stopping.epoch
    trainer.save_checkpoint_payload(output/'checkpoint_final.pt', dict(epoch=final_epoch, model=model.state_dict(), run=run))
    selected = torch.load(output/'checkpoint_best.pt', map_location='cpu', weights_only=True)
    assert selected['epoch'] == best_epoch
    model.load_state_dict(selected['model'], strict=True)
    audit(model, cohorts, config, device, output/'selected_cases.json')
    save(output/'completion.json', dict(status='complete', selected_epoch=best_epoch, stopped_epoch=final_epoch,
        best_val_dice_hard=best, checkpoint_sha256=digest(output/'checkpoint_best.pt'),
        final_checkpoint_sha256=digest(output/'checkpoint_final.pt'),
        selected_cases_sha256=digest(output/'selected_cases.json'), run=run,
        checkpoint_storage='Selected and final inference weights retained; only this completed run resumable intermediate retired.'))
    latest.unlink()
    del model, optimizer, scheduler, scaler, selected, cp
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('/mnt/beegfsstudents/home/3160552'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arms', nargs='+', choices=['sum', 'dice', 'sum_aug', 'dice_aug'], required=True)
    args = p.parse_args()
    verify_payload()
    prior = args.base/'pcgrad_50cases_seed0_20260929_01'
    config = json.loads((prior/'fold0/seed0/pcgrad/config.json').read_text())
    done = json.loads((prior/'fold0/seed0/pcgrad/completion.json').read_text())
    assert digest(prior/'fold0/seed0/pcgrad/checkpoint_best.pt') == done['checkpoint_sha256']
    assert done['run'] == config['run']
    for key in ('pkl', 'splits_json'): assert digest(config[key]) == config[key+'_sha256']
    for name, h in json.loads((prior/'source/PAYLOAD.json').read_text())['files'].items():
        assert digest(REPO/name) == h, name
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.training_augmentation import MILD_V1
    from run_degree_bands_ab import check_quota
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    runtime = trainer.collect_runtime_provenance()
    assert runtime == config['runtime_provenance']
    execution = trainer.collect_execution_provenance(device)
    trainer.validate_calibration_execution(config['execution_provenance'], execution, allow_mig_profile_change=True)
    save(args.root/'PREFLIGHT.json', dict(status='passed', runtime=runtime, execution=execution,
        old_checkpoint_sha256=done['checkpoint_sha256'], job_id=os.getenv('SLURM_JOB_ID'),
        payload_sha256=digest(REPO/'PAYLOAD.json'), all_prior_source_files_unchanged=True))
    for arm in args.arms:
        candidate = copy.deepcopy(config)
        candidate['run'].update(seed=args.seed, gradient_method='sum', causal_arm=arm,
            translation_seed=args.seed+1, projection_seed=20260930+args.seed,
            causal_objective='sum' if arm.startswith('sum') else 'dice',
            training_augmentation='mild_v1' if arm.endswith('_aug') else 'none',
            constraint_set='causal_'+arm,
            augmentation_policy=MILD_V1 if arm.endswith('_aug') else None)
        candidate.update(experiment_payload_sha256=digest(REPO/'PAYLOAD.json'),
            train_samples=50, training_augmentation=MILD_V1 if arm.endswith('_aug') else None,
            execution_provenance=execution, training_script_sha256=digest(__file__),
            prior_config_sha256=digest(prior/'fold0/seed0/pcgrad/config.json'))
        output = args.root/f'seed{args.seed}'/arm
        output.mkdir(parents=True, exist_ok=True)
        if (output/'completion.json').exists():
            complete = json.loads((output/'completion.json').read_text())
            assert complete['run'] == candidate['run']
            assert digest(output/'checkpoint_best.pt') == complete['checkpoint_sha256']
            continue
        check_quota(output, 'start', 1.2)
        if (output/'config.json').exists(): assert json.loads((output/'config.json').read_text()) == candidate
        else: save(output/'config.json', candidate)
        fit(candidate, output, device)
    save(args.root/f'completion_seed{args.seed}.json', dict(status='complete', arms=args.arms,
        seed=args.seed, job_id=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()
